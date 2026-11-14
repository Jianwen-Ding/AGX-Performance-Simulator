//! Fragment-stage context for the functional simulator: coefficient registers, pixel positions of the lanes of a
//! SIMD group, and sample positions. The hardware's rasterizer setup computes coefficients; the simulator takes them as
//! input (a text file, `agx_run --frag FILE`), like the uniforms of a compute dispatch.
//!
//! File format (one item per line, `#` starts a comment):
//!   size W H                     target size in pixels (sets the default group layout: 8x4-pixel blocks, row-major)
//!   cf I A B C                   coefficient register I: value(x, y) = A*x + B*y + C at pixel position (x, y)
//!   group G X0 Y0                SIMD group G covers the block whose top-left pixel is (X0, Y0) (overrides the default)
//!   samplepos S DX DY            offset inside the pixel of sample S (default 0.5 0.5)
//!   coverage X Y MASK            samples of pixel (X, Y) the triangle covers (default: all); sr60 / sr124, initial live mask
//!   depthfunc F                  zs_emit depth compare: never less equal lequal greater notequal gequal always (default less)
//!   depthclear Z                 depth the buffer holds before any zs_emit (default 1.0)
//!   samples N                    samples per pixel (default 1)
//!   lane L X Y                   explicit pixel of lane L of every group (overrides the quad layout; integer pixel coords)
//!
//! Assumptions not yet verified on hardware (tasks/render-02-fragment-isa.md): the lane -> pixel layout (default: quads
//! of 2x2, four quads across), the f32 evaluation order (fma(A, x, fma(B, y, C))), centroid = center for covered pixels.
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

/// Empirical factor on the simulated core cycles of a dispatch that runs a fragment program (`agx_dispatch --frag`). Measured on hardware
/// (results/2026-11-14-frag-cost): a 1-varying smooth fragment shader with a dependent ALU loop of 16..512 iterations over 2048x2048
/// pixels costs 4.14 us per iteration, the compute issue model 3.30 us (x1.25); the 1 / 24 varying and smooth / flat shaders come out
/// x1.25..x1.38. With the factor the four sweep shaders are within 3-10% (without: 24-28% low). Cause unknown (the same loop as a
/// compute kernel was not measurable cleanly: the store dominates). Fitted on one shader family (a second one, two dependent fmadds per iteration, is still 28% low: results/2026-11-14-frag-cost): treat fragment ALU-loop predictions as -30% .. +5%.
pub const FRAGMENT_ISSUE_FACTOR: f64 = 1.25;

#[derive(Debug, Default, Clone)]
pub struct FragSetup {
    pub width: u32,
    pub height: u32,
    pub cf: Vec<[f32; 3]>,
    pub groups: Vec<(u32, u32, u32)>,
    pub sample_pos: Vec<[f32; 2]>,
    pub lanes: Vec<(u32, u32, u32)>,
    /// samples per pixel of the tile (default 1; sample_mask / st_tile address samples 0..samples)
    pub samples: u32,
    /// covered samples per pixel (default: all); what sr60 / sr124 read and the initial live mask of a pixel
    pub coverage: Vec<(u32, u32, u32)>,
    /// depth compare for zs_emit: pass when `new <func> stored` (default less) and the value the depth buffer holds before any write
    pub depth_func: String,
    pub depth_clear: Option<f32>,
}

static SETUP: OnceLock<FragSetup> = OnceLock::new();

pub fn install(s: FragSetup) -> Result<(), String> {
    SETUP.set(s).map_err(|_| "fragment setup installed twice".to_string())
}
pub fn get() -> Option<&'static FragSetup> {
    SETUP.get()
}

impl FragSetup {
    pub fn parse(text: &str) -> Result<FragSetup, String> {
        let mut s = FragSetup::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let w: Vec<&str> = line.split_whitespace().collect();
            let u = |i: usize| -> Result<u32, String> {
                w.get(i).ok_or(format!("line {}: missing field", n + 1))?.parse::<u32>().map_err(|e| format!("line {}: {}", n + 1, e))
            };
            let fl = |i: usize| -> Result<f32, String> {
                let t = w.get(i).ok_or(format!("line {}: missing field", n + 1))?;
                if let Some(h) = t.strip_prefix("0x") {
                    return u32::from_str_radix(h, 16).map(f32::from_bits).map_err(|e| format!("line {}: {}", n + 1, e));
                }
                t.parse::<f32>().map_err(|e| format!("line {}: {}", n + 1, e))
            };
            match w[0] {
                "size" => {
                    s.width = u(1)?;
                    s.height = u(2)?;
                }
                "cf" => {
                    let i = u(1)? as usize;
                    if s.cf.len() <= i {
                        s.cf.resize(i + 1, [0.0; 3]);
                    }
                    s.cf[i] = [fl(2)?, fl(3)?, fl(4)?];
                }
                "group" => s.groups.push((u(1)?, u(2)?, u(3)?)),
                "samplepos" => {
                    let i = u(1)? as usize;
                    if s.sample_pos.len() <= i {
                        s.sample_pos.resize(i + 1, [0.5; 2]);
                    }
                    s.sample_pos[i] = [fl(2)?, fl(3)?];
                }
                "coverage" => {
                    let t = w.get(3).ok_or(format!("line {}: missing mask", n + 1))?;
                    let m = if let Some(h) = t.strip_prefix("0x") { u32::from_str_radix(h, 16) } else { t.parse() }.map_err(|e| format!("line {}: {}", n + 1, e))?;
                    s.coverage.push((u(1)?, u(2)?, m));
                }
                "depthfunc" => s.depth_func = w.get(1).ok_or(format!("line {}: missing function", n + 1))?.to_string(),
                "depthclear" => s.depth_clear = Some(fl(1)?),
                "samples" => s.samples = u(1)?,
                "lane" => s.lanes.push((u(1)?, u(2)?, u(3)?)),
                other => return Err(format!("line {}: unknown item {}", n + 1, other)),
            }
        }
        Ok(s)
    }

    /// pixel (x, y) of `lane` in SIMD group `group`
    pub fn pixel(&self, group: u32, lane: usize) -> (u32, u32) {
        if let Some(&(_, x, y)) = self.lanes.iter().find(|l| l.0 as usize == lane) {
            return (x, y);
        }
        let (x0, y0) = match self.groups.iter().find(|g| g.0 == group) {
            Some(&(_, x, y)) => (x, y),
            None => {
                let per_row = self.width.div_ceil(8).max(1);
                (8 * (group % per_row), 4 * (group / per_row))
            }
        };
        let q = (lane / 4) as u32;
        (x0 + 2 * (q % 4) + (lane as u32 & 1), y0 + 2 * (q / 4) + ((lane as u32 >> 1) & 1))
    }

    /// value of coefficient register `i` at a position: fma(A, x, fma(B, y, C))
    pub fn plane(&self, i: usize, x: f32, y: f32) -> Result<f32, String> {
        let c = self.cf.get(i).ok_or_else(|| format!("coefficient register {} not set up (--frag)", i))?;
        Ok(c[0].mul_add(x, c[1].mul_add(y, c[2])))
    }
    pub fn sample(&self, s: usize) -> [f32; 2] {
        self.sample_pos.get(s).copied().unwrap_or([0.5; 2])
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;
    #[test]
    fn packed_formats() {
        // rgb10a2: 1.0 -> 1023, 0.5 -> 512 (511.5 ties to even), alpha 2 bits
        assert_eq!(pack_word(8, [1.0, 0.5, 0.0, 1.0]), 1023 | 512 << 10 | 3 << 30);
        assert_eq!(unpack_word(8, 1023 | 3 << 30)[3], 1.0);
        // srgba8: 0.5 linear -> 188 (0.7354 * 255), alpha linear
        assert_eq!(pack_word(10, [0.5, 0.0, 1.0, 0.5]) & 0xff, 188);
        assert_eq!(pack_word(10, [0.0, 0.0, 0.0, 0.5]) >> 24, 128);
        // r11g11b10f: 1.0 = exponent 15, mantissa 0; 0.5 = exponent 14; 65504 the largest finite (exp 30, mant all ones); negatives -> 0
        let w = pack_word(12, [1.0, 0.5, 2.0, 0.0]);
        assert_eq!((w & 0x7ff, w >> 11 & 0x7ff, w >> 22), (15 << 6, 14 << 6, 16 << 5));
        assert_eq!(unpack_word(12, w)[..3], [1.0, 0.5, 2.0]);
        assert_eq!(pack_word(12, [-1.0, 1e9, 0.0, 0.0]) & 0x7ff, 0);
        assert_eq!(pack_word(12, [-1.0, 1e9, 0.0, 0.0]) >> 11 & 0x7ff, 30 << 6 | 63);
        // 1.0 + 2^-7 (halfway, 6 mantissa bits): ties to even rounds down, rtz truncates; just above rounds up
        assert_eq!(pack_word(12, [1.0 + 2f64.powi(-7), 0.0, 0.0, 0.0]) & 0x7ff, 15 << 6);
        assert_eq!(pack_word(12, [1.0 + 2f64.powi(-7) + 2f64.powi(-10), 0.0, 0.0, 0.0]) & 0x7ff, 15 << 6 | 1);
        assert_eq!(pack_word(14, [1.0 + 2f64.powi(-7) + 2f64.powi(-10), 0.0, 0.0, 0.0]) & 0x7ff, 15 << 6);
        // rgb9e5: 1.0 -> exponent 16 (2^1), mantissa 256 / 512
        let w = pack_word(13, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!((w & 511, w >> 27), (256, 16));
        assert_eq!(unpack_word(13, w)[0], 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_and_eval() {
        let s = FragSetup::parse("size 16 8\ncf 1 0.5 0 1\ncf 0 1 2 3 # c\n").unwrap();
        assert_eq!(s.plane(1, 4.5, 2.5).unwrap(), 3.25);
        assert_eq!(s.plane(0, 1.0, 1.0).unwrap(), 6.0);
        assert!(s.plane(2, 0.0, 0.0).is_err());
        assert_eq!(s.pixel(0, 0), (0, 0));
        assert_eq!(s.pixel(0, 5), (3, 0));
        assert_eq!(s.pixel(1, 0), (8, 0));
        assert_eq!(s.pixel(2, 4), (2, 4 + 0));
    }
}

#[cfg(test)]
mod decode_tests {
    use crate::isa;
    #[test]
    fn captured_iterproj_operands() {
        let code = [0x61u8, 0x91, 0x01, 0, 0, 0, 0, 0];
        let n = isa::opcode_to_number(&code);
        let d = isa::get_instruction_descriptor(n).unwrap();
        let f = d.decode_fields(d.mask_instr(n));
        let ops = d.fields_to_operands(&f).unwrap();
        assert_eq!(d.name, "iterproj");
        // count 0 means four channels; the destination is 32-bit registers, I = cf1, J (the 1/w plane) = cf0
        assert!(matches!(&ops[0], crate::isa::operand::Operand::Tuple(t, _) if t.len() == 4));
        assert!(matches!(ops[1], crate::isa::operand::Operand::CF(1)));
        assert!(matches!(ops[2], crate::isa::operand::Operand::CF(0)));
    }
}

/// Tile (pixel-local) state of the simulated fragment pass: tile memory per (pixel, sample), the live / tested sample masks that
/// `sample_mask` and `zs_emit` maintain, the depth / stencil last emitted per sample, and the vertex outputs `st_var` stored.
/// Semantics from Mesa's documentation of `sample_mask` (asahi/compiler/agx_nir_lower_sample_mask.c): TARGET samples that are in
/// LIVE run the depth/stencil test (modelled as always passing) and stay live; TARGET samples not in LIVE are killed; a killed
/// sample ignores later `sample_mask` and `st_tile` writes. Tile memory is byte addressed (`pixel_offset` = the render target's
/// offset in the per-sample layout, Mesa agx_tilebuffer_offset_B). Not yet compared with hardware.
#[derive(Default, Debug)]
pub struct TileState {
    pub mem: BTreeMap<(u32, u32, u32), Vec<u8>>,
    pub live: HashMap<(u32, u32), u32>,
    pub tested: HashMap<(u32, u32), u32>,
    pub zs: BTreeMap<(u32, u32, u32), (Option<u32>, Option<u16>)>,
    pub uvs: BTreeMap<(u32, u32, u32), (u32, bool)>,
}
static TILE: Mutex<Option<TileState>> = Mutex::new(None);

pub fn with_tile<R>(f: impl FnOnce(&mut TileState) -> R) -> R {
    let mut g = TILE.lock().unwrap();
    f(g.get_or_insert_with(TileState::default))
}
pub fn samples() -> u32 {
    get().map_or(1, |s| s.samples.max(1))
}
/// samples of pixel (x, y) covered at the start of the fragment program
pub fn coverage(x: u32, y: u32) -> u32 {
    get().and_then(|s| s.coverage.iter().find(|c| c.0 == x && c.1 == y).map(|c| c.2)).unwrap_or_else(all_samples) & all_samples()
}
/// whether a depth value passes against the stored one
pub fn depth_passes(new: f32, stored: f32) -> bool {
    match get().map(|s| s.depth_func.as_str()).filter(|f| !f.is_empty()).unwrap_or("less") {
        "never" => false,
        "equal" => new == stored,
        "lequal" => new <= stored,
        "greater" => new > stored,
        "notequal" => new != stored,
        "gequal" => new >= stored,
        "always" => true,
        _ => new < stored,
    }
}
pub fn depth_clear() -> f32 {
    get().and_then(|s| s.depth_clear).unwrap_or(1.0)
}
pub fn all_samples() -> u32 {
    (1u32 << samples().min(8)) - 1
}

/// size in bytes of one component of a tile format, None when the format is not modelled
pub fn format_bytes(fmt: u64) -> Option<usize> {
    Some(match fmt { 0 | 4 | 5 => 1, 1 | 3 | 6 | 7 => 2, 2 => 4, 8 | 10 | 12..=15 => 4, _ => return None })
}
/// formats that pack all components into one 32-bit word (rgb10a2, srgba8, rg11b10f, rgb9e5 and their .rtz forms)
pub fn is_packed(fmt: u64) -> bool {
    matches!(fmt, 8 | 10 | 12..=15)
}

fn ufloat_from_f32(x: f32, mbits: u32, rtz: bool) -> u32 {
    let emax = 31u32;
    if x.is_nan() {
        return (emax << mbits) | 1;
    }
    if x <= 0.0 {
        return 0;
    }
    if x.is_infinite() {
        return emax << mbits;
    }
    let b = x.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = (b & 0x7f_ffff) as u64 | if (b >> 23) & 0xff != 0 { 0x80_0000 } else { 0 };
    // value = m * 2^(e - 15 - 23); the target has mbits mantissa bits and a minimum exponent of 1 (denormals below)
    let shift = if e >= 1 { 23 - mbits as i32 } else { 23 - mbits as i32 + (1 - e) };
    let (q, r) = if shift >= 64 { (0, m) } else { (m >> shift, m & ((1u64 << shift) - 1)) };
    let half = if shift > 0 { 1u64 << (shift - 1) } else { 0 };
    let q = if rtz || shift == 0 { q } else if r > half || (r == half && q & 1 == 1) { q + 1 } else { q };
    let (mut exp, mut mant) = if e >= 1 { (e as u64, q - (1u64 << mbits)) } else { (0, q) };
    if e >= 1 && q >= (2u64 << mbits) {
        exp += 1; // rounding carried into the exponent
        mant = 0;
    } else if e < 1 && q >= (1u64 << mbits) {
        exp = 1; // a denormal rounded up to the smallest normal
        mant = 0;
    }
    if exp >= emax as u64 {
        return ((emax - 1) << mbits) | ((1 << mbits) - 1); // saturate to the largest finite value
    }
    ((exp as u32) << mbits) | mant as u32
}
fn ufloat_to_f32(bits: u32, mbits: u32) -> f32 {
    let exp = (bits >> mbits) & 31;
    let mant = (bits & ((1 << mbits) - 1)) as f32 / (1u32 << mbits) as f32;
    match exp {
        0 => mant * 2f32.powi(-14),
        31 => if mant == 0.0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + mant) * 2f32.powi(exp as i32 - 15),
    }
}
fn srgb_encode(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}
fn srgb_decode(x: f64) -> f64 {
    if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}

/// the 32-bit word of a packed format from the components x, y, z, w (rg11b10f / rgb9e5 ignore w)
pub fn pack_word(fmt: u64, c: [f64; 4]) -> u32 {
    let un = |x: f64, n: f64| (x.clamp(0.0, 1.0) * n).round_ties_even() as u32;
    match fmt {
        8 => un(c[0], 1023.0) | un(c[1], 1023.0) << 10 | un(c[2], 1023.0) << 20 | un(c[3], 3.0) << 30,
        10 => un(srgb_encode(c[0]), 255.0) | un(srgb_encode(c[1]), 255.0) << 8 | un(srgb_encode(c[2]), 255.0) << 16 | un(c[3], 255.0) << 24,
        12 | 14 => {
            let rtz = fmt == 14;
            ufloat_from_f32(c[0] as f32, 6, rtz) | ufloat_from_f32(c[1] as f32, 6, rtz) << 11 | ufloat_from_f32(c[2] as f32, 5, rtz) << 22
        }
        _ => {
            // rgb9e5 (EXT_texture_shared_exponent): 9-bit mantissas under one 5-bit exponent; .rtz truncates
            let max = 65408.0f64;
            let v: Vec<f64> = c[..3].iter().map(|x| if x.is_nan() { 0.0 } else { x.clamp(0.0, max) }).collect();
            let m = v.iter().cloned().fold(0.0, f64::max);
            let mut e = if m > 0.0 { (m.log2().floor() as i32).max(-16) + 1 + 15 } else { 0 };
            let mut denom = 2f64.powi(e - 15 - 9);
            let rnd = |x: f64, d: f64| if fmt == 15 { (x / d).floor() } else { (x / d + 0.5).floor() };
            if rnd(m, denom) as u32 == 512 {
                e += 1;
                denom *= 2.0;
            }
            (rnd(v[0], denom) as u32) | (rnd(v[1], denom) as u32) << 9 | (rnd(v[2], denom) as u32) << 18 | (e as u32) << 27
        }
    }
}
pub fn unpack_word(fmt: u64, w: u32) -> [f32; 4] {
    match fmt {
        8 => [(w & 1023) as f32 / 1023.0, (w >> 10 & 1023) as f32 / 1023.0, (w >> 20 & 1023) as f32 / 1023.0, (w >> 30) as f32 / 3.0],
        10 => {
            let d = |b: u32| srgb_decode(b as f64 / 255.0) as f32;
            [d(w & 255), d(w >> 8 & 255), d(w >> 16 & 255), (w >> 24) as f32 / 255.0]
        }
        12 | 14 => [ufloat_to_f32(w & 0x7ff, 6), ufloat_to_f32(w >> 11 & 0x7ff, 6), ufloat_to_f32(w >> 22 & 0x3ff, 5), 1.0],
        _ => {
            let s = 2f32.powi((w >> 27) as i32 - 15 - 9);
            [(w & 511) as f32 * s, (w >> 9 & 511) as f32 * s, (w >> 18 & 511) as f32 * s, 1.0]
        }
    }
}
/// bytes of a component given as `x` (an fp value for the float / norm formats, raw bits for i8 / i16 / i32)
pub fn pack(fmt: u64, x: f64, raw: u32, f16_bits: impl Fn(f64) -> u32) -> u32 {
    let norm = |lo: f64, hi: f64, scale: f64| (x.clamp(lo, hi) * scale).round_ties_even() as i64 as u32;
    match fmt {
        0 | 1 | 2 => raw,
        3 => f16_bits(x),
        4 => norm(0.0, 1.0, 255.0),
        5 => norm(-1.0, 1.0, 127.0),
        6 => norm(0.0, 1.0, 65535.0),
        _ => norm(-1.0, 1.0, 32767.0),
    }
}
/// the f32 (or raw integer) value of a component read from tile memory
pub fn unpack(fmt: u64, bits: u32, f16_value: impl Fn(u32) -> f64) -> f32 {
    match fmt {
        0 | 1 | 2 => f32::from_bits(bits),
        3 => f16_value(bits) as f32,
        4 => (bits & 0xff) as f32 / 255.0,
        5 => ((bits as u8 as i8) as f32 / 127.0).max(-1.0),
        6 => (bits & 0xffff) as f32 / 65535.0,
        _ => ((bits as u16 as i16) as f32 / 32767.0).max(-1.0),
    }
}

/// `--tile-dump` text: tile memory, live masks, depth / stencil and vertex outputs
pub fn dump() -> String {
    with_tile(|t| {
        let mut o = String::new();
        for ((x, y, s), m) in &t.mem {
            o += &format!("tile {} {} {} {}\n", x, y, s, m.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        }
        let mut live: Vec<_> = t.live.iter().collect();
        live.sort();
        for ((x, y), m) in live {
            o += &format!("live {} {} {:#x} tested {:#x}\n", x, y, m, t.tested.get(&(*x, *y)).copied().unwrap_or(0));
        }
        for ((x, y, s), (z, st)) in &t.zs {
            o += &format!("zs {} {} {} z {} s {}\n", x, y, s, z.map_or("-".into(), |v| format!("{:#010x}", v)), st.map_or("-".into(), |v| v.to_string()));
        }
        for ((g, l, slot), (v, fin)) in &t.uvs {
            o += &format!("uvs {} {} {} {:#010x}{}\n", g, l, slot, v, if *fin { " final" } else { "" });
        }
        o
    })
}
