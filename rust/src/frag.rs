//! Fragment-stage context for the functional simulator: coefficient registers, pixel positions of the lanes of a
//! SIMD group, and sample positions. The hardware's rasterizer setup computes coefficients; the simulator takes them as
//! input (a text file, `agx_run --frag FILE`), like the uniforms of a compute dispatch.
//!
//! File format (one item per line, `#` starts a comment):
//!   size W H                     target size in pixels (sets the default group layout: 8x4-pixel blocks, row-major)
//!   cf I A B C                   coefficient register I: value(x, y) = A*x + B*y + C at pixel position (x, y)
//!   group G X0 Y0                SIMD group G covers the block whose top-left pixel is (X0, Y0) (overrides the default)
//!   samplepos S DX DY            offset inside the pixel of sample S (default 0.5 0.5)
//!   samples N                    samples per pixel (default 1)
//!   lane L X Y                   explicit pixel of lane L of every group (overrides the quad layout; integer pixel coords)
//!
//! Assumptions not yet verified on hardware (tasks/render-02-fragment-isa.md): the lane -> pixel layout (default: quads
//! of 2x2, four quads across), the f32 evaluation order (fma(A, x, fma(B, y, C))), centroid = center for covered pixels.
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

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
pub fn all_samples() -> u32 {
    (1u32 << samples().min(8)) - 1
}

/// size in bytes of one component of a tile format, None when the format is not modelled
pub fn format_bytes(fmt: u64) -> Option<usize> {
    Some(match fmt { 0 | 4 | 5 => 1, 1 | 3 | 6 | 7 => 2, 2 => 4, _ => return None })
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
