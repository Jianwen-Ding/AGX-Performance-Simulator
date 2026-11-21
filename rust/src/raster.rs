//! Render (raster) pass model: executes a `drm_asahi_cmd_render` the way the hardware's fixed-function front end does,
//! with the shader stages run by the functional simulator (`func`), and reports what a timing model needs.
//!
//! Input: the memory image and command buffer of one SUBMIT (a replay plan of `tests/drm_render_replay.c`, written by
//! `scripts/render_replay.py`, so the simulator sees exactly what the GPU was given). Field layouts are Mesa's
//! open genxml (`src/asahi/genxml/cmdbuf.xml`); semantics marked HW were checked by direct-ioctl replays of mutated
//! passes (`scripts/render_probes.py`, results/2026-11-21-render-ioctl).
//!
//! Pipeline:
//! 1. VDM control stream: barriers, PPP state records (pointer + size), VDM state (vertex pipeline, output count),
//!    index lists (draws), stream links (with return), terminate.
//! 2. Vertex stage per draw: the USC words of the vertex pipeline (uniforms, textures, samplers, preshader) and one
//!    lane per (instance, vertex); r4 = vertex id without the base, r5 = vertex id, r6 = instance id (Mesa agx_abi.h).
//!    `st_var` slot k of a lane is output word k of its vertex; words 0..3 = clip position.
//! 3. Primitive assembly (points / lines are counted, not rasterized), W clamp, perspective divide, viewport.
//! 4. Rasterization: vertices snapped to 1/256 pixel (assumption, `SUBPIXEL`), edge functions with a top-left rule,
//!    sample positions from ppp_multisamplectl (pixel center for 1x), scissor (ISP scissor array, `Scissor enable`),
//!    region clip (32x32-tile rectangle), framebuffer bounds; cull front/back from the signed area and `Front face CCW`.
//! 5. Depth / stencil (ISP) per sample: plane-interpolated screen z, front/back face functions (both words must pass),
//!    write unless disabled, stencil ops; buffers start at the clear values (isp_bgobjdepth, isp_bgobjvals).
//! 6. Fragment work in API order per primitive over 8x4-pixel SIMD groups (HW: quads of 2x2, four across), coefficient
//!    registers from the CF bindings (slot 0 = 1/w, then z if read, then user varyings; perspective = v/w planes).
//!    Background program over every processed tile first, end-of-tile program last.
use crate::frag::{self, FragSetup};
use crate::func::{self, Dispatch, Memory};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Sub-pixel precision of the rasterizer's vertex snapping (assumption: 8 bits; `CR PPP Control` Fixed point format = 1).
pub const SUBPIXEL: i64 = 256;

fn bits(raw: &[u8], start: usize, size: usize) -> u64 {
    let mut v = 0u128;
    let first = start / 8;
    let last = (start + size + 7) / 8;
    for (k, i) in (first..last).enumerate() {
        v |= (*raw.get(i).unwrap_or(&0) as u128) << (8 * k);
    }
    ((v >> (start % 8)) & ((1u128 << size) - 1)) as u64
}
fn f32b(raw: &[u8], start: usize) -> f32 {
    f32::from_bits(bits(raw, start, 32) as u32)
}
fn rd(mem: &Memory, va: u64, n: usize) -> Result<Vec<u8>, String> {
    mem.read_bytes(va, n)
}
fn rd32(mem: &Memory, va: u64) -> Result<u32, String> {
    Ok(mem.read(va, 4)? as u32)
}

// ------------------------------------------------------------------------------------------------- input
/// Memory image + command buffer of one submit, from a drm_render_replay plan file.
pub struct PassInput {
    pub mem: Memory,
    pub usc_base: u64,
    pub cmdbuf: Vec<u8>,
    /// regions the plan asks to dump after the run: (va, size, hardware output path)
    pub dumps: Vec<(u64, u64, String)>,
}

impl PassInput {
    pub fn from_plan(path: &str) -> Result<PassInput, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        let mut bos: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut binds = vec![];
        let (mut usc_base, mut cmdbuf, mut dumps) = (0u64, vec![], vec![]);
        let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|e| format!("{}: {}", s, e));
        let dec = |s: &str| s.parse::<u64>().map_err(|e| format!("{}: {}", s, e));
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            match w.first().copied() {
                Some("usc_base") => usc_base = hex(w[1])?,
                Some("bo") => {
                    bos.insert(dec(w[1])? as u32, vec![0u8; dec(w[2])? as usize]);
                }
                Some("fill") => {
                    let id = dec(w[1])? as u32;
                    let off = dec(w[2])? as usize;
                    let data = std::fs::read(w[3]).map_err(|e| format!("{}: {}", w[3], e))?;
                    let bo = bos.get_mut(&id).ok_or("fill of unknown bo")?;
                    if off + data.len() > bo.len() {
                        return Err("fill past bo".into());
                    }
                    bo[off..off + data.len()].copy_from_slice(&data);
                }
                Some("bind") => binds.push((dec(w[1])? as u32, hex(w[2])?, dec(w[3])?, dec(w[4])?)),
                Some("cmdbuf") => cmdbuf = std::fs::read(w[1]).map_err(|e| format!("{}: {}", w[1], e))?,
                Some("dump") => dumps.push((hex(w[1])?, dec(w[2])?, w[3].to_string())),
                _ => {}
            }
        }
        let mut mem = Memory::default();
        for (id, va, range, off) in binds {
            let bo = bos.get(&id).ok_or("bind of unknown bo")?;
            mem.map(va, bo[off as usize..(off + range) as usize].to_vec());
        }
        Ok(PassInput { mem, usc_base, cmdbuf, dumps })
    }

    /// render records of the command buffer: (offset of the payload, payload bytes)
    pub fn renders(&self) -> Vec<Vec<u8>> {
        let mut out = vec![];
        let mut off = 0;
        while off + 8 <= self.cmdbuf.len() {
            let ty = u16::from_le_bytes([self.cmdbuf[off], self.cmdbuf[off + 1]]);
            let size = u16::from_le_bytes([self.cmdbuf[off + 2], self.cmdbuf[off + 3]]) as usize;
            if ty == 0 {
                out.push(self.cmdbuf[off + 8..(off + 8 + size).min(self.cmdbuf.len())].to_vec());
            }
            off += 8 + size;
        }
        out
    }
}

// ------------------------------------------------------------------------------------------------- render command
/// The fields of `drm_asahi_cmd_render` the model uses (uapi layout, 240 bytes).
#[derive(Debug, Clone, Default)]
pub struct RenderCmd {
    pub flags: u32,
    pub vdm_base: u64,
    pub scissor_base: u64,
    pub dbias_base: u64,
    pub zls_ctrl: u64,
    pub depth_base: u64,
    pub stencil_base: u64,
    pub msaa_ctl: u64,
    pub sampler_heap: u64,
    pub ppp_ctrl: u32,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub sampler_count: u32,
    pub utile_w: u32,
    pub utile_h: u32,
    pub samples: u32,
    pub sample_size: u32,
    pub bg: (u32, u32),
    pub eot: (u32, u32),
    pub bg_depth: u32,
    pub bg_vals: u32,
}

impl RenderCmd {
    pub fn parse(r: &[u8]) -> Result<RenderCmd, String> {
        if r.len() < 240 {
            return Err(format!("render record of {} bytes (want 240)", r.len()));
        }
        let u32a = |o: usize| u32::from_le_bytes(r[o..o + 4].try_into().unwrap());
        let u64a = |o: usize| u64::from_le_bytes(r[o..o + 8].try_into().unwrap());
        let u16a = |o: usize| u16::from_le_bytes(r[o..o + 2].try_into().unwrap()) as u32;
        Ok(RenderCmd {
            flags: u32a(0),
            vdm_base: u64a(8),
            scissor_base: u64a(48),
            dbias_base: u64a(56),
            depth_base: u64a(72),
            stencil_base: u64a(96),
            zls_ctrl: u64a(120),
            msaa_ctl: u64a(128),
            sampler_heap: u64a(136),
            ppp_ctrl: u32a(144),
            width: u16a(148),
            height: u16a(150),
            layers: u16a(152),
            sampler_count: u16a(154),
            utile_w: r[156] as u32,
            utile_h: r[157] as u32,
            samples: r[158] as u32,
            sample_size: r[159] as u32,
            bg: (u32a(168), u32a(172)),
            eot: (u32a(176), u32a(180)),
            bg_depth: u32a(200),
            bg_vals: u32a(204),
        })
    }
    pub fn process_empty_tiles(&self) -> bool {
        self.flags & 2 != 0
    }
}

// ------------------------------------------------------------------------------------------------- USC pipelines
#[derive(Debug, Clone, Default)]
pub struct Pipeline {
    pub words_va: u64,
    pub code_va: u64,
    pub preshader_va: Option<u64>,
    pub uniforms: Vec<(u32, u32, u64)>,
    pub textures: Vec<(u32, u32, u64)>,
    pub samplers: Vec<(u32, u32, u64)>,
    pub regs: u32,
    pub frag_props: Option<u32>,
    pub shared: u32,
}

pub fn parse_usc(mem: &Memory, usc_base: u64, va: u64) -> Result<Pipeline, String> {
    let mut p = Pipeline { words_va: va, ..Default::default() };
    let mut a = va;
    for _ in 0..64 {
        let tag = mem.read(a, 1)? as u8;
        let n = match tag {
            0x38 => 8, 0x58 => 4, 0x88 => 2, 0x0d => 6, 0x1d | 0x3d => 8, 0x4d => 4, 0x8d => 4, 0x9d | 0xdd => 8,
            t => return Err(format!("USC word tag {:#x} at {:#x} not known", t, a)),
        };
        let w = rd(mem, a, n)?;
        match tag {
            0x1d | 0x3d => {
                let start = bits(&w, 8, 8) as u32 + if tag == 0x3d { 256 } else { 0 };
                let size = match bits(&w, 20, 6) { 0 => 64, s => s } as u32;
                p.uniforms.push((start, size, bits(&w, 26, 38) << 2));
            }
            0xdd => p.textures.push((bits(&w, 8, 8) as u32, bits(&w, 20, 7) as u32, bits(&w, 27, 36) << 3)),
            0x9d => p.samplers.push((bits(&w, 8, 8) as u32, bits(&w, 20, 7) as u32, bits(&w, 27, 36) << 3)),
            0x4d => p.shared = bits(&w, 0, 32) as u32,
            0x0d => p.code_va = usc_base + bits(&w, 16, 32),
            0x8d => p.regs = match bits(&w, 8, 5) { 0 => 256, r => 8 * r as u32 },
            0x58 => p.frag_props = Some(bits(&w, 0, 32) as u32),
            0x38 => p.preshader_va = Some(usc_base + bits(&w, 32, 32)),
            _ => {}
        }
        a += n as u64;
        if tag == 0x38 || tag == 0x88 {
            return Ok(p);
        }
    }
    Err(format!("USC words at {:#x} do not end", va))
}

/// code bytes from `va` to the end of its region (at most 256 KiB)
fn code_at(mem: &Memory, va: u64) -> Result<Vec<u8>, String> {
    for (b, d) in &mem.regions {
        if *b <= va && va < *b + d.len() as u64 {
            let o = (va - b) as usize;
            return Ok(d[o..d.len().min(o + (256 << 10))].to_vec());
        }
    }
    Err(format!("shader code at unmapped {:#x}", va))
}

/// Bind a pipeline's resources: uniforms (after its preshader), texture / sampler state slots. Returns the 32-bit uniforms.
fn bind(mem: &mut Memory, p: &Pipeline, stats: &mut Stats) -> Result<Vec<(usize, u32)>, String> {
    let mut halves = vec![None::<u16>; 512];
    for &(start, n, va) in &p.uniforms {
        let b = rd(mem, va, 2 * n as usize)?;
        for i in 0..n as usize {
            if let Some(h) = halves.get_mut(start as usize + i) {
                *h = Some(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
            }
        }
    }
    let mut uni: Vec<(usize, u32)> = (0..256)
        .filter(|u| halves[2 * u].is_some() || halves[2 * u + 1].is_some())
        .map(|u| (u, halves[2 * u].unwrap_or(0) as u32 | (halves[2 * u + 1].unwrap_or(0) as u32) << 16))
        .collect();
    for &(start, n, va) in &p.textures {
        for i in 0..n {
            mem.tex_state.insert(start + i, rd(mem, va + 24 * i as u64, 24)?);
        }
    }
    for &(start, n, va) in &p.samplers {
        for i in 0..n {
            mem.samp_state.insert(start + i, rd(mem, va + 8 * i as u64, 8)?);
        }
    }
    if let Some(pva) = p.preshader_va {
        let code = code_at(mem, pva)?;
        uni = func::run_preamble(&code, &uni, mem, 256)?.into_iter().filter(|&(_, v)| v != 0).collect();
        stats.preshaders += 1;
    }
    Ok(uni)
}

// ------------------------------------------------------------------------------------------------- PPP state
/// Latest value of every PPP record kind (raw words).
#[derive(Debug, Clone, Default)]
pub struct PppState {
    pub frag_control: Option<[u8; 4]>,
    pub front: Option<[u8; 4]>,
    pub front2: Option<[u8; 4]>,
    pub front_stencil: Option<[u8; 4]>,
    pub back: Option<[u8; 4]>,
    pub back2: Option<[u8; 4]>,
    pub back_stencil: Option<[u8; 4]>,
    pub dbias_scissor: Option<[u8; 4]>,
    pub region_clip: Option<[u8; 8]>,
    pub viewport: Option<[f32; 6]>,
    pub w_clamp: Option<f32>,
    pub output_select: Option<[u8; 4]>,
    pub cull: Option<[u8; 4]>,
    pub cull2: Option<[u8; 4]>,
    pub fs: Option<[u8; 16]>,
    pub output_size: Option<u32>,
}

fn take<const N: usize>(rec: &[u8], o: &mut usize) -> Result<[u8; N], String> {
    let v = rec.get(*o..*o + N).ok_or("PPP record overrun")?.try_into().unwrap();
    *o += N;
    Ok(v)
}

pub fn apply_ppp(st: &mut PppState, rec: &[u8]) -> Result<(), String> {
    let h = &rec[..4];
    let b = |i: usize| bits(h, i, 1) != 0;
    let mut o = 4;
    if b(0) { st.frag_control = Some(take::<4>(rec, &mut o)?); }
    if b(1) { take::<4>(rec, &mut o)?; } // fragment control 2: same struct, second copy (model: ignored)
    if b(2) { st.front = Some(take::<4>(rec, &mut o)?); }
    if b(3) { st.front2 = Some(take::<4>(rec, &mut o)?); }
    if b(4) { st.front_stencil = Some(take::<4>(rec, &mut o)?); }
    if b(5) { st.back = Some(take::<4>(rec, &mut o)?); }
    if b(6) { st.back2 = Some(take::<4>(rec, &mut o)?); }
    if b(7) { st.back_stencil = Some(take::<4>(rec, &mut o)?); }
    if b(8) { st.dbias_scissor = Some(take::<4>(rec, &mut o)?); }
    let vpc = bits(h, 12, 4) as usize + 1;
    if b(10) {
        st.region_clip = Some(take::<8>(rec, &mut o)?);
        for _ in 1..vpc { take::<8>(rec, &mut o)?; }
    }
    if b(11) {
        take::<4>(rec, &mut o)?;
        let v = take::<24>(rec, &mut o)?;
        st.viewport = Some(std::array::from_fn(|i| f32b(&v, 32 * i)));
        for _ in 1..vpc { take::<24>(rec, &mut o)?; }
    }
    if b(16) { st.w_clamp = Some(f32::from_le_bytes(take::<4>(rec, &mut o)?)); }
    if b(17) { st.output_select = Some(take::<4>(rec, &mut o)?); }
    if b(18) { take::<4>(rec, &mut o)?; }
    if b(19) { take::<4>(rec, &mut o)?; }
    if b(21) { st.cull = Some(take::<4>(rec, &mut o)?); }
    if b(22) { st.cull2 = Some(take::<4>(rec, &mut o)?); }
    if b(23) { st.fs = Some(take::<16>(rec, &mut o)?); }
    if b(24) { take::<4>(rec, &mut o)?; }
    if b(25) { take::<4>(rec, &mut o)?; }
    if b(26) { take::<4>(rec, &mut o)?; }
    if b(27) { st.output_size = Some(u32::from_le_bytes(take::<4>(rec, &mut o)?)); }
    if b(28) { take::<8>(rec, &mut o)?; }
    if o != rec.len() {
        return Err(format!("PPP record of {} bytes, fields cover {}", rec.len(), o));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------- VDM stream
#[derive(Debug, Clone)]
pub struct Draw {
    pub primitive: u32,
    pub indexed: Option<(u64, u32)>, // (index buffer VA, index size bytes)
    pub restart: Option<u32>,
    pub count: u32,
    pub instances: u32,
    pub start: u32,
    pub vs: Pipeline,
    pub vs_outputs: u32,
    pub ppp: PppState,
    pub flat_vdm: u32,
}

pub fn walk_vdm(mem: &Memory, usc_base: u64, base: u64) -> Result<(Vec<Draw>, Vec<String>), String> {
    let mut draws = vec![];
    let mut log = vec![];
    let mut ppp = PppState::default();
    let (mut vs, mut vs_out, mut restart, mut flat_vdm) = (None::<Pipeline>, 0u32, None::<u32>, 0u32);
    let mut stack = vec![];
    let mut va = base;
    for _ in 0..1_000_000 {
        let w0 = rd32(mem, va)?;
        match w0 >> 29 {
            0 => {
                let w = rd(mem, va, 8)?;
                let ptr = (bits(&w, 0, 8) << 32) | bits(&w, 32, 32);
                let words = bits(&w, 8, 8) as usize;
                apply_ppp(&mut ppp, &rd(mem, ptr, 4 * words)?)?;
                log.push(format!("ppp {:#x} {}w", ptr, words));
                va += 8;
            }
            1 => {
                log.push(format!("barrier {:#x}", w0));
                if w0 >> 27 & 1 != 0 {
                    match stack.pop() { Some(r) => { va = r; continue; } None => break }
                }
                va += 4;
            }
            2 => {
                let mut n = 4u64;
                let pres = |i: u32| w0 >> i & 1 != 0;
                if pres(0) { restart = Some(rd32(mem, va + n)?); n += 4; }
                if pres(1) { n += 4; }
                if pres(2) {
                    let w1 = rd32(mem, va + n)?;
                    vs = Some(parse_usc(mem, usc_base, usc_base + ((w1 >> 6) as u64) * 64)?);
                    n += 4;
                }
                if pres(3) { vs_out = rd32(mem, va + n)? & 0xff; n += 4; }
                if pres(4) { n += 4; }
                if pres(5) { flat_vdm = rd32(mem, va + n)? & 3; n += 4; }
                if pres(7) { n += 4; } else { n = (n + 7) & !7; }
                log.push(format!("vdm_state {:#x}", w0));
                va += n;
            }
            3 => {
                let mut n = 4u64;
                let pres = |i: u32| w0 >> i & 1 != 0;
                let mut ib_lo = None;
                let (mut count, mut inst, mut start) = (0u32, 1u32, 0u32);
                if pres(21) { ib_lo = Some(rd32(mem, va + n)?); n += 4; }
                if pres(22) { count = rd32(mem, va + n)?; n += 4; }
                if pres(23) { inst = rd32(mem, va + n)?; n += 4; }
                if pres(24) { start = rd32(mem, va + n)?; n += 4; }
                if pres(25) { n += 8; }
                if pres(26) { return Err("indirect draws are not modelled".into()); }
                if pres(20) { n += 4; }
                if pres(27) { n += 4; }
                let isz = 1u32 << (w0 >> 17 & 7);
                let indexed = ib_lo.map(|lo| ((((w0 & 0xff) as u64) << 32) | lo as u64, isz));
                draws.push(Draw {
                    primitive: w0 >> 8 & 0xff,
                    indexed,
                    restart: if w0 >> 16 & 1 != 0 { restart } else { None },
                    count,
                    instances: inst,
                    start,
                    vs: vs.clone().ok_or("draw before any vertex pipeline")?,
                    vs_outputs: vs_out,
                    ppp: ppp.clone(),
                    flat_vdm,
                });
                log.push(format!("draw prim {} count {} inst {} start {} indexed {:?}", w0 >> 8 & 0xff, count, inst, start, indexed));
                va += n;
            }
            4 => {
                let w = rd(mem, va, 8)?;
                let tgt = (bits(&w, 0, 8) << 32) | bits(&w, 32, 32);
                if bits(&w, 28, 1) != 0 {
                    stack.push(va + 8);
                }
                va = tgt;
            }
            6 => break,
            t => return Err(format!("VDM block type {} at {:#x}", t, va)),
        }
    }
    Ok((draws, log))
}

// ------------------------------------------------------------------------------------------------- statistics
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub draws: u64,
    pub vertices: u64,
    pub vertex_groups: u64,
    pub vertex_instructions: u64,
    pub primitives: u64,
    pub culled: u64,
    pub clipped_unsupported: u64,
    pub tiles: u64,
    pub tiles_processed: u64,
    pub bg_groups: u64,
    pub eot_groups: u64,
    pub fragment_groups: u64,
    pub fragment_lanes: u64,
    pub fragment_samples: u64,
    pub fragment_instructions: u64,
    pub bg_instructions: u64,
    pub eot_instructions: u64,
    pub depth_tested: u64,
    pub depth_passed: u64,
    pub preshaders: u64,
    /// SIMD groups per (tile, draw) with triangle merging: distinct 8x4 blocks a draw covers in a tile
    pub merged_fragment_groups: u64,
    /// per-draw vertex / fragment group counts and instruction counts (for the timing model)
    pub per_draw: Vec<DrawStats>,
}

#[derive(Debug, Clone, Default)]
pub struct DrawStats {
    pub vertices: u64,
    pub vs_instructions: u64,
    pub primitives: u64,
    pub fragment_groups: u64,
    pub merged_fragment_groups: u64,
    pub fragment_instructions: u64,
    pub covered_pixels: u64,
    pub pass_type: u32,
}

// ------------------------------------------------------------------------------------------------- geometry
#[derive(Debug, Clone, Copy)]
struct SVert {
    x: f64,
    y: f64,
    z: f64,
    inv_w: f64,
    fx: i64,
    fy: i64,
}

/// 2x2 sample positions of the 2x / 4x patterns (ppp_multisamplectl: 4-bit x, 4-bit y per sample in 1/16 pixel, as Mesa's
/// agx_default_sample_positions). Returns positions in 1/SUBPIXEL units inside the pixel.
fn sample_positions(ctl: u64, samples: u32) -> Vec<(i64, i64)> {
    if samples <= 1 {
        return vec![(SUBPIXEL / 2, SUBPIXEL / 2)];
    }
    (0..samples as usize)
        .map(|s| {
            let b = (ctl >> (8 * s)) & 0xff;
            (((b & 0xf) as i64) * SUBPIXEL / 16, ((b >> 4) as i64) * SUBPIXEL / 16)
        })
        .collect()
}

fn edge(a: (i64, i64), b: (i64, i64), p: (i64, i64)) -> i64 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// Top-left rule in a y-down screen with the triangle made positive-area: an edge owns its pixels when it is a top edge
/// (horizontal, interior below) or a left edge (interior to the right). (Assumption, D3D/GL convention; HW probe pending.)
fn owns(a: (i64, i64), b: (i64, i64)) -> bool {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    // positive area (edge(p0, p1, p2) > 0) is clockwise on a y-down screen: the top edge runs +x, left edges run -y
    (dy == 0 && dx > 0) || dy < 0
}

fn plane(p: [(f64, f64); 3], v: [f64; 3]) -> [f32; 3] {
    let (x0, y0) = p[0];
    let (x1, y1) = (p[1].0 - x0, p[1].1 - y0);
    let (x2, y2) = (p[2].0 - x0, p[2].1 - y0);
    let det = x1 * y2 - x2 * y1;
    if det == 0.0 {
        return [0.0, 0.0, v[0] as f32];
    }
    let (d1, d2) = (v[1] - v[0], v[2] - v[0]);
    let a = (d1 * y2 - d2 * y1) / det;
    let b = (x1 * d2 - x2 * d1) / det;
    [a as f32, b as f32, (v[0] - a * x0 - b * y0) as f32]
}

fn zs_pass(func: u64, new: f32, stored: f32) -> bool {
    match func {
        0 => false, 1 => new < stored, 2 => new == stored, 3 => new <= stored,
        4 => new > stored, 5 => new != stored, 6 => new >= stored, _ => true,
    }
}
fn stencil_pass(func: u64, r: u8, s: u8) -> bool {
    match func {
        0 => false, 1 => r < s, 2 => r == s, 3 => r <= s, 4 => r > s, 5 => r != s, 6 => r >= s, _ => true,
    }
}
fn stencil_op(op: u64, s: u8, r: u8) -> u8 {
    match op {
        0 => s, 1 => 0, 2 => r, 3 => s.saturating_add(1), 4 => s.saturating_sub(1), 5 => !s,
        6 => s.wrapping_add(1), _ => s.wrapping_sub(1),
    }
}

// ------------------------------------------------------------------------------------------------- the pass
pub struct Options {
    pub max_steps: u64,
    /// skip the shader stages (geometry and counts only)
    pub no_shading: bool,
}
impl Default for Options {
    fn default() -> Self {
        Options { max_steps: 50_000_000, no_shading: false }
    }
}

pub struct PassResult {
    pub stats: Stats,
    pub log: Vec<String>,
    pub mem: Memory,
    pub cmd: RenderCmd,
}

struct Fragment {
    x: u32,
    y: u32,
    mask: u32,
}

fn frag_dispatch(mem: &mut Memory, p: &Pipeline, uniforms: &[(usize, u32)], setup: FragSetup, blocks: &[(u32, u32)],
                 max_steps: u64) -> Result<u64, String> {
    if blocks.is_empty() {
        return Ok(0);
    }
    let mut s = setup;
    s.global_groups = true;
    s.group_map = blocks.iter().enumerate().map(|(i, &b)| (i as u32, b)).collect();
    frag::replace(Some(s));
    frag::reset_masks();
    let code = code_at(mem, p.code_va)?;
    let d = Dispatch {
        code,
        grid: [32 * blocks.len() as u32, 1, 1],
        wg: [32, 1, 1],
        shared_bytes: 0,
        uniforms: uniforms.to_vec(),
        num_registers: p.regs.max(8) as usize,
        max_steps,
        placement: 0,
        launch: Default::default(),
    };
    let r = func::run(&d, mem)?;
    Ok(r.instructions)
}

pub fn run_pass(input: &PassInput, opt: &Options) -> Result<PassResult, String> {
    let renders = input.renders();
    let raw = renders.first().ok_or("no render command in the command buffer")?;
    let cmd = RenderCmd::parse(raw)?;
    let mut mem = input.mem.clone();
    if cmd.sampler_heap != 0 {
        mem.sampler_heap = Some((cmd.sampler_heap, cmd.sampler_count));
    }
    let usc = input.usc_base;
    let (draws, mut log) = walk_vdm(&mem, usc, cmd.vdm_base)?;
    let mut stats = Stats { draws: draws.len() as u64, ..Default::default() };
    let (w, h) = (cmd.width, cmd.height);
    let (tw, th) = (cmd.utile_w.max(1), cmd.utile_h.max(1));
    let samples = cmd.samples.max(1);
    let spos = sample_positions(cmd.msaa_ctl, samples);
    let tiles_x = w.div_ceil(tw);
    let tiles_y = h.div_ceil(th);
    stats.tiles = (tiles_x * tiles_y) as u64;
    frag::reset_tile();
    frag::replace(None);

    // depth / stencil buffers (ISP), per (pixel, sample)
    let clear_z = if (cmd.zls_ctrl >> 25 & 3) == 2 { (cmd.bg_depth & 0xffff) as f32 / 65535.0 } else { f32::from_bits(cmd.bg_depth) };
    let npx = (w * h * samples) as usize;
    let mut zbuf = vec![clear_z; npx];
    let mut sbuf = vec![(cmd.bg_vals & 0xff) as u8; npx];
    let zi = |x: u32, y: u32, s: u32| ((y * w + x) * samples + s) as usize;

    let base_setup = FragSetup {
        width: w,
        height: h,
        samples,
        tile: Some((tw, th)),
        sample_pos: spos.iter().map(|&(x, y)| [x as f32 / SUBPIXEL as f32, y as f32 / SUBPIXEL as f32]).collect(),
        ..Default::default()
    };

    // per primitive fragment lists, in API order (draw, primitive)
    struct Work { draw: usize, setup: FragSetup, frags: Vec<Fragment> }
    let mut work: Vec<Work> = vec![];
    let mut tile_used: HashSet<(u32, u32)> = HashSet::new();
    let mut fs_pipes: Vec<Option<(Pipeline, Vec<(usize, u32)>)>> = vec![];

    for (di, d) in draws.iter().enumerate() {
        let mut ds = DrawStats::default();
        // ---- vertex stage
        let vs_uni = bind(&mut mem, &d.vs, &mut stats)?;
        let vcode = code_at(&mem, d.vs.code_va)?;
        let nv = d.count as u64 * d.instances as u64;
        let ids: Vec<(u32, u32, u32)> = (0..d.instances)
            .flat_map(|inst| (0..d.count).map(move |i| (inst, i)))
            .map(|(inst, i)| -> Result<(u32, u32, u32), String> {
                let idx = match d.indexed {
                    Some((ib, sz)) => mem.read(ib + i as u64 * sz as u64, sz as usize)? as u32,
                    None => i,
                };
                let vid = if d.indexed.is_some() { idx.wrapping_add(d.start) } else { d.start + i };
                Ok((idx, vid, inst))
            })
            .collect::<Result<_, _>>()?;
        let nout = d.vs_outputs.max(4) as usize;
        let mut outs: Vec<Vec<u32>> = vec![vec![0; nout]; ids.len()];
        if !opt.no_shading || true {
            for chunk0 in (0..ids.len()).step_by(1024) {
                let n = (ids.len() - chunk0).min(1024);
                let chunk: Vec<(u32, u32, u32)> = ids[chunk0..chunk0 + n].to_vec();
                let chunk = std::sync::Arc::new(chunk);
                let c2 = chunk.clone();
                func::set_lane_init(Some(std::sync::Arc::new(move |gid: u32, lane: usize| {
                    let i = 32 * gid as usize + lane;
                    match c2.get(i) {
                        Some(&(idx, vid, inst)) => vec![(4, idx), (5, vid), (6, inst)],
                        None => vec![],
                    }
                })));
                frag::replace(None);
                let disp = Dispatch {
                    code: vcode.clone(), grid: [n as u32, 1, 1], wg: [n as u32, 1, 1], shared_bytes: 0, uniforms: vs_uni.clone(),
                    num_registers: d.vs.regs.max(8) as usize, max_steps: opt.max_steps, placement: 0, launch: Default::default(),
                };
                let r = func::run(&disp, &mut mem);
                func::set_lane_init(None);
                let r = r.map_err(|e| format!("vertex shader (draw {}): {}", di, e))?;
                ds.vs_instructions += r.instructions;
                stats.vertex_groups += r.groups;
                for ((g, lane, slot), (v, _)) in frag::take_uvs() {
                    let i = chunk0 + 32 * g as usize + lane as usize;
                    if i < outs.len() && (slot as usize) < nout {
                        outs[i][slot as usize] = v;
                    }
                }
            }
        }
        ds.vertices = nv;
        stats.vertices += nv;
        stats.vertex_instructions += ds.vs_instructions;

        // ---- fragment pipeline of this draw
        let ppp = &d.ppp;
        let fsw = ppp.fs.ok_or("draw without a fragment shader record")?;
        let fpipe = parse_usc(&mem, usc, usc + (bits(&fsw, 32 + 6, 26) << 6))?;
        let cf_count = bits(&fsw, 16, 7) as usize;
        let cf_va = bits(&fsw, 64 + 2, 30) << 2;
        let mut cfb = vec![];
        if cf_count > 0 && cf_va != 0 {
            let r = rd(&mem, usc + cf_va, 4 + 4 * cf_count)?;
            for i in 0..cf_count {
                let b = &r[4 + 4 * i..8 + 4 * i];
                cfb.push((bits(b, 0, 2) as usize + 1, bits(b, 2, 3), bits(b, 5, 3), bits(b, 8, 8) as usize, bits(b, 16, 8) as usize));
            }
        }
        let fctl = ppp.frag_control.unwrap_or([0; 4]);
        let pass_type = bits(&fctl, 29, 3) as u32;
        ds.pass_type = pass_type;
        let scissor_en = bits(&fctl, 16, 1) != 0;
        let stencil_en = bits(&fctl, 18, 1) != 0;
        let cull = ppp.cull.unwrap_or([0; 4]);
        let (cull_front, cull_back) = (bits(&cull, 0, 1) != 0, bits(&cull, 1, 1) != 0);
        let depth_clip = bits(&cull, 10, 1) != 0;
        let front_ccw = bits(&cull, 16, 1) != 0;
        let raster_discard = bits(&cull, 17, 1) != 0 || ppp.cull2.is_some_and(|c| bits(&c, 17, 1) != 0);
        let clamp_w = ppp.cull2.is_some_and(|c| bits(&c, 5, 1) != 0) && (cmd.ppp_ctrl >> 1 & 1) != 0;
        let vp = ppp.viewport.unwrap_or([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        // scissor rectangle (pixels, max exclusive) and region clip
        let mut rect = (0u32, 0u32, w, h);
        if scissor_en {
            let idx = ppp.dbias_scissor.map_or(0, |b| bits(&b, 0, 16)) as u64;
            let s = rd(&mem, cmd.scissor_base + 16 * idx, 16)?;
            let (maxx, minx, maxy, miny) = (bits(&s, 0, 16) as u32, bits(&s, 16, 16) as u32, bits(&s, 32, 16) as u32, bits(&s, 48, 16) as u32);
            rect = (rect.0.max(minx), rect.1.max(miny), rect.2.min(maxx), rect.3.min(maxy));
        }
        if let Some(rc) = ppp.region_clip {
            if bits(&rc, 31, 1) != 0 {
                let (maxx, minx, maxy, miny) = (bits(&rc, 0, 9) as u32 + 1, bits(&rc, 16, 9) as u32, bits(&rc, 32, 9) as u32 + 1, bits(&rc, 48, 9) as u32);
                rect = (rect.0.max(minx * tw), rect.1.max(miny * th), rect.2.min(maxx * tw), rect.3.min(maxy * th));
            }
        }
        let faces = |back: bool| -> ([u8; 4], Option<[u8; 4]>, [u8; 4]) {
            if back {
                (ppp.back.unwrap_or([0, 0, 0, 7]), ppp.back2, ppp.back_stencil.unwrap_or([0xff, 0xff, 0, 7 << 1]))
            } else {
                (ppp.front.unwrap_or([0, 0, 0, 7]), ppp.front2, ppp.front_stencil.unwrap_or([0xff, 0xff, 0, 7 << 1]))
            }
        };
        let fprops = fpipe.frag_props.unwrap_or(0);
        let _early_z = fprops >> 8 & 1 != 0;

        // ---- primitive assembly
        let tris: Vec<[usize; 3]> = assemble(d, &ids)?;
        let prov = |t: &[usize; 3], k: u64| -> usize { t[match k { 0 => 0, 6 => 1, _ => 2 }] };
        let fs_bind = if opt.no_shading { None } else { Some(bind(&mut mem, &fpipe, &mut stats)?) };
        fs_pipes.push(fs_bind.map(|u| (fpipe.clone(), u)));
        for t in &tris {
            ds.primitives += 1;
            stats.primitives += 1;
            if raster_discard {
                continue;
            }
            let clip: Vec<[f32; 4]> = t.iter().map(|&i| std::array::from_fn(|k| f32::from_bits(outs[i][k]))).collect();
            let mut sv = vec![];
            for c in &clip {
                let mut wv = c[3] as f64;
                if clamp_w {
                    wv = wv.max(ppp.w_clamp.unwrap_or(1e-6) as f64);
                }
                if wv <= 0.0 {
                    break;
                }
                let iw = 1.0 / wv;
                let (nx, ny, nz) = (c[0] as f64 * iw, c[1] as f64 * iw, c[2] as f64 * iw);
                let x = vp[0] as f64 + vp[1] as f64 * nx;
                let y = vp[2] as f64 + vp[3] as f64 * ny;
                let z = vp[4] as f64 + vp[5] as f64 * nz;
                sv.push(SVert { x, y, z, inv_w: iw, fx: (x * SUBPIXEL as f64).round() as i64, fy: (y * SUBPIXEL as f64).round() as i64 });
            }
            if sv.len() < 3 {
                stats.clipped_unsupported += 1;
                continue;
            }
            let p = [(sv[0].fx, sv[0].fy), (sv[1].fx, sv[1].fy), (sv[2].fx, sv[2].fy)];
            let area = edge(p[0], p[1], p[2]);
            if area == 0 {
                stats.culled += 1;
                continue;
            }
            // y-down screen: positive area here is clockwise on screen = counter-clockwise in GL's y-up window space.
            let ccw = area > 0;
            let back = ccw != front_ccw;
            if (back && cull_back) || (!back && cull_front) {
                stats.culled += 1;
                continue;
            }
            let (q, order) = if area > 0 { (p, [0usize, 1, 2]) } else { ([p[0], p[2], p[1]], [0usize, 2, 1]) };
            let _ = order;
            let own = [owns(q[0], q[1]), owns(q[1], q[2]), owns(q[2], q[0])];
            let bx0 = (p.iter().map(|v| v.0).min().unwrap().div_euclid(SUBPIXEL)).max(rect.0 as i64);
            let by0 = (p.iter().map(|v| v.1).min().unwrap().div_euclid(SUBPIXEL)).max(rect.1 as i64);
            let bx1 = (p.iter().map(|v| v.0).max().unwrap().div_euclid(SUBPIXEL) + 1).min(rect.2 as i64);
            let by1 = (p.iter().map(|v| v.1).max().unwrap().div_euclid(SUBPIXEL) + 1).min(rect.3 as i64);
            let pts = [(sv[0].x, sv[0].y), (sv[1].x, sv[1].y), (sv[2].x, sv[2].y)];
            let zpl = plane(pts, [sv[0].z, sv[1].z, sv[2].z]);
            let (face, face2, sten) = faces(back);
            let zfunc = bits(&face, 24, 3);
            let zfunc2 = face2.map_or(7, |f| bits(&f, 24, 3));
            let zwrite = !(bits(&face, 21, 1) != 0 && face2.is_none_or(|f| bits(&f, 21, 1) != 0));
            let sref = face[0];
            let (swm, srm) = (sten[0], sten[1]);
            let (s_dpass, s_dfail, s_sfail, s_cmp) = (bits(&sten, 16, 3), bits(&sten, 19, 3), bits(&sten, 22, 3), bits(&sten, 25, 3));
            let mut frags = vec![];
            for py in by0.max(0)..by1 {
                for px in bx0.max(0)..bx1 {
                    let mut mask = 0u32;
                    for (si, &(sx, sy)) in spos.iter().enumerate() {
                        let s = (px * SUBPIXEL + sx, py * SUBPIXEL + sy);
                        let e = [edge(q[0], q[1], s), edge(q[1], q[2], s), edge(q[2], q[0], s)];
                        let inside = (0..3).all(|k| e[k] > 0 || (e[k] == 0 && own[k]));
                        if !inside {
                            continue;
                        }
                        let fx = px as f32 + sx as f32 / SUBPIXEL as f32;
                        let fy = py as f32 + sy as f32 / SUBPIXEL as f32;
                        let z = zpl[0].mul_add(fx, zpl[1].mul_add(fy, zpl[2]));
                        if depth_clip && !(0.0..=1.0).contains(&z) {
                            continue;
                        }
                        let k = zi(px as u32, py as u32, si as u32);
                        stats.depth_tested += 1;
                        let sval = sbuf[k];
                        if stencil_en && !stencil_pass(s_cmp, sref & srm, sval & srm) {
                            sbuf[k] = (sval & !swm) | (stencil_op(s_sfail, sval, sref) & swm);
                            continue;
                        }
                        let zp = zs_pass(zfunc, z, zbuf[k]) && zs_pass(zfunc2, z, zbuf[k]);
                        if stencil_en {
                            let op = if zp { s_dpass } else { s_dfail };
                            sbuf[k] = (sval & !swm) | (stencil_op(op, sval, sref) & swm);
                        }
                        if !zp {
                            continue;
                        }
                        if zwrite {
                            zbuf[k] = z;
                        }
                        stats.depth_passed += 1;
                        mask |= 1 << si;
                    }
                    if mask != 0 {
                        frags.push(Fragment { x: px as u32, y: py as u32, mask });
                        tile_used.insert((px as u32 / tw, py as u32 / th));
                    }
                }
            }
            if frags.is_empty() {
                continue;
            }
            ds.covered_pixels += frags.len() as u64;
            // coefficient registers
            let mut setup = base_setup.clone();
            let ncf = cfb.iter().map(|b| b.4 + b.0).max().unwrap_or(0);
            setup.cf = vec![[0.0; 3]; ncf];
            for &(comps, model, source, slot, reg) in &cfb {
                for c in 0..comps {
                    let vals: [f64; 3] = std::array::from_fn(|k| {
                        let vi = t[k];
                        match (source, slot + c) {
                            (1, _) => sv[k].z,
                            (_, 0) => sv[k].inv_w,
                            (_, s) => {
                                // slots: 0 = W, [1 = Z when the shader reads z], then user varyings at UVS word 4 + ...
                                let user_base = cfb.iter().any(|b| b.2 == 1) as usize + 1;
                                let word = 4 + s - user_base;
                                f32::from_bits(*outs[vi].get(word).unwrap_or(&0)) as f64
                            }
                        }
                    });
                    let cf = match model {
                        7 => plane(pts, std::array::from_fn(|k| vals[k] * sv[k].inv_w)),
                        3 => plane(pts, vals),
                        k => {
                            let pv = prov(t, k);
                            let ki = t.iter().position(|&v| v == pv).unwrap();
                            [0.0, 0.0, vals[ki] as f32]
                        }
                    };
                    if reg + c < setup.cf.len() {
                        setup.cf[reg + c] = cf;
                    }
                }
            }
            setup.depth_func = "always".into();
            work.push(Work { draw: di, setup, frags });
        }
        stats.per_draw.push(ds);
    }

    // ---- fragment stage: background, primitives in API order, end of tile
    let processed: Vec<(u32, u32)> = (0..tiles_y)
        .flat_map(|ty| (0..tiles_x).map(move |tx| (tx, ty)))
        .filter(|t| cmd.process_empty_tiles() || tile_used.contains(t))
        .collect();
    stats.tiles_processed = processed.len() as u64;
    let tile_blocks = |tiles: &[(u32, u32)]| -> Vec<(u32, u32)> {
        let mut v = vec![];
        for &(tx, ty) in tiles {
            for by in (ty * th..((ty + 1) * th).min(h)).step_by(4) {
                for bx in (tx * tw..((tx + 1) * tw).min(w)).step_by(8) {
                    v.push((bx, by));
                }
            }
        }
        v
    };
    let all_blocks = tile_blocks(&processed);
    let full_cov = |blocks: &[(u32, u32)]| -> HashMap<(u32, u32), u32> {
        let mut m = HashMap::new();
        for &(bx, by) in blocks {
            for y in by..by + 4 {
                for x in bx..bx + 8 {
                    m.insert((x, y), if x < w && y < h { (1u32 << samples) - 1 } else { 0 });
                }
            }
        }
        m
    };
    if !opt.no_shading {
        let bgp = parse_usc(&mem, usc, usc + (cmd.bg.0 as u64 & !63))?;
        let bgu = bind(&mut mem, &bgp, &mut stats)?;
        let mut s = base_setup.clone();
        s.coverage_map = full_cov(&all_blocks);
        stats.bg_instructions = frag_dispatch(&mut mem, &bgp, &bgu, s, &all_blocks, opt.max_steps).map_err(|e| format!("background program: {}", e))?;
        stats.bg_groups = all_blocks.len() as u64;
    }
    for wk in work {
        let mut blocks: BTreeMap<(u32, u32), ()> = BTreeMap::new();
        let mut cov: HashMap<(u32, u32), u32> = HashMap::new();
        for f in &wk.frags {
            blocks.insert((f.x & !7, f.y & !3), ());
            cov.insert((f.x, f.y), f.mask);
            stats.fragment_samples += f.mask.count_ones() as u64;
        }
        let blocks: Vec<(u32, u32)> = blocks.into_keys().collect();
        for &(bx, by) in &blocks {
            for y in by..by + 4 {
                for x in bx..bx + 8 {
                    cov.entry((x, y)).or_insert(0);
                }
            }
        }
        stats.fragment_groups += blocks.len() as u64;
        stats.fragment_lanes += wk.frags.len() as u64;
        stats.per_draw[wk.draw].fragment_groups += blocks.len() as u64;
        if let Some((fp, fu)) = fs_pipes[wk.draw].clone() {
            let mut s = wk.setup;
            s.coverage_map = cov;
            let n = frag_dispatch(&mut mem, &fp, &fu, s, &blocks, opt.max_steps).map_err(|e| format!("fragment shader (draw {}): {}", wk.draw, e))?;
            stats.fragment_instructions += n;
            stats.per_draw[wk.draw].fragment_instructions += n;
        }
    }
    // merged groups: distinct (tile, block) per draw, which is what triangle merging can reach at best
    if !opt.no_shading {
        let eop = parse_usc(&mem, usc, usc + (cmd.eot.0 as u64 & !63))?;
        let eou = bind(&mut mem, &eop, &mut stats)?;
        let mut s = base_setup.clone();
        s.coverage_map = full_cov(&all_blocks);
        stats.eot_instructions = frag_dispatch(&mut mem, &eop, &eou, s, &all_blocks, opt.max_steps).map_err(|e| format!("end-of-tile program: {}", e))?;
        stats.eot_groups = all_blocks.len() as u64;
    }
    frag::replace(None);
    log.push(format!("tiles {}x{} processed {}", tiles_x, tiles_y, stats.tiles_processed));
    Ok(PassResult { stats, log, mem, cmd })
}

/// Triangles of a draw as indices into its shaded vertex list (lanes in (instance, vertex) order).
fn assemble(d: &Draw, ids: &[(u32, u32, u32)]) -> Result<Vec<[usize; 3]>, String> {
    let n = d.count as usize;
    let mut out = vec![];
    for inst in 0..d.instances as usize {
        let base = inst * n;
        // split at primitive restart
        let mut runs: Vec<Vec<usize>> = vec![vec![]];
        for i in 0..n {
            if d.restart.is_some_and(|r| d.indexed.is_some() && ids[base + i].0 == r) {
                runs.push(vec![]);
            } else {
                runs.last_mut().unwrap().push(base + i);
            }
        }
        for r in runs {
            match d.primitive {
                6 => out.extend(r.chunks_exact(3).map(|c| [c[0], c[1], c[2]])),
                9 => {
                    for i in 2..r.len() {
                        out.push(if i % 2 == 0 { [r[i - 2], r[i - 1], r[i]] } else { [r[i - 1], r[i - 2], r[i]] });
                    }
                }
                11 => {
                    for i in 2..r.len() {
                        out.push([r[i - 1], r[i], r[0]]);
                    }
                }
                0 | 1 | 3 | 5 => {} // points / lines: not rasterized by this model (counted as draws only)
                p => return Err(format!("primitive {} not modelled", p)),
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plane_fit() {
        let p = plane([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)], [1.0, 2.0, 3.0]);
        assert_eq!(p, [0.1, 0.2, 1.0]);
    }
    #[test]
    fn top_left_shared_edge() {
        // two triangles sharing the diagonal of a 4x4 square: every pixel center exactly once
        let s = SUBPIXEL;
        let a = [(0, 0), (4 * s, 0), (0, 4 * s)];
        let b = [(4 * s, 0), (4 * s, 4 * s), (0, 4 * s)];
        let mut hits = vec![0; 16];
        for t in [a, b] {
            let q = if edge(t[0], t[1], t[2]) > 0 { t } else { [t[0], t[2], t[1]] };
            let own = [owns(q[0], q[1]), owns(q[1], q[2]), owns(q[2], q[0])];
            for y in 0..4 {
                for x in 0..4 {
                    let p = (x * s + s / 2, y * s + s / 2);
                    let e = [edge(q[0], q[1], p), edge(q[1], q[2], p), edge(q[2], q[0], p)];
                    if (0..3).all(|k| e[k] > 0 || (e[k] == 0 && own[k])) {
                        hits[(y * 4 + x) as usize] += 1;
                    }
                }
            }
        }
        assert!(hits.iter().all(|&h| h == 1), "{:?}", hits);
    }
}

// ------------------------------------------------------------------------------------------------- timing
/// Predicted firmware vertex / fragment times of a pass (ns) with the parts they are made of.
#[derive(Debug, Clone, Default)]
pub struct Timing {
    pub vtx_ns: f64,
    pub frag_ns: f64,
    pub parts: BTreeMap<String, f64>,
}

/// Timing model of a render pass (placeholder constants until calibrated: results/2026-11-21-render-ioctl).
pub fn timing(r: &PassResult) -> Timing {
    let s = &r.stats;
    let mut t = Timing::default();
    let mut put = |k: &str, v: f64| { t.parts.insert(k.to_string(), v); v };
    let vtx = put("vtx_fixed", 5000.0) + put("vtx_per_vertex", 0.0 * s.vertices as f64);
    let frag = put("frag_fixed", 4000.0) + put("frag_per_tile", 0.0 * s.tiles_processed as f64);
    t.vtx_ns = vtx;
    t.frag_ns = frag;
    t
}
