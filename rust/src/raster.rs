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
    pub depth_comp_base: u64,
    pub stencil_comp_base: u64,
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
            depth_comp_base: u64a(80),
            stencil_comp_base: u64a(104),
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
    pub frag_control2: Option<[u8; 4]>,
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
    if b(1) { st.frag_control2 = Some(take::<4>(rec, &mut o)?); } // its pass type selects HSR (HW, render probe pass_type)
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
pub const BLOCK_SIZES: [u64; 8] = [2, 4, 8, 16, 32, 64, 128, 256];
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub draws: u64,
    pub vertices: u64,
    pub vertex_groups: u64,
    pub vertex_instructions: u64,
    pub primitives: u64,
    pub culled: u64,
    pub clipped_unsupported: u64,
    pub clipped: u64,
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
    /// hidden-surface removal: what the hardware shades (opaque fragments only when finally visible or flushed by a
    /// translucent / punch-through / late-z fragment on the same pixel)
    pub hsr_fragment_groups: u64,
    /// samples of non-opaque (translucent / punch-through / late-z) primitives: they are shaded in order, never removed
    pub translucent_samples: u64,
    pub hsr_fragment_lanes: u64,
    pub hsr_fragment_instructions: f64,
    /// core-model time (ns, busiest core) of the shaded fragment work, the background and the end-of-tile programs
    pub frag_core_ns: f64,
    pub vs_core_ns: f64,
    /// (primitive, tile) pairs of the binned primitives' bounding boxes (the tiler's output); rasterized = primitives that
    /// reached the rasterizer (not culled / clipped away / zero area)
    pub prim_tiles: u64,
    /// tile-list entries if primitives were packed into aligned blocks of BLOCK_SIZES[i] consecutive primitives: a (block, tile)
    /// pair counts once (the tiler's primitive blocks; HW: shuffled triangle order costs up to 2.8x the fragment time)
    pub prim_blocks: [u64; 8],
    /// recency runs: a tile's primitive starts a new entry unless the tile's previous primitive was within RUN_GAPS[i] primitives
    pub prim_runs: [u64; 8],
    pub tile_last_prim: HashMap<(u32, u32), u64>,
    /// per tile: the binned primitives (rasterized index, in order) and the depth-tested samples (for the ISP timing study)
    pub tile_prims: HashMap<(u32, u32), Vec<u32>>,
    pub tile_samples: HashMap<(u32, u32), u64>,
    /// (primitive, tile) pairs with at least one shaded pixel (after HSR), total and per tile
    pub hsr_prim_tiles: u64,
    /// vertex-shader output words written (vertices x VDM output count; 4 of them are the clip position)
    pub vertex_output_words: u64,
    pub tile_visible: HashMap<(u32, u32), u64>,
    pub rasterized: u64,
    /// depth / stencil bytes the ZLS unit loads and stores
    pub zls_bytes: u64,
    pub bg_core_ns: f64,
    pub eot_core_ns: f64,
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

/// Byte offset of pixel (x, y) in a GPU-tiled ("twiddled") depth / stencil surface: Mesa ail tiles of 128x128 (1 B),
/// 128x64 (2 B), 64x64 (4 B) elements in row-major order, Morton order inside (x in the even bits). HW: a 64x64 f32 depth
/// buffer decoded bit by bit through direct-ioctl Z loads (render probe zls_layout).
pub fn zls_layout(width: u32, bpe: u64) -> impl Fn(u32, u32) -> u64 {
    let (tw, th) = match bpe { 1 => (128u32, 128u32), 2 => (128, 64), _ => (64, 64) };
    let tiles_x = width.div_ceil(tw) as u64;
    move |x: u32, y: u32| {
        let (tx, ty, ix, iy) = ((x / tw) as u64, (y / th) as u64, x % tw, y % th);
        let mut m = 0u64;
        for b in 0..8 {
            m |= (((ix >> b) & 1) as u64) << (2 * b) | (((iy >> b) & 1) as u64) << (2 * b + 1);
        }
        // non-square tiles (128x64): the extra x bit sits above the interleaved bits
        if tw > th {
            let k = th.trailing_zeros();
            m = (m & ((1u64 << (2 * k)) - 1)) | (((ix >> k) as u64) << (2 * k));
        }
        ((ty * tiles_x + tx) * (tw * th) as u64 + m) * bpe
    }
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

/// Plane through three screen points: (A, B, C) in f64 with value(x, y) = A x + B y + C (C at the screen origin).
fn plane(p: [(f64, f64); 3], v: [f64; 3]) -> [f64; 3] {
    let (x0, y0) = p[0];
    let (x1, y1) = (p[1].0 - x0, p[1].1 - y0);
    let (x2, y2) = (p[2].0 - x0, p[2].1 - y0);
    let det = x1 * y2 - x2 * y1;
    if det == 0.0 {
        return [0.0, 0.0, v[0]];
    }
    let (d1, d2) = (v[1] - v[0], v[2] - v[0]);
    let a = (d1 * y2 - d2 * y1) / det;
    let b = (x1 * d2 - x2 * d1) / det;
    [a, b, v[0] - a * x0 - b * y0]
}

/// The coefficient register the hardware loads for a tile whose origin is (ox, oy): A and B rounded to f32, C = the plane's value
/// at the tile origin rounded to f32 (HW: ldcf readback of 12 tile / register pairs, 9 bit-exact, 3 within 1 ulp).
fn tile_cf(pl: [f64; 3], ox: u32, oy: u32) -> [f32; 3] {
    [pl[0] as f32, pl[1] as f32, (pl[2] + pl[0] * ox as f64 + pl[1] * oy as f64) as f32]
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
    let core_ns = core_time_ns(mem, &d, blocks, p.regs);
    let r = func::run(&d, mem)?;
    CORE_NS.with(|c| *c.borrow_mut() += core_ns.unwrap_or(0.0));
    Ok(r.instructions)
}

thread_local! {
    /// core-model time (ns on the busiest core) of the fragment dispatches since the last `take_core_ns`
    static CORE_NS: std::cell::RefCell<f64> = const { std::cell::RefCell::new(0.0) };
}
fn take_core_ns() -> f64 {
    CORE_NS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Calibrated core-model time of a fragment-like dispatch spread over the GPU's 38 cores: one representative SIMD group (block 0) is traced
/// (its tile state saved and restored, so blending shaders stay correct) and `cost::simulate_trace` runs G copies of it at the
/// register-limited occupancy; the dispatch takes ceil(blocks / 38 cores) groups' worth of that throughput at 1398 MHz.
fn core_time_ns(mem: &Memory, d: &Dispatch, blocks: &[(u32, u32)], regs: u32) -> Option<f64> {
    if std::env::var("AGX_RENDER_NO_CORE").is_ok() { return Some(0.0); }
    static PARAMS: std::sync::OnceLock<Option<crate::cost::Params>> = std::sync::OnceLock::new();
    let params = PARAMS.get_or_init(|| crate::cost::Params::new().ok()).as_ref()?;
    let (bx, by) = *blocks.first()?;
    let keys: Vec<(u32, u32)> = (by..by + 4).flat_map(|y| (bx..bx + 8).map(move |x| (x, y))).collect();
    // direct lookups of the block's 32 pixels x 8 sample slots (a scan of the whole tile memory per primitive cost ~8 ms at 512x512)
    let skeys: Vec<(u32, u32, u32)> = keys.iter().flat_map(|&(x, y)| (0..8).map(move |s| (x, y, s))).collect();
    let saved = frag::with_tile(|ts| {
        let mem_k: Vec<_> = skeys.iter().map(|k| (*k, ts.mem.get(k).cloned())).collect();
        let zs_k: Vec<_> = skeys.iter().map(|k| (*k, ts.zs.get(k).copied())).collect();
        let live: Vec<_> = keys.iter().map(|k| (*k, ts.live.get(k).copied(), ts.tested.get(k).copied())).collect();
        (mem_k, zs_k, live)
    });
    let one = Dispatch { grid: [32, 1, 1], ..d.clone() };
    let tr = func::trace_workgroups(&one, mem, &[[0, 0, 0]]);
    frag::with_tile(|ts| {
        for (k, v) in saved.0 { match v { Some(v) => { ts.mem.insert(k, v); } None => { ts.mem.remove(&k); } } }
        for (k, v) in saved.1 { match v { Some(v) => { ts.zs.insert(k, v); } None => { ts.zs.remove(&k); } } }
        for (k, l, t) in saved.2 {
            match l { Some(l) => { ts.live.insert(k, l); } None => { ts.live.remove(&k); } }
            match t { Some(t) => { ts.tested.insert(k, t); } None => { ts.tested.remove(&k); } }
        }
    });
    let tr = tr.ok()?.into_iter().next()?.into_iter().next()?;
    let end = (tr.pcs.iter().copied().max()? + 16).min(d.code.len());
    // throughput of a full core (primitives' groups share all cores at once, so their work adds up GPU-wide, not per dispatch)
    let cap = 96u32.min(2 * (3328 / (2 * regs.max(8)))).clamp(1, 48) as usize;
    // cycles per group depend only on the program, its dynamic path and the occupancy: cached (thousands of primitives
    // share one fragment shader; uncached this was ~9 ms per primitive)
    use std::hash::{Hash, Hasher};
    let mut hc = std::collections::hash_map::DefaultHasher::new();
    d.code[..end].hash(&mut hc);
    let code_key = hc.finish();
    let mut hp = std::collections::hash_map::DefaultHasher::new();
    tr.pcs.hash(&mut hp);
    tr.masked.hash(&mut hp);
    let key = (code_key, hp.finish(), cap);
    thread_local! {
        static INSTRS: std::cell::RefCell<HashMap<u64, std::rc::Rc<Vec<crate::cost::Instr>>>> = std::cell::RefCell::new(HashMap::new());
        static CYC: std::cell::RefCell<HashMap<(u64, u64, usize), f64>> = std::cell::RefCell::new(HashMap::new());
    }
    let per_group = match CYC.with(|c| c.borrow().get(&key).copied()) {
        Some(v) => v,
        None => {
            let instrs = match INSTRS.with(|c| c.borrow().get(&code_key).cloned()) {
                Some(i) => i,
                None => {
                    let i = std::rc::Rc::new(crate::cost::decode(&d.code[..end], params).ok()?);
                    INSTRS.with(|c| c.borrow_mut().insert(code_key, i.clone()));
                    i
                }
            };
            let (cycles, _) = crate::cost::simulate_trace(&instrs, &tr, params, cap, false).ok()?;
            let v = cycles / cap as f64;
            CYC.with(|c| c.borrow_mut().insert(key, v));
            v
        }
    };
    Some(per_group * blocks.len() as f64 / 38.0 / 1.398)
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
    let z16 = (cmd.zls_ctrl >> 25 & 3) == 2;
    let zl = zls_layout(w, if z16 { 2 } else { 4 });
    let sl = zls_layout(w, 1);
    if cmd.zls_ctrl >> 15 & 1 != 0 && cmd.depth_base != 0 && samples == 1 {
        // Z load. Compressed with all-zero metadata: HW passes every z <= 0.95 (Less), modelled as 1.0 (render probe zls_*).
        let compressed = cmd.zls_ctrl >> 2 & 1 != 0;
        if compressed && !z16 {
            // Lossless depth compression (f32): the surface is split into 16x16-pixel blocks (256 Morton-ordered words);
            // one metadata byte per 8x4 sub-block (32 consecutive Morton words), 8 per block. HW-observed codes:
            //   3 = constant sub-block, value in data word (block * 256 + sub-block) (a stored clear to 1.0);
            //   0 = never written: loads as 1.0 (Less passes every z <= 0.95; render probe zls_*).
            // Other codes are not decoded: the sub-block loads the clear value and the pass log says so.
            let mut unknown = 0;
            for y in 0..h {
                for x in 0..w {
                    let word = zl(x, y) / 4;
                    let (blk, sub) = (word / 256, (word % 256) / 32);
                    let code = mem.read(cmd.depth_comp_base + blk * 8 + sub, 1).unwrap_or(0);
                    zbuf[zi(x, y, 0)] = match code {
                        0 => 1.0,
                        3 => f32::from_bits(mem.read(cmd.depth_base + 4 * (blk * 256 + sub), 4)? as u32),
                        _ => { unknown += 1; clear_z }
                    };
                }
            }
            if unknown > 0 {
                log.push(format!("warning: {} depth pixels in compressed sub-blocks with undecoded codes (loaded as clear)", unknown));
            }
        } else if compressed {
            log.push("warning: compressed z16 depth load not decoded (loaded as clear)".into());
        } else {
            for y in 0..h {
                for x in 0..w {
                    let a = cmd.depth_base + zl(x, y);
                    zbuf[zi(x, y, 0)] = if z16 { mem.read(a, 2)? as f32 / 65535.0 } else { f32::from_bits(mem.read(a, 4)? as u32) };
                }
            }
        }
        log.push(format!("z load (compressed {})", compressed));
    }
    if cmd.zls_ctrl >> 14 & 1 != 0 && cmd.zls_ctrl >> 4 & 1 != 0 {
        log.push("warning: compressed stencil load not decoded (loaded as the clear value)".into());
    }
    if cmd.zls_ctrl >> 14 & 1 != 0 && cmd.stencil_base != 0 && samples == 1 && cmd.zls_ctrl >> 4 & 1 == 0 {
        for y in 0..h {
            for x in 0..w {
                sbuf[zi(x, y, 0)] = mem.read(cmd.stencil_base + sl(x, y), 1)? as u8;
            }
        }
    }

    let base_setup = FragSetup {
        width: w,
        height: h,
        samples,
        tile: Some((tw, th)),
        sample_pos: spos.iter().map(|&(x, y)| [x as f32 / SUBPIXEL as f32, y as f32 / SUBPIXEL as f32]).collect(),
        ..Default::default()
    };

    // per primitive fragment lists, in API order (draw, primitive)
    let mut nprims_logged = 0usize;
    let mut tile_used: HashSet<(u32, u32)> = HashSet::new();
    let mut fs_pipes: Vec<Option<(Pipeline, Vec<(usize, u32)>)>> = vec![];

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
    // HSR bookkeeping: per pixel the opaque primitive that owns it and has not been shaded yet
    let mut owner: Vec<Option<u32>> = vec![None; (w * h) as usize];
    let mut prims: Vec<Prim> = vec![];
    let every_tile: Vec<(u32, u32)> = (0..tiles_y).flat_map(|ty| (0..tiles_x).map(move |tx| (tx, ty))).collect();
    if !opt.no_shading {
        let bgp = parse_usc(&mem, usc, usc + (cmd.bg.0 as u64 & !63))?;
        let bgu = bind(&mut mem, &bgp, &mut stats)?;
        let blocks = tile_blocks(&every_tile);
        let mut s = base_setup.clone();
        s.coverage_map = full_cov(&blocks);
        take_core_ns();
        stats.bg_instructions = frag_dispatch(&mut mem, &bgp, &bgu, s, &blocks, opt.max_steps).map_err(|e| format!("background program: {}", e))?;
        stats.bg_core_ns = take_core_ns();
    }
    for (di, d) in draws.iter().enumerate() {
        let mut ds = DrawStats::default();
        // ---- vertex stage
        let vs_uni = bind(&mut mem, &d.vs, &mut stats)?;
        let vcode = code_at(&mem, d.vs.code_va)?;
        let nv = d.count as u64 * d.instances as u64;
        stats.vertex_output_words += nv * d.vs_outputs as u64;
        let ids: Vec<(u32, u32, u32)> = (0..d.instances)
            .flat_map(|inst| (0..d.count).map(move |i| (inst, i)))
            .map(|(inst, i)| -> Result<(u32, u32, u32), String> {
                let idx = match d.indexed {
                    Some((ib, sz)) => mem.read(ib + i as u64 * sz as u64, sz as usize)? as u32,
                    None => i,
                };
                // a primitive-restart entry is not a vertex: shade a harmless id (its outputs are never used)
                if d.indexed.is_some() && d.restart == Some(idx) {
                    return Ok((idx, d.start, inst));
                }
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
                let groups: Vec<(u32, u32)> = (0..n.div_ceil(32) as u32).map(|g| (g, 0)).collect();
                stats.vs_core_ns += core_time_ns(&mem, &disp, &groups, d.vs.regs).unwrap_or(0.0);
                frag::take_uvs();
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
        let pass_type = bits(&ppp.frag_control2.unwrap_or(fctl), 29, 3) as u32;
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
        // late z: the fragment shader emits depth (zs_emit with z); the test then happens there (fsem::frag_zs_emit)
        let late_z = writes_depth(&code_at(&mem, fpipe.code_va)?);
        let _early_z = fprops >> 8 & 1 != 0;

        // ---- primitive assembly
        let tris: Vec<[usize; 3]> = assemble(d, &ids)?;
        let _ = &tris;
        let prov = |t: &[usize; 3], k: u64| -> usize { t[match k { 0 => 0, 6 => 1, _ => 2 }] };
        let fs_bind = if opt.no_shading { None } else { Some(bind(&mut mem, &fpipe, &mut stats)?) };
        fs_pipes.push(fs_bind.map(|u| (fpipe.clone(), u)));
        // Near-plane clipping in clip space (window z >= 0 with depth clip, else w >= eps): new vertices get every output
        // word interpolated as f32; the clipped polygon is fanned into triangles (screen-linear planes are unchanged).
        // x / y need no clipping (guard band; off-screen vertices are rasterized exactly, suite case clip_offscreen).
        let n_api = tris.len();
        let tris: Vec<[usize; 3]> = {
            let (tz, sz) = (vp[4] as f64, vp[5] as f64);
            let dist = |o: &Vec<u32>| -> f64 {
                let (z, wv) = (f32::from_bits(o[2]) as f64, f32::from_bits(o[3]) as f64);
                if depth_clip { tz * wv + sz * z } else { wv - 1e-6 }
            };
            let mut out_t = vec![];
            for t in &tris {
                let d: Vec<f64> = t.iter().map(|&i| dist(&outs[i])).collect();
                if d.iter().all(|&x| x >= 0.0) && t.iter().all(|&i| f32::from_bits(outs[i][3]) > 0.0) {
                    out_t.push(*t);
                    continue;
                }
                stats.clipped += 1;
                let mut poly: Vec<usize> = vec![];
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let (da, db) = (d[k], d[(k + 1) % 3]);
                    if da >= 0.0 {
                        poly.push(a);
                    }
                    if (da >= 0.0) != (db >= 0.0) {
                        let f = da / (da - db);
                        let nv: Vec<u32> = (0..nout).map(|w| {
                            let (x, y) = (f32::from_bits(outs[a][w]) as f64, f32::from_bits(outs[b][w]) as f64);
                            ((x + (y - x) * f) as f32).to_bits()
                        }).collect();
                        outs.push(nv);
                        poly.push(outs.len() - 1);
                    }
                }
                for k in 1..poly.len().saturating_sub(1) {
                    out_t.push([poly[0], poly[k], poly[k + 1]]);
                }
            }
            out_t
        };
        ds.primitives += n_api as u64;
        stats.primitives += n_api as u64;
        for t in &tris {
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
            // HW (cull_* suite cases): with `Front face CCW` set, a positive-area triangle here is the back face
            let ccw = area > 0;
            let back = ccw == front_ccw;
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
            stats.rasterized += 1;
            if bx1 > bx0 && by1 > by0 {
                let (t0x, t1x) = (bx0.max(0) as u32 / tw, (bx1 - 1).max(0) as u32 / tw);
                let (t0y, t1y) = (by0.max(0) as u32 / th, (by1 - 1).max(0) as u32 / th);
                stats.prim_tiles += ((t1x - t0x + 1) * (t1y - t0y + 1)) as u64;
                let pi = stats.rasterized;
                for ty in t0y..=t1y {
                    for tx in t0x..=t1x {
                        let last = stats.tile_last_prim.insert((tx, ty), pi);
                        stats.tile_prims.entry((tx, ty)).or_default().push(pi as u32);
                        for (i, b) in BLOCK_SIZES.iter().enumerate() {
                            if last.is_none_or(|l| l / b != pi / b) {
                                stats.prim_blocks[i] += 1;
                            }
                            if last.is_none_or(|l| pi - l > *b) {
                                stats.prim_runs[i] += 1;
                            }
                        }
                    }
                }
            }
            // HW: planes are fitted to the vertices snapped to 1/256 pixel (ldcf readback: 0-1 ulp vs 600-2900 ulp unsnapped)
            let sp = SUBPIXEL as f64;
            let pts = [(sv[0].fx as f64 / sp, sv[0].fy as f64 / sp), (sv[1].fx as f64 / sp, sv[1].fy as f64 / sp), (sv[2].fx as f64 / sp, sv[2].fy as f64 / sp)];
            let zpl = plane(pts, [sv[0].z, sv[1].z, sv[2].z]);
            let mut ztiles: HashMap<(u32, u32), [f32; 3]> = HashMap::new();
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
                        let (tx, ty) = (px as u32 / tw, py as u32 / th);
                        let zc = *ztiles.entry((tx, ty)).or_insert_with(|| tile_cf(zpl, tx * tw, ty * th));
                        let fx = (px as u32 - tx * tw) as f32 + sx as f32 / SUBPIXEL as f32;
                        let fy = (py as u32 - ty * th) as f32 + sy as f32 / SUBPIXEL as f32;
                        let z = zc[0].mul_add(fx, zc[1].mul_add(fy, zc[2]));
                        if depth_clip && !(0.0..=1.0).contains(&z) {
                            continue;
                        }
                        let k = zi(px as u32, py as u32, si as u32);
                        stats.depth_tested += 1;
                        *stats.tile_samples.entry((tx, ty)).or_default() += 1;
                        let sval = sbuf[k];
                        if stencil_en && !stencil_pass(s_cmp, sref & srm, sval & srm) {
                            sbuf[k] = (sval & !swm) | (stencil_op(s_sfail, sval, sref) & swm);
                            continue;
                        }
                        let zp = late_z || (zs_pass(zfunc, z, zbuf[k]) && zs_pass(zfunc2, z, zbuf[k]));
                        if stencil_en {
                            let op = if zp { s_dpass } else { s_dfail };
                            sbuf[k] = (sval & !swm) | (stencil_op(op, sval, sref) & swm);
                        }
                        if !zp {
                            continue;
                        }
                        if zwrite && !late_z {
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
            let mut planes: Vec<[f64; 3]> = vec![[0.0; 3]; ncf];
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
                    let pl = match model {
                        7 => plane(pts, std::array::from_fn(|k| vals[k] * sv[k].inv_w)),
                        3 => plane(pts, vals),
                        k => {
                            let pv = prov(t, k);
                            let ki = t.iter().position(|&v| v == pv).unwrap();
                            [0.0, 0.0, vals[ki]]
                        }
                    };
                    if reg + c < setup.cf.len() {
                        setup.cf[reg + c] = tile_cf(pl, 0, 0);
                        planes[reg + c] = pl;
                    }
                }
            }
            nprims_logged += 1;
            if nprims_logged <= 64 {
                log.push(format!("cf draw {} prim {}: {}", di, nprims_logged - 1, setup.cf.iter()
                    .map(|c| format!("{:08x},{:08x},{:08x}", c[0].to_bits(), c[1].to_bits(), c[2].to_bits())).collect::<Vec<_>>().join(" ")));
                log.push(format!("verts draw {} prim {}: {}", di, nprims_logged - 1, sv.iter()
                    .map(|v| format!("{:.9},{:.9},{:.9},{:.9}", v.x, v.y, v.z, v.inv_w)).collect::<Vec<_>>().join(" ")));
            }
            for f in &frags {
                let t = (f.x / tw, f.y / th);
                setup.cf_tiles.entry(t).or_insert_with(|| planes.iter().map(|&pl| tile_cf(pl, t.0 * tw, t.1 * th)).collect());
            }
            // ---- hidden-surface removal and shading (HW, render probe pass_type: Fragment control 2's pass type selects it).
            // Opaque: deferred, a pixel is shaded only by its last covering opaque fragment, when a non-opaque one lands on it
            // or at the end of the pass. Translucent / punch-through (blend, discard, late z): shaded at once, in API order,
            // after flushing the deferred opaque fragments under them.
            let mut nblocks: BTreeMap<(u32, u32), ()> = BTreeMap::new();
            for f in &frags {
                nblocks.insert((f.x & !7, f.y & !3), ());
                stats.fragment_samples += f.mask.count_ones() as u64;
            }
            stats.fragment_groups += nblocks.len() as u64;
            stats.fragment_lanes += frags.len() as u64;
            ds.fragment_groups += nblocks.len() as u64;
            let pid = prims.len() as u32;
            let opaque = pass_type == 0 && !late_z;
            setup.depth_func = if late_z { ["never", "less", "equal", "lequal", "greater", "notequal", "gequal", "always"][zfunc as usize].into() } else { "always".into() };
            let masks: HashMap<(u32, u32), u32> = frags.iter().map(|f| ((f.x, f.y), f.mask)).collect();
            prims.push(Prim { draw: di, setup, masks, late_z, zwrite });
            if opaque {
                for f in &frags {
                    owner[(f.y * w + f.x) as usize] = Some(pid);
                }
            } else {
                stats.translucent_samples += frags.iter().map(|f| f.mask.count_ones() as u64).sum::<u64>();
                let mut flush: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
                for f in &frags {
                    if let Some(o) = owner[(f.y * w + f.x) as usize].take() {
                        flush.entry(o).or_default().push((f.x, f.y));
                    }
                }
                for (o, px) in flush {
                    shade(&mut mem, &mut stats, &prims[o as usize], &px, &fs_pipes, &mut zbuf, &zi, samples, opt.max_steps)?;
                }
                let px: Vec<(u32, u32)> = frags.iter().map(|f| (f.x, f.y)).collect();
                shade(&mut mem, &mut stats, &prims[pid as usize], &px, &fs_pipes, &mut zbuf, &zi, samples, opt.max_steps)?;
            }
        }
        stats.per_draw.push(ds);
    }

    // end of the pass: the deferred opaque fragments still owning a pixel
    let mut flush: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
    for (k, o) in owner.iter().enumerate() {
        if let Some(o) = o {
            flush.entry(*o).or_default().push((k as u32 % w, k as u32 / w));
        }
    }
    // Timing: the hardware merges deferred fragments of different primitives with the same fragment program into shared SIMD
    // groups (a block shaded for 3 surviving primitives is one group, not 3): scale this flush's core time by distinct (program,
    // block) / per-primitive blocks. Functionally each primitive still runs with its own coefficients.
    let mut per_prim = 0u64;
    let mut distinct: HashSet<(u64, u32, u32)> = HashSet::new();
    for (o, px) in &flush {
        let code = fs_pipes[prims[*o as usize].draw].as_ref().map_or(0, |p| p.0.code_va);
        let mut b: HashSet<(u32, u32)> = HashSet::new();
        for &(x, y) in px {
            b.insert((x & !7, y & !3));
            distinct.insert((code, x & !7, y & !3));
        }
        per_prim += b.len() as u64;
    }
    let before = (stats.hsr_fragment_groups, take_core_ns());
    for (o, px) in flush {
        shade(&mut mem, &mut stats, &prims[o as usize], &px, &fs_pipes, &mut zbuf, &zi, samples, opt.max_steps)?;
    }
    let merge = if per_prim > 0 { distinct.len() as f64 / per_prim as f64 } else { 1.0 };
    let flushed = take_core_ns();
    CORE_NS.with(|c| *c.borrow_mut() += before.1 + flushed * merge);
    stats.merged_fragment_groups = distinct.len() as u64 + (before.0);
    // ---- end of tile over the processed tiles (bg ran over every tile before the geometry; unprocessed tiles are never stored)
    let processed: Vec<(u32, u32)> = (0..tiles_y)
        .flat_map(|ty| (0..tiles_x).map(move |tx| (tx, ty)))
        .filter(|t| cmd.process_empty_tiles() || tile_used.contains(t))
        .collect();
    stats.tiles_processed = processed.len() as u64;
    stats.bg_groups = tile_blocks(&processed).len() as u64;
    let all_blocks = tile_blocks(&processed);
    // merged groups: distinct (tile, block) per draw, which is what triangle merging can reach at best
    if !opt.no_shading {
        let eop = parse_usc(&mem, usc, usc + (cmd.eot.0 as u64 & !63))?;
        let eou = bind(&mut mem, &eop, &mut stats)?;
        let mut s = base_setup.clone();
        s.coverage_map = full_cov(&all_blocks);
        stats.frag_core_ns = take_core_ns();
        stats.eot_instructions = frag_dispatch(&mut mem, &eop, &eou, s, &all_blocks, opt.max_steps).map_err(|e| format!("end-of-tile program: {}", e))?;
        stats.eot_core_ns = take_core_ns();
        stats.eot_groups = all_blocks.len() as u64;
    }
    frag::replace(None);
    // Z / S store (uncompressed GPU-tiled only; a compressed store's metadata format is not modelled, so it is skipped)
    if cmd.zls_ctrl >> 19 & 1 != 0 && cmd.zls_ctrl >> 6 & 1 == 0 && cmd.depth_base != 0 && samples == 1 {
        for y in 0..h {
            for x in 0..w {
                let a = cmd.depth_base + zl(x, y);
                let z = zbuf[zi(x, y, 0)];
                if z16 { mem.write(a, 2, (z.clamp(0.0, 1.0) * 65535.0).round() as u64)?; } else { mem.write(a, 4, z.to_bits() as u64)?; }
            }
        }
    }
    if cmd.zls_ctrl >> 18 & 1 != 0 && cmd.zls_ctrl >> 8 & 1 == 0 && cmd.stencil_base != 0 && samples == 1 {
        for y in 0..h {
            for x in 0..w {
                mem.write(cmd.stencil_base + sl(x, y), 1, sbuf[zi(x, y, 0)] as u64)?;
            }
        }
    }
    let zb = if z16 { 2u64 } else { 4 };
    let px = (w * h * samples) as u64;
    for (bit, bytes) in [(15, zb), (19, zb), (14, 1), (18, 1)] {
        if cmd.zls_ctrl >> bit & 1 != 0 {
            stats.zls_bytes += px * bytes;
        }
    }
    log.push(format!("tiles {}x{} processed {}", tiles_x, tiles_y, stats.tiles_processed));
    Ok(PassResult { stats, log, mem, cmd })
}

struct Prim {
    draw: usize,
    setup: FragSetup,
    masks: HashMap<(u32, u32), u32>,
    late_z: bool,
    zwrite: bool,
}

/// Run a primitive's fragment shader over a subset of its pixels (the ones the hardware shades after hidden-surface removal).
#[allow(clippy::too_many_arguments)]
fn shade(mem: &mut Memory, stats: &mut Stats, p: &Prim, px: &[(u32, u32)], pipes: &[Option<(Pipeline, Vec<(usize, u32)>)>],
         zbuf: &mut [f32], zi: &dyn Fn(u32, u32, u32) -> usize, samples: u32, max_steps: u64) -> Result<(), String> {
    let mut blocks: BTreeMap<(u32, u32), ()> = BTreeMap::new();
    let mut cov: HashMap<(u32, u32), u32> = HashMap::new();
    for &(x, y) in px {
        blocks.insert((x & !7, y & !3), ());
        cov.insert((x, y), p.masks[&(x, y)]);
    }
    let blocks: Vec<(u32, u32)> = blocks.into_keys().collect();
    for &(bx, by) in &blocks {
        for y in by..by + 4 {
            for x in bx..bx + 8 {
                cov.entry((x, y)).or_insert(0);
            }
        }
    }
    stats.hsr_fragment_groups += blocks.len() as u64;
    stats.hsr_fragment_lanes += px.len() as u64;
    let vt: HashSet<(u32, u32)> = px.iter().map(|&(x, y)| (x / 32, y / 32)).collect();
    stats.hsr_prim_tiles += vt.len() as u64;
    for t in vt {
        *stats.tile_visible.entry(t).or_default() += 1;
    }
    if stats.per_draw.len() > p.draw {
        stats.per_draw[p.draw].merged_fragment_groups += blocks.len() as u64;
    }
    let Some((fp, fu)) = pipes[p.draw].clone() else { return Ok(()) };
    let mut setup = p.setup.clone();
    setup.coverage_map = cov;
    if p.late_z {
        frag::with_tile(|ts| {
            ts.zs.clear();
            for &(x, y) in px {
                for sm in 0..samples {
                    ts.zs.insert((x, y, sm), (Some(zbuf[zi(x, y, sm)].to_bits()), None));
                }
            }
        });
    }
    let n = frag_dispatch(mem, &fp, &fu, setup, &blocks, max_steps).map_err(|e| format!("fragment shader (draw {}): {}", p.draw, e))?;
    if p.late_z && p.zwrite {
        frag::with_tile(|ts| {
            for (&(x, y, sm), &(z, _)) in &ts.zs {
                if let Some(z) = z { zbuf[zi(x, y, sm)] = f32::from_bits(z); }
            }
        });
    }
    stats.fragment_instructions += n;
    stats.hsr_fragment_instructions += n as f64;
    if stats.per_draw.len() > p.draw {
        stats.per_draw[p.draw].fragment_instructions += n;
    }
    Ok(())
}

/// whether a fragment program emits depth: a `zs_emit` with the z bit before the first `stop`
fn writes_depth(code: &[u8]) -> bool {
    let mut p = 0;
    while p + 2 <= code.len() {
        let mut buf = code[p..(p + 16).min(code.len())].to_vec();
        buf.resize(16, 0);
        let Some((d, n, size)) = crate::isa::decode_one(&buf) else { return false };
        if size == 0 { return false; }
        if d.name == "zs_emit" && (n >> 8) & 0xff != 0 && (code[p + 3] >> 0) != 0 {
            let text = crate::isa::disassemble_normalized(d, n).unwrap_or_default();
            if text.contains("zs_emit") { return true; }
        }
        if d.name == "stop" { return false; }
        p += size;
    }
    false
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
        assert_eq!(tile_cf(p, 0, 0), [0.1, 0.2, 1.0]);
        assert_eq!(tile_cf(p, 32, 0)[2], 4.2);
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

/// Timing model of a render pass at the top GPU clock (1398 MHz). Fitted 2026-11-21 on 118 direct-ioctl replays (results/2026-11-21-
/// render-ioctl: suite1 + perf1 + perf2; scripts/render_fit.py nl, mean relative error; timed after a clock-keeper warm-up in back-to-back
/// bursts) and checked on held-out scenes (held1 22 cases, held2: see the results README):
///
/// vertex   = 6774 ns + 0.957 ns / screen tile (tiler setup) + 41.7 ns / draw + 2.18 ns x v / (1 + v / 1846) for v vertices
///            (the stage spreads over more clusters as it grows) + 0.092 ns / (primitive, tile) bin
/// fragment = 3226 ns + max(10.6 ns / processed tile, 1.212 x C + 2.784 x C x 38 / processed tiles) + 0.0014 ns / translucent sample
///            + 642 ns if any primitive is rasterized + 0.110 ns / bin
///            C = calibrated core-model time of the shaded fragment, background and end-of-tile groups spread over 38 cores
///            (`cost::simulate_trace` of one traced group at the register-limited occupancy, cached per program path); the second
///            shading term is a tail: ~2.8 tiles' worth of one tile's shading per core is not overlapped (hardware / core model 2.2x
///            at 144 tiles, 1.35x at 2304). Tile overhead and shading overlap (max). Hidden-surface removal decides which groups
///            count; deferred fragments of different primitives with one program share groups.
/// Known misses: thousands of small randomly overlapping depth-tested triangles (fragment ~50% low: a per-primitive ISP cost
/// not modelled), Z/S load-store traffic (one stencil case 30% low), 1-2 us of the fixed costs on empty / fully culled passes.
pub fn timing(r: &PassResult) -> Timing {
    let s = &r.stats;
    let mut t = Timing::default();
    let mut put = |k: &str, v: f64| { t.parts.insert(k.to_string(), v); v };
    let v = s.vertices as f64;
    let vtx = put("vtx_fixed", 6773.7) + put("vtx_tiles", 0.95669 * s.tiles as f64) + put("vtx_draws", 41.676 * s.draws as f64)
        + put("vtx_vertices", 2.18134 * v / (1.0 + v / 1846.23)) + put("vtx_bins", 0.09185 * s.prim_tiles as f64);
    let tiles = 10.61574 * s.tiles_processed as f64;
    let core = s.frag_core_ns + s.bg_core_ns + s.eot_core_ns;
    let shade = 1.21221 * core + 2.78401 * core * 38.0 / (s.tiles_processed.max(1) as f64);
    put("frag_tiles", tiles);
    put("frag_shading", shade);
    let frag = put("frag_fixed", 3225.65) + tiles.max(shade) + put("frag_translucent", 0.00139 * s.translucent_samples as f64)
        + put("frag_geometry", if s.rasterized > 0 { 642.27 } else { 0.0 }) + put("frag_bins", 0.10965 * s.prim_tiles as f64);
    t.vtx_ns = vtx;
    t.frag_ns = frag;
    t
}
