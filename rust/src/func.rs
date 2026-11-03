//! Functional executor: runs every SIMD group of a compute dispatch over real buffers.
//!
//! Instructions the applegpu-derived emulator (`isa::compiled`) implements run through it; this module adds
//! what a dispatch needs around it: device memory mapped at virtual addresses, the harness uniforms, special
//! registers per thread (`cost::GroupCtx`), device loads / stores of every format and mask, threadgroup memory per
//! workgroup, device and threadgroup atomics, barriers (whole workgroup, and 0x78 over aligned blocks of 4
//! groups), and the exec-mask jumps. Semantics follow the playground findings (vector loads index format
//! elements; 8/16-bit loads write the low half; threadgroup byte = base + 4*size*index + size*component; atomics
//! return the old value; cmpxchg's source holds the new value, source+1 the compare value).

use crate::cost::{is_barrier78, GroupCtx, Trace};
use crate::fsem;
use crate::isa::{self, compiled::Compiled, exec::CoreState, fget, Desc, Fields, ImmVal, Operand};
use std::sync::OnceLock;

/// Sparse byte-addressed memory made of mapped regions. Regions are shared copy-on-write: cloning a Memory (one per
/// traced workgroup) shares every buffer, and a write copies only the region it lands in.
#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub regions: Vec<(u64, std::sync::Arc<Vec<u8>>)>,
    /// optional log of device stores, for checking racing writes: byte address -> the last byte each writer
    /// (SIMD group, `writer`) stored there, in first-write order. None: not recorded.
    pub log: Option<std::collections::HashMap<u64, Vec<(u64, u8)>>>,
    /// the SIMD group now running: (linear workgroup index << 16) | group in workgroup
    pub writer: u64,
    /// texture state slots (24-byte descriptors) written by tex_state_store, read by texture ops
    pub tex_state: std::collections::BTreeMap<u32, Vec<u8>>,
    /// sampler state slots (8-byte descriptors) written by sampler_state_store
    pub samp_state: std::collections::BTreeMap<u32, Vec<u8>>,
    /// the command's sampler heap (drm_asahi_cmd_compute sampler_heap / sampler_count): 8-byte sampler descriptors a
    /// register sampler index selects
    pub sampler_heap: Option<(u64, u32)>,
}

impl Memory {
    pub fn map(&mut self, va: u64, bytes: Vec<u8>) {
        self.regions.push((va, std::sync::Arc::new(bytes)));
    }
    fn find(&self, va: u64, len: u64) -> Option<(usize, usize)> {
        if len > 0 && va.checked_add(len - 1).is_none() { return None; }
        self.regions.iter().enumerate().find_map(|(i, (b, d))| {
            let off = va.checked_sub(*b)?;
            (off <= d.len() as u64 && len <= d.len() as u64 - off).then_some((i, off as usize))
        })
    }
    /// Copy a mapped byte range in bulk. Adjacent regions retain byte-wise dump semantics.
    pub fn read_bytes(&self, va: u64, len: usize) -> Result<Vec<u8>, String> {
        if len == 0 { return Ok(Vec::new()); }
        va.checked_add((len - 1) as u64).ok_or("memory range wraps the address space")?;
        if let Some((i, o)) = self.find(va, len as u64) {
            // A preceding overlapping region must still win for individual bytes.
            let end = va as u128 + len as u128;
            if !self.regions[..i].iter().any(|(b, d)| (*b as u128) < end && (*b as u128 + d.len() as u128) > va as u128) {
                return Ok(self.regions[i].1[o..o + len].to_vec());
            }
        }
        (0..len).map(|k| self.read(va + k as u64, 1).map(|v| v as u8)).collect()
    }
    pub fn read(&self, va: u64, len: usize) -> Result<u64, String> {
        if len > 8 { return Err("scalar memory read exceeds 8 bytes".into()); }
        let (i, o) = self.find(va, len as u64).ok_or_else(|| format!("read of unmapped address {:#x}", va))?;
        let d = &self.regions[i].1;
        let mut v = 0u64;
        for k in 0..len {
            v |= (d[o + k] as u64) << (8 * k);
        }
        Ok(v)
    }
    pub fn write(&mut self, va: u64, len: usize, v: u64) -> Result<(), String> {
        if len > 8 { return Err("scalar memory write exceeds 8 bytes".into()); }
        let (i, o) = self.find(va, len as u64).ok_or_else(|| format!("write to unmapped address {:#x}", va))?;
        let d = std::sync::Arc::make_mut(&mut self.regions[i].1);
        for k in 0..len {
            d[o + k] = (v >> (8 * k)) as u8;
        }
        Ok(())
    }
    /// Overwrite a mapped byte range (one region) in bulk.
    pub fn write_bytes(&mut self, va: u64, bytes: &[u8]) -> Result<(), String> {
        let (i, o) = self.find(va, bytes.len() as u64).ok_or_else(|| format!("write to unmapped address {:#x}", va))?;
        std::sync::Arc::make_mut(&mut self.regions[i].1)[o..o + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    /// `write` from a device store, also recorded in `log` when logging is on
    pub fn store(&mut self, va: u64, len: usize, v: u64) -> Result<(), String> {
        self.write(va, len, v)?;
        let w = self.writer;
        if let Some(log) = self.log.as_mut() {
            for k in 0..len {
                let e = log.entry(va + k as u64).or_default();
                let b = (v >> (8 * k)) as u8;
                match e.iter_mut().find(|(x, _)| *x == w) {
                    Some(slot) => slot.1 = b,
                    None => e.push((w, b)),
                }
            }
        }
        Ok(())
    }
    /// The values a byte can hold after the run when SIMD groups raced on it: the last byte of every writer
    /// group, or None if at most one group stored to it (then the simulated value is exact).
    pub fn racing(&self, va: u64) -> Option<Vec<u8>> {
        let e = self.log.as_ref()?.get(&va)?;
        (e.len() > 1).then(|| e.iter().map(|x| x.1).collect())
    }
    pub fn words(&self, va: u64, n: usize) -> Result<Vec<u32>, String> {
        n.checked_mul(4).ok_or("word range is too large")?;
        (0..n).map(|k| va.checked_add(4 * k as u64).ok_or_else(|| "word range wraps the address space".to_string())
            .and_then(|a| self.read(a, 4).map(|x| x as u32))).collect()
    }
}

#[derive(Clone, Debug)]
pub struct Dispatch {
    pub code: Vec<u8>,
    pub grid: [u32; 3],
    pub wg: [u32; 3],
    pub shared_bytes: u32,
    /// 32-bit uniform registers (index, value)
    pub uniforms: Vec<(usize, u32)>,
    pub num_registers: usize,
    /// Maximum executed instructions per SIMD group, including all barrier resumes.
    pub max_steps: u64,
    /// 0: every workgroup alone on core 0 (sr20 = 0, sr17 = SIMD group in workgroup). Otherwise a seed for a
    /// different placement (other core ids; with more than one workgroup, other hardware slots), so a checker can
    /// tell which results depend on where the hardware happened to place the workgroups.
    pub placement: u32,
    /// what SR18 / SR19 / SR21 / SR32 read
    pub launch: crate::cost::SrLaunch,
}

#[derive(Debug, Default)]
pub struct RunStats {
    pub instructions: u64,
    pub groups: u64,
    /// the program read a placement-dependent special register (sr17 slot, sr20 core id)
    pub placement_read: bool,
}

struct Info {
    desc: Option<&'static Desc>,
    size: usize,
    fields: Fields,
    op: Option<Compiled>,
    /// the value operands (registers, uniforms, immediates) the semantic helpers take, computed once
    vals: Vec<Operand>,
    /// what step_group does with it, decided once from the name instead of on every execution
    kind: OpKind,
    access: Option<Box<crate::memtrace::Access>>,
}

// Instruction cells are shared by independent traced workgroups. Tail
// pages stay unallocated unless visited; execution state remains in Group.
enum CodeCache {
    Dense(Vec<OnceLock<Box<Info>>>),
    Paged(crate::decode_cache::DecodeCache<Info, 1>),
}

fn code_cache(bytes: usize) -> CodeCache {
    if bytes <= 8192 {
        CodeCache::Dense((0..bytes).map(|_| OnceLock::new()).collect())
    } else {
        CodeCache::Paged(crate::decode_cache::DecodeCache::new(bytes))
    }
}

trait InstructionCells {
    fn cell(&self, pc: usize) -> &OnceLock<Box<Info>>;
}

impl InstructionCells for Vec<OnceLock<Box<Info>>> {
    #[inline]
    fn cell(&self, pc: usize) -> &OnceLock<Box<Info>> { &self[pc] }
}

impl InstructionCells for crate::decode_cache::DecodeCache<Info, 1> {
    #[inline]
    fn cell(&self, pc: usize) -> &OnceLock<Box<Info>> { self.at(pc) }
}

enum OpKind {
    Stop,
    Jmp { any: bool },
    Call,
    Ret,
    TgBarrier,
    Nop,
    Wait,
    GetSr(Box<SpecialRegister>),
    DevMem { load: bool, args: Box<DeviceMem> },
    Stack { load: bool },
    UniformStore,
    TgMem { load: bool, args: Box<ThreadgroupMem> },
    Atomic(Box<DeviceAtomic>),
    FloatArith,
    CmpSel { float: bool },
    Ballot { float: bool, quad: bool },
    MaskCtl { kind: &'static str, float: bool },
    TgAtomic(Box<ThreadgroupAtomic>),
    Round,
    Intl,
    Transcendental,
    Convert,
    QuadShuffle,
    CrossLane,
    TexStateStore { ts: u32 },
    SampStateStore { ss: u32 },
    TexLoad { args: Box<TexOp> },
    TexSample { args: Box<TexOp> },
    ImageWrite { args: Box<TexOp> },
    Generic,
}

/// One decoded texture/image instruction: destination regs, texture source (state slot or
/// bindless uniform base + offset register), sampler slot, coordinate regs and LOD source.
#[derive(Clone, Debug, Default)]
pub struct TexOp {
    pub dest: Vec<(bool, u32)>,
    pub tslot: Option<u32>,
    pub bind_ureg: Option<u32>,
    pub bind_reg: Option<(bool, u32)>,
    pub ss: Option<u32>,
    /// register-form sampler (bindless heap index, or the clamp_0 override): value resolved per lane
    pub sreg: Option<(bool, u32)>,
    pub coords: Vec<(bool, u32)>,
    pub lod_ureg: Option<u32>,
    pub lod_imm: Option<i64>,
    pub lod_reg: Option<(bool, u32)>,
    pub mode: u64,
}

fn reg16_32(o: &Operand) -> Option<(bool, u32)> {
    match o {
        Operand::Reg16(r, _) => Some((true, *r)),
        Operand::Reg32(r, _) => Some((false, *r)),
        _ => None,
    }
}

fn reg_tuple(o: &Operand) -> Option<Vec<(bool, u32)>> {
    match o {
        Operand::Tuple(rs, _) => {
            let v: Vec<(bool, u32)> = rs.iter().filter_map(reg16_32).collect();
            if v.len() == rs.len() && !v.is_empty() {
                Some(v)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn tex_op(ops: &[Operand], sampler_reg: bool) -> TexOp {
    // Operand order varies by op (texture_sample: R,U?,T,S?,C,lod; image_write: R,C,D,U,T,...).
    // Order-independent parse: the first register group (lone regs or tuple) is the destination,
    // the second is the coordinates; slots / bindless base attach wherever they appear.
    let mut a = TexOp::default();
    let mut dest_done = false;
    let mut coords_done = false;
    for o in ops {
        if let Some(rs) = reg_tuple(o) {
            if !dest_done {
                a.dest = rs;
                dest_done = true;
            } else if !coords_done {
                a.coords = rs;
                coords_done = true;
            }
            continue;
        }
        match o {
            Operand::UReg64(n, _) => {
                if a.bind_ureg.is_none() {
                    a.bind_ureg = Some(*n);
                }
                dest_done = true;
            }
            Operand::TextureState(n) => {
                a.tslot = Some(*n);
                dest_done = true;
            }
            Operand::SamplerState(n) => {
                a.ss = Some(*n);
            }
            Operand::Reg16(..) | Operand::Reg32(..) => {
                if !dest_done {
                    a.dest.push(reg16_32(o).unwrap());
                } else if a.bind_ureg.is_some() && a.tslot.is_none() && a.bind_reg.is_none() {
                    // bindless texture offset register (uniform base seen, no slot)
                    a.bind_reg = reg16_32(o);
                } else if sampler_reg && a.bind_reg.is_some() && !coords_done && a.sreg.is_none() {
                    // bindless texture with a register sampler (St = 1): a sampler heap index
                    a.sreg = reg16_32(o);
                } else if a.tslot.is_some() && !coords_done && a.sreg.is_none() {
                    // register-form sampler (bindless heap index / clamp_0 override).
                    // NOTE: a lone 1D coordinate register after a slot would land here too;
                    // 1D sampled coords are not modelled.
                    a.sreg = reg16_32(o);
                } else if !coords_done {
                    a.coords.push(reg16_32(o).unwrap());
                    coords_done = true;
                } else if a.lod_reg.is_none() {
                    a.lod_reg = reg16_32(o);
                }
            }
            Operand::UReg16(n, _) if a.lod_ureg.is_none() => {
                a.lod_ureg = Some(*n);
            }
            Operand::Imm(ImmVal::Int(v), _) if a.lod_imm.is_none() => {
                a.lod_imm = Some(*v);
            }
            _ => {}
        }
    }
    a
}

fn op_kind(name: &'static str, f: &Fields, ops: &[Operand]) -> OpKind {
    use OpKind::*;
    match name {
        "stop" | "end_group" => Stop,
        // Measured on G14C (ptx-re-blockers B21, playground results/2026-10-20-b21-trap): an executed trap does
        // nothing; the stores after it land and the command ends normally.
        "nop6" | "fixed18" | "trap" => Nop,
        "jmp_exec_any" => Jmp { any: true },
        "jmp_exec_none" => Jmp { any: false },
        "call" => Call,
        "ret" => Ret,
        "threadgroup_barrier" => TgBarrier,
        "wait" => Wait,
        "memory_barrier" => Nop,
        "get_sr" => GetSr(Box::new(SpecialRegister::new(f, ops))),
        // the .TODO encodings differ from the plain ones only in control bits (Mesa emits them for Vulkan-memory-model
        // atomic loads / stores, between memory_barrier sequences); on one sequentially simulated memory they read and
        // write exactly as the plain access does
        "device_load" | "device_store" | "device_load.TODO" | "device_store.TODO" => DevMem { load: name.starts_with("device_load"), args: Box::new(DeviceMem::new(f)) },
        "stack_load" | "stack_store" => Stack { load: name == "stack_load" },
        "uniform_store" => UniformStore,
        "threadgroup_load" | "threadgroup_store" => TgMem { load: name == "threadgroup_load", args: Box::new(ThreadgroupMem::new(f)) },
        "atomic" => Atomic(Box::new(DeviceAtomic::new(f))),
        "fadd32" | "fmul32" | "fmadd32" | "fadd16" | "fmul16" | "fmadd16" => FloatArith,
        "icmpsel" | "fcmpsel" => CmpSel { float: name == "fcmpsel" },
        "icmp_ballot" | "fcmp_ballot" | "icmp_quad_ballot" | "fcmp_quad_ballot" => Ballot { float: name.starts_with('f'), quad: name.contains("quad") },
        "if_icmp" | "if_fcmp" | "else_icmp" | "else_fcmp" | "while_icmp" | "while_fcmp" => {
            MaskCtl { kind: name.split('_').next().unwrap(), float: name.ends_with("fcmp") }
        }
        "threadgroup_atomic" => TgAtomic(Box::new(ThreadgroupAtomic::new(f))),
        "floor" | "ceil" | "trunc" | "rint" => Round,
        "intl" => Intl,
        "rcp" | "rsqrt" | "rsqrt_special" | "log2" | "exp2" | "sin_pt_1" | "sin_pt_2" => Transcendental,
        "convert" => Convert,
        "tex_state_store" => TexStateStore { ts: fget(f, "TS").unwrap_or(0) as u32 },
        "sampler_state_store" => SampStateStore { ss: fget(f, "SS").unwrap_or(0) as u32 },
        "texture_load" => TexLoad { args: Box::new(tex_op(ops, fget(f, "St") == Some(1))) },
        "texture_sample" => TexSample { args: Box::new(tex_op(ops, fget(f, "St") == Some(1))) },
        "image_write" => ImageWrite { args: Box::new(tex_op(ops, false)) },
        n if n.starts_with("quad_shuffle") => QuadShuffle,
        n if (n.starts_with("simd_") || n.starts_with("quad_")) && !n.starts_with("simd_shuffle") && !n.starts_with("simd_matrix") => CrossLane,
        _ => Generic,
    }
}

enum Stop {
    Barrier(u8),
    Done,
}

struct Group {
    st: CoreState,
    ctx: GroupCtx,
    pc: usize,
    done: bool,
    steps: u64,
    /// recorded dynamic path in the cost model's format (pcs, memory segments / threadgroup words per access)
    trace: Option<Trace>,
    /// per-lane private stack (scratch) memory for stack_store / stack_load, grown on demand
    stack: Vec<Vec<u8>>,
    /// device loads issued on scoreboard slot != 0 whose register writes land at the `wait` on that slot:
    /// (slot, [(32-bit register?, register, component, lane, value)])
    pending: Vec<(u64, Vec<(bool, usize, usize, usize, i128)>)>,
    /// AGX_TRACE_LANE: the traced lane's registers at the previous instruction
    lane_regs: Vec<u32>,
}

fn ext(v: i128, bits: u32, signed: bool) -> i128 {
    let m = (1i128 << bits) - 1;
    let x = v & m;
    if signed && x >> (bits - 1) & 1 == 1 { x - (1i128 << bits) } else { x }
}

fn item_size(fmt: u64) -> u64 {
    match fmt {
        1 | 3 | 6 | 7 => 2,
        2 => 4,
        _ => 1,
    }
}

/// Device memory formats 0..7 (coverage mem_load / mem_store, recorded on the M2):
/// - 0 i8, 1 i16, 2 i32 and 3 "f16" are raw: zero-extended into a wider register, truncated on store. Format 3 is a
///   16-bit pass-through like i16 (no f16 <-> f32 conversion even with a 32-bit register);
/// - 4 u8norm, 5 s8norm, 6 u16norm, 7 s16norm are normalised: see `norm_load` / `norm_store`.
fn norm_format(fmt: u64) -> bool {
    matches!(fmt, 4..=7)
}

/// Normalised load: the exact ratio v/255, max(v,-127)/127, v/65535 or max(v,-32767)/32767 rounded once
/// (nearest-even) to the destination's precision, f32 in a 32-bit register or f16 in a half, subnormals kept. The
/// rounding is direct, not via f32: s16norm 0x8009 into a half gives 0xbbff where f32-then-f16 would give -1.0.
/// (f64 division then one conversion is exact here: an odd denominator below 2^16 keeps the ratio further from
/// every f16 / f32 rounding boundary than the f64 error.)
fn norm_load(fmt: u64, v: u64, rt: bool) -> i128 {
    let x = match fmt {
        4 => v as f64 / 255.0,
        5 => (v as u8 as i8 as f64).max(-127.0) / 127.0,
        6 => v as f64 / 65535.0,
        _ => (v as u16 as i16 as f64).max(-32767.0) / 32767.0,
    };
    if rt { (x as f32).to_bits() as i128 } else { crate::isa::fma::f64_to_f16(x.to_bits(), false) as i128 }
}

/// Normalised store: the register read as f32 (32-bit) or f16 (half), NaN -> 0, clamped to [0, 1] or [-1, 1],
/// scaled by 255 / 127 / 65535 / 32767 and rounded to the nearest integer (-1 gives -127 / -32767; the odd scales
/// make a tie impossible). The product is exact in f64. A subnormal f16 source is flushed to zero (u16norm from
/// 0x0311 stores 0, from the smallest normal 0x0411 stores 4); whether an f32 subnormal is flushed is invisible
/// (it scales to below 0.5 either way).
fn norm_store(fmt: u64, v: i128, rt: bool) -> u64 {
    let x = if rt {
        f32::from_bits(v as u32) as f64
    } else {
        f64::from_bits(crate::isa::fma::f16_to_f64(v as u64 & 0xffff, true))
    };
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = match fmt {
        4 => (0.0, 255.0),
        5 => (-1.0, 127.0),
        6 => (0.0, 65535.0),
        _ => (-1.0, 32767.0),
    };
    (x.clamp(lo, 1.0) * scale).round() as i64 as u64
}

pub fn run(d: &Dispatch, mem: &mut Memory) -> Result<RunStats, String> {
    run_with_optional_core_map(d, mem, None)
}

/// Replay supplied M2 Max SR20 identities, one per linear workgroup. This is
/// scheduling input, not a placement predictor; SR17 slots remain unchanged.
pub fn run_with_m2_core_map(d: &Dispatch, mem: &mut Memory, cores: &[u32]) -> Result<RunStats, String> {
    run_with_optional_core_map(d, mem, Some(cores))
}

/// Validate before shader/preamble execution or mutations of mapped memory.
pub fn validate_m2_core_map(d: &Dispatch, cores: &[u32]) -> Result<(), String> {
    validate_m2_core_map_geometry(d, &validate_dispatch(d)?, cores)
}

fn validate_m2_core_map_geometry(d: &Dispatch, geometry: &Geometry, cores: &[u32]) -> Result<(), String> {
    let count = geometry.dims.iter().fold(1u128, |n, &v| n * v as u128);
    if cores.len() as u128 != count {
        return Err(format!("M2 core map has {} entries, expected {} workgroups", cores.len(), count));
    }
    const MASKS: [u32; 4] = [0x3ff, 0x1ff, 0x1ff, 0x3ff];
    for (i, &core) in cores.iter().enumerate() {
        let cluster = (core >> 4) as usize;
        if cluster >= MASKS.len() || MASKS[cluster] & (1 << (core & 15)) == 0 {
            return Err(format!("M2 core map workgroup {} has disabled core ID {}", i, core));
        }
    }
    // Observed on 1-D, whole-workgroup grids at these four local sizes.
    // Do not impose this spatial constraint on unmeasured geometries.
    if d.grid[1..] == [1, 1] && d.wg[1..] == [1, 1]
        && matches!(d.wg[0], 32 | 64 | 128 | 256) && d.grid[0] % d.wg[0] == 0 {
        for (i, block) in cores.chunks(8).enumerate() {
            if block.iter().any(|core| core >> 4 != block[0] >> 4) {
                return Err(format!("M2 core map aligned eight-workgroup block {} spans clusters", i));
            }
        }
    }
    Ok(())
}

fn run_with_optional_core_map(d: &Dispatch, mem: &mut Memory, cores: Option<&[u32]>) -> Result<RunStats, String> {
    let geometry = validate_dispatch(d)?;
    if let Some(cores) = cores { validate_m2_core_map_geometry(d, &geometry, cores)?; }
    if geometry.dims.contains(&0) { return Ok(RunStats::default()); }
    let cache = code_cache(d.code.len());
    let dims = geometry.dims;
    let mut stats = RunStats::default();
    for wz in 0..dims[2] {
        for wy in 0..dims[1] {
            for wx in 0..dims[0] {
                let core = cores.map(|c| c[geometry.linear([wx, wy, wz]) as usize]);
                run_workgroup(d, &geometry, &cache, mem, [wx, wy, wz], core, false, &mut stats)?;
            }
        }
    }
    Ok(stats)
}

/// Dynamic path of every SIMD group of the given workgroups (each run alone from `mem` as it is, with its own copy),
/// for the cost model: exact control flow and addresses, whatever the data.
pub fn trace_workgroups(d: &Dispatch, mem: &Memory, wgs: &[[u32; 3]]) -> Result<Vec<Vec<Trace>>, String> {
    PreparedDispatch::new(d)?.trace_workgroups(mem, wgs)
}

/// Trace with supplied M2 Max SR20 identities. Each workgroup still receives
/// an isolated memory snapshot; this does not replay concurrent execution.
pub fn trace_workgroups_with_m2_core_map(d: &Dispatch, mem: &Memory, wgs: &[[u32; 3]], cores: &[u32]) -> Result<Vec<Vec<Trace>>, String> {
    PreparedDispatch::with_m2_core_map(d, cores)?.trace_workgroups(mem, wgs)
}

/// Reusable, thread-safe lazy decoding for functional traces of one immutable dispatch.
/// Only instruction metadata is shared: every trace starts with fresh registers,
/// shared memory, execution masks and a copy-on-write snapshot of the supplied memory.
pub struct PreparedDispatch<'a> {
    dispatch: &'a Dispatch,
    geometry: Geometry,
    cache: CodeCache,
    cores: Option<&'a [u32]>,
}

impl<'a> PreparedDispatch<'a> {
    pub fn new(dispatch: &'a Dispatch) -> Result<Self, String> {
        Self::prepare(dispatch, None)
    }

    /// Validate and borrow a fixed observed map for all subsequent traces.
    /// Values are resolved for each requested workgroup, never cached in the
    /// shared instruction metadata. SR17 behavior remains selected by Dispatch.
    pub fn with_m2_core_map(dispatch: &'a Dispatch, cores: &'a [u32]) -> Result<Self, String> {
        Self::prepare(dispatch, Some(cores))
    }

    fn prepare(dispatch: &'a Dispatch, cores: Option<&'a [u32]>) -> Result<Self, String> {
        let geometry = validate_dispatch(dispatch)?;
        if let Some(cores) = cores { validate_m2_core_map_geometry(dispatch, &geometry, cores)?; }
        Ok(Self { dispatch, geometry, cache: code_cache(dispatch.code.len()), cores })
    }

    pub fn trace_workgroup(&self, mem: &Memory, wg: [u32; 3]) -> Result<Vec<Trace>, String> {
        self.geometry.check_workgroup(wg)?;
        let core = self.cores.map(|c| c[self.geometry.linear(wg) as usize]);
        let mut m = mem.clone();
        run_workgroup(self.dispatch, &self.geometry, &self.cache, &mut m, wg, core, true, &mut RunStats::default())
    }

    pub fn trace_workgroups(&self, mem: &Memory, wgs: &[[u32; 3]]) -> Result<Vec<Vec<Trace>>, String> {
        // Reject the entire request before starting workers or copying memory.
        for &wg in wgs { self.geometry.check_workgroup(wg)?; }
        crate::par::map(wgs, |&wg| self.trace_workgroup(mem, wg)).into_iter().collect()
    }
}

struct Geometry {
    dims: [u32; 3],
    groups: u32,
}

impl Geometry {
    fn check_workgroup(&self, wg: [u32; 3]) -> Result<(), String> {
        if (0..3).any(|i| wg[i] >= self.dims[i]) {
            return Err(format!("workgroup {:?} is outside dispatch dimensions {:?}", wg, self.dims));
        }
        Ok(())
    }

    fn linear(&self, [x, y, z]: [u32; 3]) -> u64 {
        x as u64 + self.dims[0] as u64 * (y as u64 + self.dims[1] as u64 * z as u64)
    }
}

#[cfg(test)]
#[test]
fn geometry_linear_and_placement_keep_high_bits() {
    let geometry = Geometry { dims: [65536; 3], groups: 32 };
    let w = geometry.linear([65535; 3]);
    assert_eq!(w, (1u64 << 48) - 1);
    assert_eq!((w << 16) | 31, u64::MAX - 65504);
    let d = Dispatch { code: vec![], grid: [67108864, 65536, 65536], wg: [1024, 1, 1], shared_bytes: 0,
        uniforms: vec![], num_registers: 8, max_steps: 1, placement: u32::MAX, launch: Default::default() };
    let (core, slot) = placement(&d, &geometry, w, 31);
    assert_eq!(core, 1 + ((w as u128 * 7 + u32::MAX as u128) % 50) as u32);
    assert_eq!(slot, 31 + 32 * (1 + ((w as u128 + u32::MAX as u128) % 3) as u32));
}

fn validate_dispatch(d: &Dispatch) -> Result<Geometry, String> {
    let threads = d.wg.iter().try_fold(1u32, |n, &v| n.checked_mul(v))
        .ok_or("workgroup size overflows")?;
    if threads == 0 || threads > 1024 {
        return Err("workgroup must contain 1..1024 threads with nonzero dimensions".into());
    }
    if d.uniforms.iter().any(|&(u, _)| u >= 256) {
        return Err("uniform register index exceeds u255".into());
    }
    let dims = std::array::from_fn(|i| d.grid[i].div_ceil(d.wg[i]));
    // Writer tags reserve 16 low bits for the SIMD group. This is an executor
    // representation limit, not a hardware grid limit. Do not silently alias
    // writers (or wrap placement arithmetic) on enormous synthetic dispatches.
    let count = dims.iter().fold(1u128, |n, &v| n * v as u128);
    if count > (1u128 << 48) {
        return Err("dispatch exceeds the functional executor's 48-bit workgroup index capacity".into());
    }
    Ok(Geometry { dims, groups: threads.div_ceil(32) })
}

/// (sr20 core id, sr17 hardware slot) of SIMD group `g` of a workgroup under `d.placement`. The hardware gives
/// every workgroup some core (SR20 = cluster << 4 | core) and its groups slots that equal the group index only
/// while the core holds nothing else (playground sr-geometry-core-id, sr17-sr32).
fn placement(d: &Dispatch, geometry: &Geometry, w: u64, g: u32) -> (u32, u32) {
    if d.placement == 0 {
        return (0, g);
    }
    let gpw = geometry.groups;
    let several = geometry.dims.iter().any(|&n| n > 1);
    let core = 1 + ((w * 7 + d.placement as u64) % 50) as u32;
    let slot = if several { g + gpw * (1 + ((w + d.placement as u64) % 3) as u32) } else { g };
    (core, slot)
}

fn run_workgroup(d: &Dispatch, geometry: &Geometry, cache: &CodeCache, mem: &mut Memory, [wx, wy, wz]: [u32; 3], core_override: Option<u32>, record: bool,
                 stats: &mut RunStats) -> Result<Vec<Trace>, String> {
    let code = &d.code;
    let gpw = geometry.groups;
    let linear = geometry.linear([wx, wy, wz]);
    let writer_base = linear << 16;
    let mut tg = vec![0u8; d.shared_bytes.max(16) as usize];
    let mut groups: Vec<Group> = (0..gpw)
        .map(|g| {
            let (core, slot) = placement(d, geometry, linear, g);
            let core = core_override.unwrap_or(core);
            let ctx = GroupCtx { wg_size: d.wg, wg_id: [wx, wy, wz], group: g, core, slot, uniforms: d.uniforms.clone(), launch: d.launch };
            let mut st = CoreState::new(d.num_registers, None);
            let lanes = ctx.lanes();
            for t in 0..32 {
                // lanes beyond the grid (partial workgroups at the edge) do not exist
                let i = 32 * g + t as u32;
                let local = [i % d.wg[0], (i / d.wg[0]) % d.wg[1], i / (d.wg[0] * d.wg[1])];
                let exists = t < lanes && [wx, wy, wz].iter().enumerate().all(|(axis, &w)|
                    w as u64 * d.wg[axis] as u64 + (local[axis] as u64) < d.grid[axis] as u64);
                st.live[t] = exists;
                st.exec[t] = exists;
                let _ = st.set_reg16(0, t, if exists { 0 } else { 1 });
            }
            for &(u, v) in &d.uniforms {
                st.uniforms[2 * u] = (v & 0xffff) as u16;
                st.uniforms[2 * u + 1] = (v >> 16) as u16;
            }
            Group { st, ctx, pc: 0, done: false, steps: 0, trace: if record { Some(Trace::default()) } else { None }, stack: vec![], pending: vec![], lane_regs: vec![] }
        })
        .collect();
    stats.groups += groups.len() as u64;
    // run groups to their next barrier; release when every live group of the scope is there
    loop {
        let mut at_barrier: Vec<Option<u8>> = vec![None; groups.len()];
        for (gi, g) in groups.iter_mut().enumerate() {
            if g.done {
                continue;
            }
            mem.writer = writer_base | gi as u64;
            match step_group(g, code, cache, mem, &mut tg, d.max_steps, stats)? {
                Stop::Done => g.done = true,
                Stop::Barrier(k) => at_barrier[gi] = Some(k),
            }
        }
        if groups.iter().all(|g| g.done) {
            break;
        }
        // every group is either done or waiting at a barrier: release all (0x68 and 0x78 alike, since a
        // sequential run cannot reorder within a block)
        if at_barrier.iter().zip(&groups).any(|(b, g)| b.is_none() && !g.done) {
            return Err("internal: group neither done nor at a barrier".into());
        }
    }
    Ok(groups.into_iter().filter_map(|g| g.trace).collect())
}

/// `AGX_TRACE_LANE=GROUP:LANE` (with a trace: agx_run --trace-pcs): prints to stderr, before each instruction of
/// that SIMD group of the traced workgroup, its pc, whether the lane is active, and the 32-bit registers of the lane
/// that changed since the previous instruction (`rN=0x...`, as the instruction before wrote them).
fn trace_lane(g: &mut Group) {
    static LANE: std::sync::OnceLock<Option<(u32, usize)>> = std::sync::OnceLock::new();
    let Some((group, lane)) = *LANE.get_or_init(|| {
        let v = std::env::var("AGX_TRACE_LANE").ok()?;
        let (g, l) = v.split_once(':')?;
        Some((g.parse().ok()?, l.parse().ok()?))
    }) else { return };
    if g.ctx.group != group || lane >= 32 {
        return;
    }
    let regs: Vec<u32> = (0..g.st.reg16s.len() / 2).map(|r| g.st.get_reg32(r, lane).unwrap_or(0) as u32).collect();
    let changed: Vec<String> = regs.iter().enumerate()
        .filter(|&(r, v)| g.lane_regs.get(r) != Some(v))
        .map(|(r, v)| format!("r{}={:#x}", r, v)).collect();
    eprintln!("lane pc {:#x} {} {}", g.pc, if g.st.exec[lane] { "on" } else { "off" }, changed.join(" "));
    g.lane_regs = regs;
}

/// Link-register base for call / ret: the value `call` stores is `call_base() + offset of the next instruction`.
/// 0x2000 on G14C (hardware probes C1a-C1c, playground novel-results/2026-10-17-call-ret).
fn call_base() -> i128 {
    std::env::var("AGX_CALL_BASE").ok().and_then(|v| i128::from_str_radix(v.trim_start_matches("0x"), if v.starts_with("0x") { 16 } else { 10 }).ok()).unwrap_or(0x2000)
}

/// In strict mode (`isa::mesa`), fails on an instruction Mesa's compiler would not emit.
pub(crate) fn mesa_check(code: &[u8], pc: usize, size: usize, name: &str) -> Result<(), String> {
    match isa::mesa::mode() {
        Some(g) => isa::mesa::check(&code[pc..pc + size], g)
            .map(|_| ())
            .map_err(|e| format!("{} at {} is not an instruction Mesa emits for {:?} (--agx-strict): {}", name, pc, g, e)),
        None => Ok(()),
    }
}

fn decode_at(code: &[u8], pc: usize, nregs: usize) -> Result<Info, String> {
    if pc % 2 != 0 { return Err(format!("unaligned instruction address {}", pc)); }
    if is_barrier78(code, pc) {
        mesa_check(code, pc, 2, "barrier_4groups")?;
        return Ok(Info { desc: None, size: 2, fields: vec![], op: None, vals: vec![], kind: OpKind::Nop, access: None });
    }
    let n0 = isa::opcode_to_number(&code[pc..code.len().min(pc + 16)]);
    let desc = isa::get_instruction_descriptor(n0).ok_or_else(|| format!("undecodable instruction at {}", pc))?;
    let size = desc.decode_size(n0);
    if size > code.len() - pc {
        return Err(format!("truncated {} at {}: needs {} bytes, has {}", desc.name, pc, size, code.len() - pc));
    }
    mesa_check(code, pc, size, desc.name)?;
    let n = desc.mask_instr(n0);
    let fields = desc.decode_fields(n);
    let ops = if desc.name == "stop" || desc.name == "trap" { vec![] } else { desc.fields_to_operands(&fields)? };
    let op = Compiled::new(desc, &fields, &ops, nregs);
    let vals = vals(&ops);
    let kind = op_kind(desc.name, &fields, &ops);
    let access = crate::memtrace::Access::decode(desc.name, &fields, true);
    Ok(Info { desc: Some(desc), size, fields, op: Some(op), vals, kind, access })
}

/// Where the traces place stack memory: no device address, and not part of a launch's device footprint.
const STACK_TRACE_BASE: u64 = 0x7c00_0000_0000;

fn step_group(g: &mut Group, code: &[u8], cache: &CodeCache, mem: &mut Memory, tg: &mut [u8], max_steps: u64,
              stats: &mut RunStats) -> Result<Stop, String> {
    // Select once per group/resume, not on every instruction. Small programs
    // retain their original dense hot loop, while padded/large programs page.
    let r = match cache {
        CodeCache::Dense(c) => step_group_impl(g, code, c, mem, tg, max_steps, stats),
        CodeCache::Paged(c) => step_group_impl(g, code, c, mem, tg, max_steps, stats),
    };
    // Every error names the PC of the instruction that failed (agx_run --lines maps it to a source line).
    r.map_err(|e| if e.contains(" at pc ") { e } else { format!("{} at pc {}", e, g.pc) })
}

// Keep the large interpreter loop out of workgroup setup/barrier scheduling.
// Checked CPU setup/SR/memory controls are retained in the geometry benchmark.
#[inline(never)]
fn step_group_impl<C: InstructionCells>(g: &mut Group, code: &[u8], cache: &C, mem: &mut Memory, tg: &mut [u8], max_steps: u64,
                                       stats: &mut RunStats) -> Result<Stop, String> {
    if !g.st.live.iter().any(|&live| live) { return Ok(Stop::Done); }
    let nregs = g.st.reg16s.len(); // 16-bit registers, as Compiled::new expects
    while g.pc < code.len() {
        if g.steps >= max_steps {
            return Err("step limit (non-terminating loop?)".into());
        }
        g.steps += 1;
        stats.instructions += 1;
        let cell = cache.cell(g.pc);
        let info = match cell.get() {
            Some(info) => info,
            None => {
                let info = Box::new(decode_at(code, g.pc, nregs)?);
                // Another tracing worker may have decoded this PC concurrently.
                // Both decodes are identical; retain the first successfully installed one.
                let _ = cell.set(info);
                cell.get().unwrap()
            }
        };
        if g.trace.is_some() {
            trace_lane(g);
        }
        if let Some(tr) = g.trace.as_mut() {
            // as cost::trace_group records it: every pc (barriers, jumps and the stop included) and, before the
            // access executes, the device segments / threadgroup words / atomic words of the active lanes
            if !g.st.exec.iter().any(|&e| e) {
                tr.masked.push(tr.pcs.len() as u32);
            }
            tr.pcs.push(g.pc);
            let at = (tr.pcs.len() - 1) as u32;
            let touched = info.access.as_ref().and_then(|a| a.capture(&g.st));
            if matches!(info.kind, OpKind::Stack { .. }) {
                // stack memory, for the memory model: each SIMD group's block holds a slot per 32-bit word with one 4-byte
                // cell per lane (zluda_agx's scratch layout, agx_runtime scratch.rs), so a word spans 128 bytes
                let x = |k: &str| fget(&info.fields, k).unwrap_or(0);
                let mask = x("mask");
                let item = match x("F") { 0 => 1usize, 1 => 2, _ => 4 };
                let block = STACK_TRACE_BASE + ((g.ctx.slot as u64) << 28);
                let mut segs = vec![];
                for t in 0..32 {
                    if !g.st.exec[t] { continue; }
                    let base = if x("Ot") != 0 { x("O") as usize } else { g.st.get_reg32((x("O") >> 1) as usize, t)? as u32 as usize };
                    for c in 0..4 {
                        if mask & (1 << c) != 0 {
                            let a = base + c * item;
                            segs.push((block + (a as u64 / 4) * 128 + t as u64 * 4 + a as u64 % 4) >> 6);
                        }
                    }
                }
                segs.sort_unstable();
                segs.dedup();
                tr.mem.push((at, segs));
            }
            if let Some(w) = touched {
                if let Some(segments) = info.access.as_ref().and_then(|a| a.device_segments(&g.st, &w)) {
                    tr.device_segments.push((at, segments));
                }
                tr.mem.push((at, w));
            }
        }
        let Some(desc) = info.desc else {
            g.st.check_overflow_activation(false).map_err(str::to_string)?;
            g.pc += 2; // 0x78 barrier
            return Ok(Stop::Barrier(2));
        };
        let f = &info.fields;
        let x = |k: &str| fget(f, k).unwrap_or(0);
        let v = &info.vals;
        if !matches!(info.kind, OpKind::Generic) {
            g.st.check_overflow_activation(matches!(info.kind, OpKind::Wait) && x("i") == 0)
                .map_err(|e| format!("{} at pc {}", e, g.pc))?;
        }
        match info.kind {
            OpKind::Stop => {
                // G14C empty-stop and ret-prepop hardware matrices (2026-10-20):
                // stop ends issuing lanes; masked survivors continue through it.
                // An empty stop is a no-op, independent of nesting depth.
                for t in 0..32 {
                    if g.st.exec[t] {
                        g.st.live[t] = false;
                        g.st.exec[t] = false;
                    }
                }
                if !g.st.live.iter().any(|&live| live) {
                    return Ok(Stop::Done);
                }
            }
            OpKind::Jmp { any: want_any } => {
                let any = g.st.exec.iter().any(|&e| e);
                let take = if want_any { any } else { !any };
                let mut off = x("off") as i64;
                if off & (1 << 31) != 0 {
                    off -= 1 << 32;
                }
                if take {
                    let target = g.pc as i128 + off as i128;
                    if target < 0 || target >= code.len() as i128 || target % 2 != 0 {
                        return Err(format!("jump at {} targets {} outside the program or unaligned", g.pc, target));
                    }
                    g.pc = target as usize;
                } else {
                    g.pc += info.size;
                }
                continue;
            }
            // Direct call: measured by C1a-C1c; ret is unresolved (C2b/C2c contradict the old model). The link
            // register is r1; the stored value is AGX_CALL_BASE + the byte offset of the next instruction. The base is 0x2000, measured
            // on G14C (stage 1 probes). A target outside the program is an error, never a jump.
            OpKind::Call => {
                let indirect = info.size == 2;
                // Measured on G14C (empty-mask probes 2026-10-17): with no active
                // lane, call (direct and register) falls through without
                // transferring or writing the link — like ret with an empty mask.
                if !g.st.exec.iter().any(|&active| active) {
                    g.pc += info.size;
                    continue;
                }
                if indirect && std::env::var("AGX_EXPERIMENTAL_RET").as_deref() != Ok("buffer-pop2") {
                    return Err("call (register form): semantics unknown, not modelled".into());
                }
                let mut off = x("off") as i64;
                if off & (1 << 31) != 0 {
                    off -= 1 << 32;
                }
                let target = if indirect {
                    // G14C hardware, playground 2026-10-20 call-winner and
                    // sparse-control: the lowest active lane selects the PC.
                    // Every issue-active lane receives the link, including
                    // nonmatching lanes, which then travel to the callee masked.
                    // Inactive lanes neither select the target nor receive r1.
                    let r = x("reg32") as usize;
                    let link = call_base() + (g.pc + info.size) as i128;
                    for t in 0..32 {
                        if g.st.exec[t] {
                            g.st.set_reg32(1, t, link)?;
                        }
                    }
                    // G14C call-r1 probes (2026-10-20): the implicit link is
                    // visible to target selection and matching. call r1 falls
                    // through with the original issue-active mask unchanged.
                    let first = g.st.exec.iter().position(|&active| active).unwrap();
                    let value = g.st.get_reg32(r, first)?;
                    for t in 0..32 {
                        if g.st.exec[t] {
                            if g.st.get_reg32(r, t)? != value {
                                g.st.set_reg16(0, t, 1)?;
                                g.st.exec[t] = false;
                            }
                        }
                    }
                    (value - call_base()) as i64
                } else { g.pc as i64 + off };
                if target < 0 || target as usize >= code.len() || target % 2 != 0 {
                    return Err(format!("call at {} targets {} outside the program (0..{})", g.pc, target, code.len()));
                }
                if !indirect {
                    let link = call_base() + (g.pc + info.size) as i128;
                    for t in 0..32 {
                        if g.st.exec[t] {
                            g.st.set_reg32(1, t, link)?;
                        }
                    }
                }
                g.pc = target as usize;
                continue;
            }
            OpKind::Ret => {
                let ret_model = std::env::var("AGX_EXPERIMENTAL_RET").unwrap_or_default();
                if ret_model != "1" && ret_model != "buffer-pop2" {
                    return Err("ret semantics unresolved without opt-in: use AGX_EXPERIMENTAL_RET=buffer-pop2 for the measured buffer-relative target, lowest-active winner and mask-depth-2 model; value 1 retains the contradicted code-relative hypothesis for offline study only".into());
                }
                let r = x("reg32") as usize;
                let mut first_active: Option<i128> = None;
                for t in 0..32 {
                    if g.st.exec[t] {
                        first_active = Some(g.st.get_reg32(r, t)? & 0xffff_ffff);
                        break;
                    }
                }
                // Measured on G14C (2026-10-17):
                // - Empty active mask falls through.
                // - Lowest active lane selects rX even when active operands
                //   disagree (G14C 2026-10-20 suffix and scattered-mask matrix).
                // - Nonempty return maps depth 0 -> 2, depth 1 -> 0, and
                //   leaves deeper live lanes unchanged (measured through depth 4).
                //   Empty-issued return does not transform any counters.
                let v = match first_active {
                    None => {
                        g.pc += info.size; // no active lane: falls through
                        continue;
                    }
                    Some(a) => a,
                };
                let target = if ret_model == "buffer-pop2" { v - call_base() } else { v };
                if target < 0 || target as usize >= code.len() || target % 2 != 0 {
                    return Err(format!("ret at {} targets {} outside the program (0..{})", g.pc, target, code.len()));
                }
                if ret_model == "buffer-pop2" {
                    // G14C stale-execution probes (2026-10-20): ordinary r0
                    // writes can decouple the counter and issuing mask. RET
                    // updates counter 0/1 and their execution bits conditionally;
                    // deeper counters retain even a stale active bit. Selection
                    // and empty-issue fallthrough above still use issuing exec.
                    for t in 0..32 {
                        if !g.st.live[t] { continue; }
                        match g.st.get_reg16(0, t)? {
                            0 => {
                                g.st.set_reg16(0, t, 2)?;
                                g.st.exec[t] = false;
                            }
                            1 => {
                                g.st.set_reg16(0, t, 0)?;
                                g.st.exec[t] = true;
                            }
                            _ => {}
                        }
                    }
                }
                g.pc = target as usize;
                continue;
            }
            OpKind::TgBarrier => {
                g.pc += info.size;
                return Ok(Stop::Barrier(1));
            }
            OpKind::Nop => {}
            OpKind::Wait => {
                // a wait on slot s completes the loads issued on s (in issue order)
                let slot = x("i");
                let mut keep = vec![];
                for (s, writes) in std::mem::take(&mut g.pending) {
                    if s == slot {
                        for (rt, r, k, t, v) in writes {
                            put(&mut g.st, rt, r, k, t, v)?;
                        }
                    } else {
                        keep.push((s, writes));
                    }
                }
                g.pending = keep;
            }
            OpKind::GetSr(ref args) => special_register(g, args, stats)?,
            OpKind::DevMem { load, ref args } => device_mem(g, args, mem, load)?,
            OpKind::Stack { load } => stack_mem(g, f, load)?,
            OpKind::UniformStore => uniform_store(g, f)?,
            OpKind::TgMem { load, ref args } => tg_mem(g, args, tg, load)?,
            // Measured only with Mesa's u5=2: ops 11..15 end the whole issuing group,
            // including inactive lanes, without touching memory.
            OpKind::Atomic(ref args) => {
                if args.terminate { return Ok(Stop::Done); }
                device_atomic(g, args, mem)?;
            }
            OpKind::FloatArith => fsem::float_arith(desc.name, f, v, &mut g.st)?,
            OpKind::CmpSel { float } => fsem::cmpsel(float, f, v, &mut g.st)?,
            OpKind::Ballot { float, quad } => fsem::ballot(float, quad, f, v, &mut g.st)?,
            OpKind::MaskCtl { kind, float } => fsem::mask_ctl(kind, float, f, v, &mut g.st)?,
            OpKind::TgAtomic(ref args) => tg_atomic(g, args, tg)?,
            OpKind::TexStateStore { ts } => tex_state_store(g, mem, f, ts)?,
            OpKind::SampStateStore { ss } => samp_state_store(g, mem, f, ss)?,
            OpKind::TexLoad { ref args } => tex_load(g, mem, f, args.as_ref())?,
            OpKind::TexSample { ref args } => tex_sample(g, mem, f, args.as_ref())?,
            OpKind::ImageWrite { ref args } => image_write(g, mem, f, args.as_ref())?,
            OpKind::Round => fsem::round_op(desc.name, v, &mut g.st)?,
            OpKind::Intl => fsem::intl(v, &mut g.st)?,
            OpKind::Transcendental => fsem::transcendental(desc.name, f, v, &mut g.st)?,
            OpKind::Convert => fsem::convert(f, v, &mut g.st)?,
            OpKind::QuadShuffle => fsem::quad_shuffle(desc.name, v, &mut g.st)?,
            OpKind::CrossLane => fsem::cross_lane(desc.name, v, &mut g.st)?,
            OpKind::Generic => {
                let pc = g.pc;
                info.op.as_ref().unwrap().exec(&mut g.st).map_err(|e| {
                    let bytes: String = code[pc..pc + info.size].iter().map(|b| format!("{:02x}", b)).collect();
                    format!("{} at {:#x} ({}): {}", desc.name, pc, bytes, e)
                })?;
            }
        }
        g.pc += info.size;
    }
    if std::env::var("AGX_EXPERIMENTAL_RET").as_deref() == Ok("buffer-pop2") {
        return Err("fell off end of program without an active stop (return needs reconvergence?)".into());
    }
    Ok(Stop::Done)
}

/// stack_store / stack_load: per-lane private memory (the driver's register spills and private arrays). Byte address
/// = offset (16-bit immediate, or the 32-bit register O/2 of the lane) + component * item size; i8 / i16 / i32 items
/// from consecutive registers (32-bit) or halves (16-bit). A mask of 0 accesses nothing (the program's stack setup,
/// whose immediate is the scratch size).
fn stack_mem(g: &mut Group, f: &Fields, load: bool) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    let mask = x("mask");
    if mask == 0 {
        return Ok(());
    }
    let item = match x("F") {
        0 => 1usize,
        1 => 2,
        2 => 4,
        v => return Err(format!("stack format {} not implemented", v)),
    };
    let (r, rt) = (x("R") as usize, x("Rt") != 0);
    if g.stack.is_empty() {
        g.stack = vec![vec![]; 32];
    }
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let base = if x("Ot") != 0 { x("O") as usize } else { g.st.get_reg32((x("O") >> 1) as usize, t)? as u32 as usize };
        let mut k = 0;
        for c in 0..4 {
            if mask & (1 << c) == 0 {
                continue;
            }
            let a = base + c * item;
            if a + item > 1 << 20 {
                return Err(format!("stack access beyond 1 MiB ({:#x})", a));
            }
            let lane = &mut g.stack[t];
            if lane.len() < a + item {
                lane.resize(a + item, 0);
            }
            if load {
                let v = (0..item).fold(0i128, |v, b| v | (lane[a + b] as i128) << (8 * b));
                put(&mut g.st, rt, r, k, t, v)?;
            } else {
                let v = get(&g.st, rt, r, k, t)?;
                for b in 0..item {
                    lane[a + b] = (v >> (8 * b)) as u8;
                }
            }
            k += 1;
        }
    }
    Ok(())
}

/// uniform_store (the driver preamble's way of filling the uniforms from the root table): components of R from the
/// first active lane into uniform halves O<<s + c*(1 or 2), for i16 / i32 formats.
fn uniform_store(g: &mut Group, f: &Fields) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    let fmt = x("F");
    if fmt != 1 && fmt != 2 {
        return Err(format!("uniform_store format {} not implemented", fmt));
    }
    let Some(t) = (0..32).find(|&t| g.st.exec[t]) else { return Ok(()) };
    let idx = if x("Ot") != 0 { x("O") as usize } else { g.st.get_reg32((x("O") >> 1) as usize, t)? as u32 as usize } << x("s");
    let (r, rt) = (x("R") as usize, x("Rt") != 0);
    let halves = if fmt == 2 { 2 } else { 1 };
    let mut k = 0;
    for c in 0..4 {
        if x("mask") & (1 << c) == 0 {
            continue;
        }
        let v = get(&g.st, rt, r, k, t)?;
        k += 1;
        for h in 0..halves {
            let u = idx + c * halves + h;
            if u >= g.st.uniforms.len() {
                return Err(format!("uniform_store beyond the uniform file ({})", u));
            }
            g.st.uniforms[u] = (v >> (16 * h)) as u16;
        }
    }
    Ok(())
}

/// Run a driver preamble (Mesa's once-per-dispatch program that loads the root table into uniforms) as a single
/// lane with the given 32-bit uniforms; returns every 32-bit uniform afterwards, for the main shader's Dispatch.
pub fn run_preamble(code: &[u8], uniforms: &[(usize, u32)], mem: &mut Memory, num_registers: usize) -> Result<Vec<(usize, u32)>, String> {
    let ctx = GroupCtx { wg_size: [1, 1, 1], wg_id: [0, 0, 0], group: 0, core: 0, slot: 96, uniforms: uniforms.to_vec(),
                          launch: crate::cost::SrLaunch { preamble: true, ..Default::default() } };
    let mut st = CoreState::new(num_registers, None);
    for t in 0..32 {
        st.live[t] = t == 0;
        st.exec[t] = t == 0;
        let _ = st.set_reg16(0, t, if t == 0 { 0 } else { 1 });
    }
    for &(u, v) in uniforms {
        st.uniforms[2 * u] = (v & 0xffff) as u16;
        st.uniforms[2 * u + 1] = (v >> 16) as u16;
    }
    let mut g = Group { st, ctx, pc: 0, done: false, steps: 0, trace: None, stack: vec![], pending: vec![], lane_regs: vec![] };
    let cache = code_cache(code.len());
    let mut tg = vec![0u8; 16];
    let mut stats = RunStats::default();
    match step_group(&mut g, code, &cache, mem, &mut tg, 1_000_000, &mut stats)? {
        Stop::Done => {}
        Stop::Barrier(_) => return Err("barrier in a preamble".into()),
    }
    Ok((0..g.st.uniforms.len() / 2).map(|u| (u, g.st.uniforms[2 * u] as u32 | (g.st.uniforms[2 * u + 1] as u32) << 16)).collect())
}

/// register for component k of a memory operand: 32-bit (rt) registers or 16-bit halves from half index r
fn put(st: &mut CoreState, rt: bool, r: usize, k: usize, t: usize, v: i128) -> Result<(), String> {
    if rt { st.set_reg32(r / 2 + k, t, v) } else { st.set_reg16(r + k, t, v) }
}

fn get(st: &CoreState, rt: bool, r: usize, k: usize, t: usize) -> Result<i128, String> {
    if rt { st.get_reg32(r / 2 + k, t) } else { st.get_reg16(r + k, t) }
}

// Immutable operand fields are extracted once per visited PC, shared with the decode.
#[derive(Clone, Copy)]
enum SrKind { Uniform, Lane, QuadRank, QuadCount, SimdRank, SimdCount }

// Cache encoding metadata only. Values and active-mask-dependent ranks/counts
// are recomputed at each execution, never retained across groups or resumes.
struct SpecialRegister {
    selector: u32,
    destination: Option<(usize, bool)>, // register index, half-width?
    kind: SrKind,
}

impl SpecialRegister {
    fn new(fields: &Fields, ops: &[Operand]) -> Self {
        let selector = fget(fields, "SR").unwrap_or(0) as u32;
        let destination = match ops.first() {
            Some(Operand::Reg32(r, _)) => Some((*r as usize, false)),
            Some(Operand::Reg16(r, _)) => Some((*r as usize, true)),
            _ => None,
        };
        let kind = match selector {
            56 => SrKind::QuadRank, 57 => SrKind::QuadCount,
            58 => SrKind::SimdRank, 59 => SrKind::SimdCount,
            16 | 48..=52 | 80..=82 => SrKind::Lane,
            _ => SrKind::Uniform,
        };
        Self { selector, destination, kind }
    }
}

fn special_register(g: &mut Group, args: &SpecialRegister, stats: &mut RunStats) -> Result<(), String> {
    use SrKind::*;
    let sr = args.selector;
    stats.placement_read |= sr == 17 || sr == 20;
    let (dr, half) = args.destination.ok_or("get_sr: unexpected destination")?;
    let Some(first) = g.st.exec.iter().position(|&active| active) else { return Ok(()); };
    let mut fallback = None;
    let mut resolve = |t| g.ctx.sr(sr, t).or_else(|| {
        // Keep environment fallback lazy and local to this execution. SR113 is
        // still unmodelled, not frozen into an instruction or dispatch cache.
        *fallback.get_or_insert_with(|| std::env::var("AGX_SR_DEFAULT").ok().and_then(|v| v.parse::<u32>().ok()))
    }).ok_or_else(|| format!("get_sr: special register {} not modelled", sr));
    let uniform = match args.kind {
        Uniform => resolve(first)?,
        SimdCount => g.st.exec.iter().filter(|&&active| active).count() as u32,
        _ => 0,
    };
    // Broadcasts need no per-lane selector-class dispatch. Keep destination
    // writes checked and in lane order, including half-register preservation.
    if matches!(args.kind, Uniform | SimdCount) {
        for t in 0..32 {
            if !g.st.exec[t] { continue; }
            if half { g.st.set_reg16(dr, t, uniform as i128)?; } else { g.st.set_reg32(dr, t, uniform as i128)?; }
        }
        return Ok(());
    }
    let counts: [u32; 8] = if matches!(args.kind, QuadCount) {
        std::array::from_fn(|q| g.st.exec[4 * q..4 * q + 4].iter().filter(|&&active| active).count() as u32)
    } else { [0; 8] };
    let mut ranks = [0u32; 8];
    for t in 0..32 {
        if !g.st.exec[t] { continue; }
        let value = match args.kind {
            Uniform | SimdCount => uniform,
            Lane => resolve(t)?,
            QuadCount => counts[t / 4],
            QuadRank | SimdRank => {
                let q = if matches!(args.kind, QuadRank) { t / 4 } else { 0 };
                let rank = ranks[q]; ranks[q] += 1; rank
            }
        };
        if half { g.st.set_reg16(dr, t, value as i128)?; } else { g.st.set_reg32(dr, t, value as i128)?; }
    }
    Ok(())
}

// No register values, lane addresses or execution masks are cached here.
#[derive(Clone, Copy)]
struct DeviceMem {
    at: u64, a: u64, ot: u64, o: u64, ou: u64,
    fmt: u64, item: u64, shift: u64, mask: u64,
    r: usize, rt: bool, slot: u64,
}

impl DeviceMem {
    fn new(f: &Fields) -> Self {
        let x = |k: &str| fget(f, k).unwrap_or(0);
        let fmt = x("F") | x("Fx") << 3;
        Self { at: x("At"), a: x("A"), ot: x("Ot"), o: x("O"), ou: x("Ou"),
            fmt, item: item_size(fmt), shift: if x("s") == 3 { 2 } else { x("s") },
            mask: x("mask"), r: x("R") as usize, rt: x("Rt") != 0, slot: x("g") }
    }
}

#[derive(Clone, Copy)]
struct ThreadgroupMem {
    at: u64, a: u64, ot: u64, o: u64,
    item: usize, mask: u64, r: usize, rt: bool,
}

impl ThreadgroupMem {
    fn new(f: &Fields) -> Self {
        let x = |k: &str| fget(f, k).unwrap_or(0);
        Self { at: x("At"), a: x("A"), ot: x("Ot"), o: x("O"),
            item: item_size(x("F")) as usize, mask: x("mask"), r: x("R") as usize, rt: x("Rt") != 0 }
    }
}

fn device_mem(g: &mut Group, args: &DeviceMem, mem: &mut Memory, load: bool) -> Result<(), String> {
    let DeviceMem { at: f_at, a: f_a, ot: f_ot, o: f_o, ou: f_ou,
        fmt, item, shift, mask, r, rt, slot } = *args;
    if fmt > 7 {
        return Err(format!("device memory format {} not implemented", fmt));
    }
    // scoreboard slot (applegpu `g`, Mesa scoreboard bit 0): a load on slot 1 writes its registers only at `wait 1`;
    // read before that, the destination keeps its old value (coverage mem_load device_load_1_*: `wait 0` leaves
    // the preloaded 0xdeadbeef). Slot-0 loads complete at once (every program waits on slot 0 before reading).
    let mut deferred = vec![];
    // per-lane element address
    let mut addr = [0i128; 32];
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let base = if f_at != 0 { g.st.ureg64((f_a >> 1) as usize)? } else { g.st.get_reg64((f_a >> 1) as usize, t)? };
        let mut off: i128 = if f_ot != 0 {
            ext(f_o as i128, 16, true)
        } else {
            let o = g.st.get_reg32((f_o >> 1) as usize, t)?;
            if f_ou == 0 { ext(o, 32, true) } else { o }
        };
        off <<= shift;
        addr[t] = (base & !(item as i128 - 1)) + off * item as i128;
    }
    // components in order, all lanes per component (overlapping multi-component stores: mem_store suite cases)
    let mut k = 0;
    for c in 0..4 {
        if mask & (1 << c) == 0 {
            continue;
        }
        // lanes from the highest down: when lanes of one store hit the same address the lowest lane's value stays
        // (coverage geometry: in 2-D workgroups lane 0's row wins over the rows above it in the same SIMD group)
        for t in (0..32).rev() {
            if !g.st.exec[t] {
                continue;
            }
            let va = (addr[t] + c as i128 * item as i128) as u64;
            if load {
                let v = mem.read(va, item as usize)?;
                let v = if norm_format(fmt) { norm_load(fmt, v, rt) } else { v as i128 };
                // every load, in either scoreboard slot, lands at the `wait` on its slot: on the M2 Max a register
                // read after `wait 1` still sees the old value of a slot-0 load (playground results/2026-10-20-b34-stack-and-waits)
                deferred.push((rt, r, k, t, v));
            } else {
                let v = get(&g.st, rt, r, k, t)?;
                let v = if norm_format(fmt) { norm_store(fmt, v, rt) } else { v as u64 };
                mem.store(va, item as usize, v)?;
            }
        }
        k += 1;
    }
    if !deferred.is_empty() {
        g.pending.push((slot, deferred));
    }
    Ok(())
}

fn tg_mem(g: &mut Group, args: &ThreadgroupMem, tg: &mut [u8], load: bool) -> Result<(), String> {
    let ThreadgroupMem { at: f_at, a: f_a, ot: f_ot, o: f_o, item, mask, r, rt } = *args;
    let mut addr = [0usize; 32];
    for t in 0..32 {
        if g.st.exec[t] {
            let base = crate::cost::tg_base(&g.st, f_at, f_a, t)?;
            let idx = if f_ot != 0 { f_o as usize } else { g.st.get_reg16(f_o as usize, t)? as usize };
            addr[t] = base + 4 * item * idx;
        }
    }
    let mut k = 0;
    for c in 0..4 {
        if mask & (1 << c) == 0 {
            continue;
        }
        for t in 0..32 {
            if !g.st.exec[t] {
                continue;
            }
            let b = addr[t] + item * c;
            if b + item > tg.len() {
                // outside the declared threadgroup memory: loads read 0, stores are dropped
                if load {
                    put(&mut g.st, rt, r, k, t, 0)?;
                }
            } else if load {
                let v = (0..item).fold(0i128, |v, i| v | (tg[b + i] as i128) << (8 * i));
                put(&mut g.st, rt, r, k, t, v)?;
            } else {
                let v = get(&g.st, rt, r, k, t)?;
                for i in 0..item {
                    tg[b + i] = (v >> (8 * i)) as u8;
                }
            }
        }
        k += 1;
    }
    Ok(())
}

fn atomic_op(op: u64, old: u32, v: u32, cmp: u32) -> Result<u32, String> {
    Ok(match op {
        0 => old.wrapping_add(v),
        1 => old.wrapping_sub(v),
        2 => v,
        3 => if old == cmp { v } else { old },
        4 => old.min(v),
        5 => (old as i32).min(v as i32) as u32,
        6 => old.max(v),
        7 => (old as i32).max(v as i32) as u32,
        8 => old & v,
        9 => old | v,
        10 => old ^ v,
        _ => return Err(format!("atomic op {} not implemented", op)),
    })
}

// Cache only immutable encoding fields, never operands' register values or addresses.
struct DeviceAtomic {
    at: u64, a: u64, ot: u64, o: u64, ou: u64, op: u64,
    r: usize, ret: bool, s: usize, variant: u64, terminate: bool,
}

impl DeviceAtomic {
    fn new(f: &Fields) -> Self {
        let x = |k: &str| fget(f, k).unwrap_or(0);
        let (op, variant) = (x("op"), x("u5"));
        Self { at: x("At"), a: x("A"), ot: x("Ot"), o: x("O"), ou: x("Ou"), op,
            r: (x("R") | x("Rx") << 6) as usize, ret: x("Rt") != 0, s: x("S") as usize,
            // ops 11-15 end the SIMD group with no memory effect, whatever bits 45-46 hold (measured with 0, 1, 2
            // and 3 on the M2 Max: ptx-re-blockers B24)
            variant, terminate: op >= 11 }
    }
}

struct ThreadgroupAtomic {
    at: u64, a: u64, ot: u64, o: u64, op: u64,
    r: usize, ret: bool, s: usize, suppress: bool,
}

impl ThreadgroupAtomic {
    fn new(f: &Fields) -> Self {
        let x = |k: &str| fget(f, k).unwrap_or(0);
        let op = x("op");
        Self { at: x("At"), a: x("A"), ot: x("Ot"), o: x("O"), op,
            r: x("R") as usize, ret: x("Ra") != 0, s: x("S") as usize,
            suppress: op == 1 && x("u1") == 1 }
    }
}

fn device_atomic(g: &mut Group, args: &DeviceAtomic, mem: &mut Memory) -> Result<(), String> {
    let DeviceAtomic { at: f_at, a: f_a, ot: f_ot, o: f_o, ou: f_ou, op: f_op,
        r, ret, s, variant, .. } = *args;
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let base = if f_at != 0 { g.st.ureg64((f_a >> 1) as usize)? } else { g.st.get_reg64((f_a >> 1) as usize, t)? };
        let off: i128 = if f_ot != 0 {
            ext(f_o as i128, 16, true)
        } else {
            let o = g.st.get_reg32((f_o >> 1) as usize, t)?;
            if f_ou == 0 { ext(o, 32, true) } else { o }
        };
        let va = (base + 4 * off) as u64;
        let result = if f_op == 3 && variant & 1 != 0 {
            // G14C: bit 45 selects u64 min/max; the index still scales by four.
            // Rt returns a changed flag (0 when equal), rather than the old memory value. At an address that is 4
            // mod 8 it is a silent no-op (B24); its flag is taken as 0.
            if va % 8 != 0 {
                mem.read(va, 4)?;
                0
            } else {
                let old = mem.read(va, 8)?;
                let v = g.st.get_reg64(s / 2, t)? as u64;
                let new = if variant & 2 != 0 { old.max(v) } else { old.min(v) };
                mem.write(va, 8, new)?;
                (new != old) as u32
            }
        } else {
            let old = mem.read(va, 4)? as u32;
            let v = g.st.get_reg32(s / 2, t)? as u32;
            let new = if f_op == 1 && variant & 1 != 0 {
                atomic_fadd(old, v)
            } else {
                let cmp = if f_op == 3 { g.st.get_reg32(s / 2 + 1, t)? as u32 } else { 0 };
                atomic_op(f_op, old, v, cmp)?
            };
            mem.write(va, 4, new as u64)?;
            old
        };
        if ret {
            g.st.set_reg32(r / 2, t, result as i128)?;
        }
    }
    Ok(())
}

/// G14C atomic add.f32: nearest-even, signed-zero flush on input and output, canonical NaN.
fn atomic_fadd(a: u32, b: u32) -> u32 {
    let flush = |v: u32| if v & 0x7f80_0000 == 0 { v & 0x8000_0000 } else { v };
    let v = f32::from_bits(flush(a)) + f32::from_bits(flush(b));
    if v.is_nan() { 0x7fc0_0000 } else { flush(v.to_bits()) }
}

fn tg_atomic(g: &mut Group, args: &ThreadgroupAtomic, tg: &mut [u8]) -> Result<(), String> {
    let ThreadgroupAtomic { at: f_at, a: f_a, ot: f_ot, o: f_o, op: f_op, r, ret, s, suppress } = *args;
    // Hardware ends the issuing group even with an empty mask (2026-10-17
    // tg-atomic-followup). Shared-memory side effects are not yet observed by
    // another group, so reject these forms instead of silently skipping them.
    if f_op >= 11 {
        return Err(format!("threadgroup atomic op {}: termination observed, memory side effects not modeled", f_op));
    }
    // Recorded sub + gap bit 6 suppresses the access and returns zero.
    if suppress {
        if ret {
            for t in 0..32 {
                if g.st.exec[t] { g.st.set_reg32(r / 2, t, 0)?; }
            }
        }
        return Ok(());
    }
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let base = crate::cost::tg_base(&g.st, f_at, f_a, t)?;
        let idx = if f_ot != 0 { f_o as usize } else { g.st.get_reg16(f_o as usize, t)? as usize };
        let b = base + 16 * idx;
        if b + 4 > tg.len() {
            return Err(format!("threadgroup atomic outside the declared memory ({} bytes)", tg.len()));
        }
        let old = u32::from_le_bytes([tg[b], tg[b + 1], tg[b + 2], tg[b + 3]]);
        let v = g.st.get_reg32(s / 2, t)? as u32;
        let cmp = if f_op == 3 { g.st.get_reg32(s / 2 + 1, t)? as u32 } else { 0 };
        tg[b..b + 4].copy_from_slice(&atomic_op(f_op, old, v, cmp)?.to_le_bytes());
        if ret {
            g.st.set_reg32(r / 2, t, old as i128)?;
        }
    }
    Ok(())
}

/// Texture unit: state slots, descriptor decode and sampling. Semantics measured on hardware
/// (playground novel-results/2026-10-18-gles-texture, 2026-10-19-texture-edge); descriptor
/// layouts from Mesa cmdbuf.xml (playground 2026-10-19-mesa-texture-desc).

fn ureg64_pair(g: &Group, u: u32) -> Result<u64, String> {
    // 64-bit uniform uN:uN+1 (halves 2N .. 2N+3)
    let h = |i: usize| g.st.uniforms.get(i).copied().unwrap_or(0) as u64;
    let b = (u as usize) * 2;
    Ok(h(b) | (h(b + 1) << 16) | (h(b + 2) << 32) | (h(b + 3) << 48))
}

fn coord_lane(g: &Group, c: &(bool, u32), t: usize) -> Result<i64, String> {
    Ok(if c.0 { g.st.get_reg16(c.1 as usize, t)? } else { g.st.get_reg32(c.1 as usize, t)? } as i64)
}

fn dest_write(g: &mut Group, dest: &[(bool, u32)], t: usize, vals: &[u32]) -> Result<(), String> {
    for (i, v) in vals.iter().enumerate() {
        if i >= dest.len() {
            break;
        }
        let (half, r) = dest[i];
        if half {
            g.st.set_reg16(r as usize, t, *v as i128)?;
        } else {
            g.st.set_reg32(r as usize, t, *v as i128)?;
        }
    }
    Ok(())
}

fn lod_value(g: &Group, args: &TexOp, t: usize) -> Result<f32, String> {
    if let Some(u) = args.lod_ureg {
        let bits = g.st.uniforms.get(2 * u as usize).copied().unwrap_or(0);
        return Ok(crate::texture::f16_to_f32(bits));
    }
    if let Some(v) = args.lod_imm {
        return Ok(v as f32);
    }
    if let Some((half, r)) = args.lod_reg {
        let v = if half { g.st.get_reg16(r as usize, t)? } else { g.st.get_reg32(r as usize, t)? };
        return Ok(v as i32 as f32);
    }
    Ok(0.0)
}

/// Resolve a texture descriptor: slot form (ts# populated by tex_state_store) or bindless
/// direct form (uniform base + per-lane byte offset).
fn resolve_desc(mem: &mut Memory, g: &Group, args: &TexOp, lane: usize) -> Result<crate::texture::TexDesc, String> {
    let bytes: Vec<u8> = if let Some(slot) = args.tslot {
        mem.tex_state
            .get(&slot)
            .ok_or_else(|| format!("texture state slot {} not written (tex_state_store missing?)", slot))?
            .clone()
    } else if let Some(u) = args.bind_ureg {
        let base = ureg64_pair(g, u)?;
        let off = match args.bind_reg {
            Some((half, r)) => {
                if half {
                    g.st.get_reg16(r as usize, lane)?
                } else {
                    g.st.get_reg32(r as usize, lane)?
                }
            }
            None => 0,
        } as u64;
        let va = base.wrapping_add(off);
        mem.read_bytes(va, 24)?
    } else {
        return Err("texture op has neither slot nor bindless base".into());
    };
    let desc = crate::texture::decode_texdesc(&bytes)?;
    if desc.dim != 0 && desc.dim != 2 {
        return Err(format!("only 1D and 2D textures modelled, got dim {}", desc.dim));
    }
    Ok(desc)
}

fn texel_format(desc: &crate::texture::TexDesc) -> Result<(usize, u32), String> {
    crate::texture::tex_format(desc.channels, desc.type_)
}


/// Base address and byte offset of a tex/sampler_state_store: B (bits 58..63) holds the
/// offset-is-immediate flag in bit 58 and U = heap uniform (16-bit halves) >> 2 in bits 59..62
/// (agx_pack.c: U = heap.value >> 2; the heap uniform of set s is halves 4+4s, i.e. 32-bit
/// uniforms 2+2s — hk_shader.c lower_uniforms). The O field is then an immediate in half-words
/// (measured: sampler at set+24 reads O=12) or, when bit 58 is clear, a register holding bytes.
fn state_store_va(g: &Group, f: &Fields) -> Result<(u64, bool), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    let b = x("B");
    let imm = b & 1 != 0;
    let heap_32 = ((b >> 1) & 0xF) << 1; // 16-bit halves >> 2, back to 32-bit uniform units
    let base = g.st.ureg64(heap_32 as usize)? as u64;
    let o = x("O");
    let off = if imm {
        o * 2
    } else {
        // first active lane fixes the address (preambles run single-lane)
        let t = (0..32).find(|&t| g.st.exec[t]).unwrap_or(0);
        g.st.get_reg32(o as usize, t)? as u64
    };
    Ok((base.wrapping_add(off), imm))
}

fn tex_state_store(g: &mut Group, mem: &mut Memory, f: &Fields, ts: u32) -> Result<(), String> {
    let (va, imm) = state_store_va(g, f)?;
    if !imm {
        return Err("tex_state_store with register offset not modelled".into());
    }
    let bytes = mem.read_bytes(va, 24)?;
    mem.tex_state.insert(ts, bytes);
    Ok(())
}

fn samp_state_store(g: &mut Group, mem: &mut Memory, f: &Fields, ss: u32) -> Result<(), String> {
    let (va, imm) = state_store_va(g, f)?;
    if !imm {
        return Err("sampler_state_store with register offset not modelled".into());
    }
    let bytes = mem.read_bytes(va, 8)?;
    mem.samp_state.insert(ss, bytes);
    Ok(())
}

fn tex_load(g: &mut Group, mem: &mut Memory, f: &Fields, args: &TexOp) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    // q3 selects the access: 0b1100 texelFetch, 0b1101 image load (direct texel read, no filtering)
    let q3 = x("q3");
    if q3 != 0b1100 && q3 != 0b1101 {
        return Err(format!("texture_load q3 {:#b} not modelled", q3));
    }
    if args.dest.is_empty() || args.coords.len() < 2 {
        return Err("texture_load needs a destination and xy coordinates".into());
    }
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let desc = resolve_desc(mem, g, args, t)?;
        let (bpp, _) = texel_format(&desc)?;
        let xi = coord_lane(g, &args.coords[0], t)?;
        let yi = coord_lane(g, &args.coords[1], t)?;
        let raw = match crate::texture::texel_off(&desc, bpp, xi, yi) {
            Some(off) => mem.read_bytes(desc.addr + off, bpp)?,
            None => vec![0u8; bpp], // texelFetch out of range reads zero (measured)
        };
        if desc.type_ == 2 || desc.type_ == 3 {
            let v = crate::texture::texel_int(&raw, desc.channels, desc.type_)?;
            let sw = crate::texture::apply_swizzle_u32(v, desc.swizzle);
            dest_write(g, &args.dest, t, &sw[..args.dest.len().min(4)])?;
        } else {
            let v = crate::texture::texel_float(&raw, desc.channels, desc.type_, desc.srgb)?;
            let sw = crate::texture::apply_swizzle(v, desc.swizzle);
            let words: Vec<u32> = sw.iter().map(|f| f.to_bits()).collect();
            dest_write(g, &args.dest, t, &words[..args.dest.len().min(4)])?;
        }
    }
    Ok(())
}

/// Register sampler indices from this one on select the command's sampler heap.
const SAMPLER_HEAP_BASE: i64 = 28;

/// Resolve the sampler for a texture_sample: state slot (ss#), or the register form. A register
/// holds the sampler heap index for the normal path, or the clamp_0 override: negative means no
/// custom border, i.e. the descriptor's own sampler (hk_nir_lower_descriptors lower_tex). With a
/// single populated sampler slot that is unambiguous; anything else needs the sampler heap.
fn resolve_sampler(mem: &mut Memory, g: &Group, args: &TexOp, lane: usize) -> Result<crate::texture::SampDesc, String> {
    if let Some(ss) = args.ss {
        let b = mem
            .samp_state
            .get(&ss)
            .ok_or_else(|| format!("sampler state slot {} not written", ss))?;
        return crate::texture::decode_sampler(b);
    }
    let (half, r) = args.sreg.ok_or_else(|| "texture_sample without sampler slot or register".to_string())?;
    // each lane's own index (bindless samplers may differ between lanes)
    let raw = if half { g.st.get_reg16(r as usize, lane)? } else { g.st.get_reg32(r as usize, lane)? };
    let v = (raw as u16) as i16 as i64;
    if v < 0 {
        // clamp_0 override: -1 means no custom border, i.e. the descriptor's own sampler.
        // The descriptor's sampler is the one the preamble stored; if a single slot exists we
        // use it, otherwise the binding is ambiguous and we error rather than guess.
        if mem.samp_state.len() == 1 {
            let b = mem.samp_state.values().next().unwrap();
            return crate::texture::decode_sampler(b);
        }
        return Err(format!("register sampler index -1 but {} sampler slots populated", mem.samp_state.len()));
    }
    // below 28: a sampler state slot; from 28 on: entry v - 28 of the command's sampler heap (Mesa's Honeykrisp
    // stores heap index + 28, hk_descriptor_set.c; on the M2 Max a texture_sample with heap index 0 as its sampler
    // reads zeros, with 28 the texels: ZLUDA agx_texture_check, 2026-10-23)
    if v < SAMPLER_HEAP_BASE {
        let b = mem
            .samp_state
            .get(&(v as u32))
            .ok_or_else(|| format!("register sampler index {} is sampler state slot {}, not written (heap samplers are {} + index)", v, v, SAMPLER_HEAP_BASE))?;
        return crate::texture::decode_sampler(b);
    }
    let v = v - SAMPLER_HEAP_BASE;
    let Some((heap, count)) = mem.sampler_heap else {
        return Err(format!("register sampler heap index {} but the command has no sampler heap", v));
    };
    if v as u64 >= count as u64 {
        return Err(format!("sampler index {} outside the sampler heap of {}", v, count));
    }
    let b = mem.read_bytes(heap + 8 * v as u64, 8)?;
    crate::texture::decode_sampler(&b)
}

fn tex_sample(g: &mut Group, mem: &mut Memory, f: &Fields, args: &TexOp) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    let q3 = x("q3");
    if q3 != 0b1100 {
        return Err(format!("texture_sample q3 {:#b} not modelled", q3));
    }
    if args.dest.len() < 4 {
        return Err("texture_sample needs a vec4 destination".into());
    }
    if args.coords.is_empty() {
        return Err("texture_sample needs coordinates".into());
    }
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let samp = resolve_sampler(mem, g, args, t)?;
        if samp.compare {
            return Err("depth compare not modelled".into());
        }
        if samp.min_filter > 1 || samp.mag_filter > 1 {
            return Err("only nearest/linear filtering modelled".into());
        }
        let desc = resolve_desc(mem, g, args, t)?;
        if desc.first_level != 0 || desc.last_level != 0 {
            return Err("mipmapped sampling not modelled".into());
        }
        let lod = lod_value(g, args, t)?;
        if lod != 0.0 {
            return Err(format!("nonzero LOD {} not modelled", lod));
        }
        let one_d = desc.dim == 0;
        if one_d && desc.height != 1 {
            return Err("1D texture with a height".into());
        }
        if !one_d && args.coords.len() < 2 {
            return Err("texture_sample needs xy coordinates".into());
        }
        let (bpp, _) = texel_format(&desc)?;
        let uf = f32::from_bits(coord_lane(g, &args.coords[0], t)? as u32);
        // a 1D texture is one row: the centre of it
        let vf = if one_d {
            0.5
        } else {
            f32::from_bits(coord_lane(g, &args.coords[1], t)? as u32)
        };
        let w = desc.width as usize;
        let h = desc.height as usize;
        let total = crate::texture::tex_stride(&desc) as usize * h.max(1);
        let texels = mem.read_bytes(desc.addr, total.max(w * bpp))?;
        let integer = desc.type_ == 2 || desc.type_ == 3;
        if integer && (samp.min_filter != 0 || samp.mag_filter != 0) {
            return Err("linear filtering of an integer texture".into());
        }
        let words: Vec<u32> = if samp.min_filter == 0 && samp.mag_filter == 0 {
            // nearest: floor(u*w), floor(v*h) (floor of the texel coordinate when unnormalized), wrapped
            let (sw, sh) = if samp.pixel { (1.0, 1.0) } else { (w as f32, h as f32) };
            let xi = (uf * sw).floor() as i32;
            let yi = (vf * sh).floor() as i32;
            let ax = crate::texture::wrap_or_border(&samp, samp.wrap[0], xi, w as i32)?;
            let ay = crate::texture::wrap_or_border(&samp, samp.wrap[1], yi, h as i32)?;
            let raw = match (ax, ay) {
                (Some(ax), Some(ay)) => crate::texture::texel_off(&desc, bpp, ax as i64, ay as i64)
                    .map(|off| texels.get(off as usize..off as usize + bpp).unwrap_or(&[][..]).to_vec()),
                _ => None,
            };
            match raw {
                // the border colour (transparent black), not swizzled
                None => vec![0; 4],
                Some(mut raw) => {
                    raw.resize(bpp, 0);
                    if integer {
                        let v = crate::texture::texel_int(&raw, desc.channels, desc.type_)?;
                        crate::texture::apply_swizzle_u32(v, desc.swizzle).to_vec()
                    } else {
                        let v = crate::texture::texel_float(&raw, desc.channels, desc.type_, desc.srgb)?;
                        crate::texture::apply_swizzle(v, desc.swizzle).iter().map(|v| v.to_bits()).collect()
                    }
                }
            }
        } else {
            crate::texture::sample_linear(&desc, &samp, &texels, bpp, uf, vf)?.iter().map(|v| v.to_bits()).collect()
        };
        dest_write(g, &args.dest, t, &words)?;
    }
    Ok(())
}

/// One stored texel as bytes: float formats convert (unorm/snorm round-half-even, float32 direct),
/// integer formats truncate to the element size. Only linear 2D images.
fn store_texel(channels: u32, type_: u32, srgb: bool, comps: &[u32]) -> Result<Vec<u8>, String> {
    if srgb {
        return Err("sRGB image store not modelled".into());
    }
    let (ncomps, esize) = match channels {
        0x00 => (1, 1), // R8
        0x09 => (1, 2), // R16
        0x21 => (1, 4), // R32
        0x23 => (2, 2), // R16G16
        0x28 => (4, 1), // R8G8B8A8
        0x32 => (4, 2), // R16G16B16A16
        0x38 => (4, 4), // R32G32B32A32
        _ => return Err(format!("image store channels {:#x} not modelled", channels)),
    };
    if comps.len() < ncomps {
        return Err(format!("image store needs {} source regs, has {}", ncomps, comps.len()));
    }
    let mut out = vec![0u8; ncomps * esize];
    for i in 0..ncomps {
        let b = &mut out[i * esize..(i + 1) * esize];
        match (type_, esize) {
            (4, 4) => b.copy_from_slice(&comps[i].to_le_bytes()), // float32 bit pattern
            (4, _) => return Err("float16 image store not modelled".into()),
            (0, 1) => b[0] = (f32::from_bits(comps[i]) * 255.0).round_ties_even().clamp(0.0, 255.0) as u8,
            (0, 2) => {
                let v = (f32::from_bits(comps[i]) * 65535.0).round_ties_even().clamp(0.0, 65535.0) as u16;
                b.copy_from_slice(&v.to_le_bytes())
            }
            (1, 1) => {
                let v = (f32::from_bits(comps[i]) * 127.0).round_ties_even().clamp(-128.0, 127.0) as i8;
                b[0] = v as u8
            }
            (2, 1) => b[0] = comps[i] as u8,
            (3, 1) => b[0] = (comps[i] as i32) as i8 as u8,
            (2, 2) => b.copy_from_slice(&(comps[i] as u16).to_le_bytes()),
            (3, 2) => b.copy_from_slice(&((comps[i] as i32) as i16).to_le_bytes()),
            (2, 4) | (3, 4) => b.copy_from_slice(&comps[i].to_le_bytes()),
            _ => return Err(format!("image store type {} esize {} not modelled", type_, esize)),
        }
    }
    Ok(out)
}

fn write_bytes(mem: &mut Memory, mut va: u64, raw: &[u8]) -> Result<(), String> {
    // Memory::write handles 8 bytes at a time
    for chunk in raw.chunks(8) {
        let mut v = 0u64;
        for (i, b) in chunk.iter().enumerate() {
            v |= (*b as u64) << (8 * i);
        }
        mem.write(va, chunk.len(), v)?;
        va += chunk.len() as u64;
    }
    Ok(())
}

/// Image addressing shared by slot form (PBE descriptor, as Mesa emits for stores) and the
/// bindless direct form (texture descriptor at set + offset, offset-in-binding 0 = tex, 24 = pbe).
struct ImgTarget {
    addr: u64,
    stride: u64,
    width: u32,
    height: u32,
    channels: u32,
    type_: u32,
    srgb: bool,
    swizzle: [u32; 4],
}

fn image_target(mem: &mut Memory, g: &Group, args: &TexOp, lane: usize) -> Result<ImgTarget, String> {
    if let Some(slot) = args.tslot {
        let bytes = mem
            .tex_state
            .get(&slot)
            .ok_or_else(|| format!("image_write state slot {} not written", slot))?
            .clone();
        // stores address the PBE descriptor (hk lower_image_intrin: writes go through PBE)
        let d = crate::texture::decode_pbe(&bytes).map_err(|e| format!("slot {} pbe: {}", slot, e))?;
        return Ok(ImgTarget {
            addr: d.addr,
            stride: d.stride,
            width: d.width,
            height: d.height,
            channels: d.channels,
            type_: d.type_,
            srgb: d.srgb,
            swizzle: d.swizzle,
        });
    }
    if let Some(u) = args.bind_ureg {
        let base = ureg64_pair(g, u)?;
        let off = match args.bind_reg {
            Some((half, r)) => {
                if half {
                    g.st.get_reg16(r as usize, lane)?
                } else {
                    g.st.get_reg32(r as usize, lane)?
                }
            }
            None => 0,
        } as u64;
        let va = base.wrapping_add(off);
        // stores go through the PBE half of hk_storage_image_descriptor (tex at +0, PBE at +24)
        let in_binding = off % 48;
        let bytes = if in_binding == 0 {
            mem.read_bytes(va + 24, 24)?
        } else if in_binding == 24 {
            mem.read_bytes(va, 24)?
        } else {
            return Err(format!("image_write direct offset {} not tex/pbe aligned", off));
        };
        let d = crate::texture::decode_pbe(&bytes)?;
        return Ok(ImgTarget {
            addr: d.addr,
            stride: d.stride,
            width: d.width,
            height: d.height,
            channels: d.channels,
            type_: d.type_,
            srgb: d.srgb,
            swizzle: d.swizzle,
        });
    }
    Err("image_write has neither slot nor bindless base".into())
}

fn image_write(g: &mut Group, mem: &mut Memory, f: &Fields, args: &TexOp) -> Result<(), String> {
    // source regs = dest field position (R holds the values to store)
    if args.dest.is_empty() || args.coords.len() < 2 {
        return Err("image_write needs source regs and xy coordinates".into());
    }
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let img = image_target(mem, g, args, t)?;
        if img.swizzle != [0, 1, 2, 3] {
            return Err("image_write with swizzle not modelled".into());
        }
        let xi = coord_lane(g, &args.coords[0], t)?;
        let yi = coord_lane(g, &args.coords[1], t)?;
        if xi < 0 || yi < 0 || xi >= img.width as i64 || yi >= img.height as i64 {
            return Err("image_write out of range".into());
        }
        let (bpp, _) = crate::texture::tex_format(img.channels, img.type_)?;
        let off = yi as u64 * img.stride + xi as u64 * bpp as u64;
        let mut comps = vec![0u32; 4];
        for (i, c) in args.dest.iter().take(4).enumerate() {
            comps[i] = if c.0 {
                g.st.get_reg16(c.1 as usize, t)? as u32
            } else {
                g.st.get_reg32(c.1 as usize, t)? as u32
            };
        }
        let raw = store_texel(img.channels, img.type_, img.srgb, &comps)?;
        if off as usize + raw.len() > img.stride as usize * img.height as usize {
            return Err("image_write past end of image".into());
        }
        write_bytes(mem, img.addr + off, &raw)?;
    }
    Ok(())
}


/// value operands (registers, uniforms, immediates) in order, without condition codes and other fields
fn vals(ops: &[Operand]) -> Vec<Operand> {
    ops.iter()
        .filter(|o| matches!(o, Operand::Reg16(..) | Operand::Reg32(..) | Operand::Reg64(..) | Operand::UReg16(..) | Operand::UReg32(..)
            | Operand::UReg64(..) | Operand::Imm(..)))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bindless texture_sample as the LLVM AGX backend emits it (llvm.agx.texture.sample): channels to
    // r7_r8_r9_r10, descriptor at u0_u1 + r4, register sampler r3l (heap index), coordinates r5 (1D) or r5_r6.
    fn sample_kernel(dim2: bool, x: f32, y: f32, offset: u32, sampler: u32) -> Vec<u8> {
        use crate::kernels::{e, get_sr, mov, store};
        let sample = [0x31, 0x1d, 0x0a, 0x00, 0xc8, if dim2 { 0x62 } else { 0x60 }, 0x0f, 0x46];
        [get_sr(80, 1), mov(4, offset), mov(3, sampler), mov(5, x.to_bits()), mov(6, y.to_bits()), sample.to_vec(),
         e("wait 0"), store(7, 8, 1), store(8, 10, 1), store(9, 12, 1), store(10, 14, 1), e("wait 0"), e("stop")]
            .concat()
    }

    fn put(b: &mut [u8], start: u32, size: u32, v: u64) {
        for i in 0..size {
            if (v >> i) & 1 != 0 {
                b[((start + i) / 8) as usize] |= 1 << ((start + i) % 8);
            }
        }
    }

    /// A linear texture descriptor (Mesa cmdbuf.xml Texture): dimension, channels, type, swizzle, size, address
    /// and row pitch (written as pitch - 16 >> 4).
    fn tex_desc(dim: u64, channels: u64, type_: u64, swizzle: [u64; 4], w: u64, h: u64, addr: u64, pitch: u64) -> Vec<u8> {
        let mut b = vec![0u8; 24];
        put(&mut b, 0, 4, dim);
        put(&mut b, 6, 7, channels);
        put(&mut b, 13, 3, type_);
        for (i, sw) in swizzle.iter().enumerate() {
            put(&mut b, 16 + 3 * i as u32, 3, *sw);
        }
        put(&mut b, 28, 14, w - 1);
        put(&mut b, 42, 14, h - 1);
        put(&mut b, 66, 36, addr >> 4);
        put(&mut b, 110, 18, (pitch - 16) >> 4);
        b
    }

    /// A sampler descriptor (Mesa cmdbuf.xml Sampler): filter (both), wrap (all three), pixel coordinates.
    fn sampler_desc(filter: u64, wrap: u64, pixel: bool) -> Vec<u8> {
        let mut b = vec![0u8; 8];
        put(&mut b, 23, 2, filter);
        put(&mut b, 25, 2, filter);
        for i in 0..3 {
            put(&mut b, 29 + 3 * i, 3, wrap);
        }
        put(&mut b, 38, 1, pixel as u64);
        b
    }

    const HEAP: u64 = 0x10000;
    const TEXELS: u64 = 0x20000;
    const SAMPLERS: u64 = 0x30000;
    const OUT: u64 = 0x40000;

    /// Runs one sample with the descriptor at HEAP + 48 and the samplers at SAMPLERS; returns lane 0's channels.
    fn sample(code: Vec<u8>, desc: Vec<u8>, texels: Vec<u8>, samplers: &[Vec<u8>]) -> Result<[u32; 4], String> {
        let mut uniforms = vec![(0, HEAP as u32), (1, 0)];
        for c in 0..4u64 {
            uniforms.extend([(8 + 2 * c as usize, (OUT + 128 * c) as u32), (9 + 2 * c as usize, 0)]);
        }
        let d = Dispatch { code, grid: [32, 1, 1], wg: [32, 1, 1], shared_bytes: 0, uniforms, num_registers: 16,
                           max_steps: 1000, placement: 0, launch: Default::default() };
        let mut mem = Memory::default();
        let mut heap = vec![0u8; 48];
        heap.extend(desc);
        mem.map(HEAP, heap);
        mem.map(TEXELS, texels);
        mem.map(SAMPLERS, samplers.concat());
        mem.map(OUT, vec![0; 512]);
        mem.sampler_heap = Some((SAMPLERS, samplers.len() as u32));
        run(&d, &mut mem)?;
        let w = mem.words(OUT, 128)?;
        Ok([w[0], w[32], w[64], w[96]])
    }

    fn floats(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn bindless_texture_sample_with_a_sampler_heap() {
        // 16 x 4 R32F, pitch 64 bytes, swizzle R001; sampler 1: nearest, clamp to edge, normalized
        let texels = floats(&(0..64).map(|i| i as f32).collect::<Vec<_>>());
        let desc = tex_desc(2, 0x21, 4, [0, 5, 5, 4], 16, 4, TEXELS, 64);
        let samplers = [sampler_desc(1, 0, false), sampler_desc(0, 0, false), sampler_desc(0, 3, true)];
        let f = |x: f32| x.to_bits();
        // (0.3, 0.6) -> texel (4, 2)
        // heap samplers are 28 + their index
        let r = sample(sample_kernel(true, 0.3, 0.6, 48, 29), desc.clone(), texels.clone(), &samplers).unwrap();
        assert_eq!(r, [f(36.0), 0, 0, f(1.0)]);
        // clamped to the edge
        let r = sample(sample_kernel(true, 1.7, -0.2, 48, 29), desc.clone(), texels.clone(), &samplers).unwrap();
        assert_eq!(r[0], f(15.0));
        // sampler 2: unnormalized coordinates, clamp to border (transparent black)
        let r = sample(sample_kernel(true, 5.5, 2.2, 48, 30), desc.clone(), texels.clone(), &samplers).unwrap();
        assert_eq!(r[0], f(37.0));
        let r = sample(sample_kernel(true, 16.5, 2.2, 48, 30), desc.clone(), texels.clone(), &samplers).unwrap();
        assert_eq!(r, [0; 4]);
        // an index past the heap is an error
        let e = sample(sample_kernel(true, 0.5, 0.5, 48, 31), desc.clone(), texels.clone(), &samplers).unwrap_err();
        assert!(e.contains("outside the sampler heap"), "{e}");
        // a heap index used as such (1) is sampler state slot 1, which nothing wrote
        let e = sample(sample_kernel(true, 0.5, 0.5, 48, 1), desc, texels, &samplers).unwrap_err();
        assert!(e.contains("state slot 1"), "{e}");
    }

    #[test]
    fn texture_sample_1d_linear_and_integer_formats() {
        // 1D R32F of 8 texels, linear: 2.75 / 8 is texel 2.25, weight 0.25
        let desc = tex_desc(0, 0x21, 4, [0, 5, 5, 4], 8, 1, TEXELS, 32);
        let texels = floats(&(0..8).map(|i| 10.0 * i as f32).collect::<Vec<_>>());
        let r = sample(sample_kernel(false, 2.75 / 8.0, 0.0, 48, 28), desc, texels, &[sampler_desc(1, 0, false)]).unwrap();
        assert_eq!(r, [22.5f32.to_bits(), 0, 0, 1.0f32.to_bits()]);
        // 2D R8G8 uint, swizzle RG01: integers, the integer 1 for alpha
        let desc = tex_desc(2, 0x0A, 2, [0, 1, 5, 4], 8, 2, TEXELS, 16);
        let texels: Vec<u8> = (0..32).collect();
        let r = sample(sample_kernel(true, 3.5 / 8.0, 0.75, 48, 28), desc, texels, &[sampler_desc(0, 0, false)]).unwrap();
        assert_eq!(r, [22, 23, 0, 1]);
    }

    fn m2_map_fixture(seed: u32) -> (Dispatch, Memory, Vec<u32>) {
        use crate::kernels::{e, get_sr, store};
        let n = 12 * 128u32;
        let code = [get_sr(80, 1), get_sr(20, 2), store(2, 8, 1), e("wait 0"),
            get_sr(17, 3), store(3, 10, 1), e("wait 0"), e("stop")].concat();
        let d = Dispatch { code, grid: [n, 1, 1], wg: [128, 1, 1], shared_bytes: 0,
            uniforms: vec![(8, 0x1000), (9, 0), (10, 0x1000 + n * 4), (11, 0)],
            num_registers: 8, max_steps: 100, placement: seed, launch: Default::default() };
        let mut mem = Memory::default();mem.map(0x1000, vec![0x5a; n as usize * 8]);
        (d, mem, vec![0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19])
    }

    #[test]
    fn m2_core_map_replays_all_lanes_and_keeps_default_and_synthetic_slots() {
        for seed in [0, 7] {
            let (d, mem, cores) = m2_map_fixture(seed);let n = d.grid[0] as usize;
            let mut original = mem.clone();let old = run(&d, &mut original).unwrap();
            let mut replay = mem;let new = run_with_m2_core_map(&d, &mut replay, &cores).unwrap();
            assert_eq!((new.instructions, new.groups), (old.instructions, old.groups));
            let original = original.words(0x1000, n * 2).unwrap();let replay = replay.words(0x1000, n * 2).unwrap();
            let synthetic_cores = [8, 15, 22, 29, 36, 43, 50, 7, 14, 21, 28, 35];
            for lane in 0..n {
                let wg = lane / 128;let group = lane % 128 / 32;
                assert_eq!(original[lane], if seed == 0 { 0 } else { synthetic_cores[wg] });
                assert_eq!(replay[lane], cores[wg]);
                let slot = if seed == 0 { group } else { group + [8, 12, 4][wg % 3] } as u32;
                assert_eq!(original[n + lane], slot);assert_eq!(replay[n + lane], slot);
            }
        }
    }

    #[test]
    fn m2_core_map_rejects_bad_input_before_any_memory_writes() {
        let (d, mem, cores) = m2_map_fixture(0);
        let mut wrong_core = cores.clone();wrong_core[0] = 15;
        let mut disabled = cores.clone();disabled[0] = 25;
        let mut mixed = cores.clone();mixed[1] = 16;
        for (map, reason) in [(vec![], "entries"), (wrong_core, "disabled"), (disabled, "disabled"), (mixed, "spans clusters")] {
            let mut m = mem.clone();let before = m.read_bytes(0x1000, 12 * 128 * 8).unwrap();
            assert!(run_with_m2_core_map(&d, &mut m, &map).unwrap_err().contains(reason));
            assert_eq!(m.read_bytes(0x1000, before.len()).unwrap(), before);
        }
        validate_m2_core_map(&d, &cores).unwrap(); // includes a valid four-workgroup tail
    }

    #[test]
    fn m2_core_map_does_not_impose_1d_blocks_on_other_geometry() {
        let (mut d, _, _) = m2_map_fixture(0);
        d.grid = [128, 12, 1];
        validate_m2_core_map(&d, &[0, 16, 32, 48, 0, 16, 32, 48, 0, 16, 32, 48]).unwrap();
    }

    #[test]
    fn m2_core_map_prepared_traces_resolve_core_and_loaded_addresses_per_snapshot() {
        use crate::kernels::{e, get_sr, mov, store};
        let (mut d, _, cores) = m2_map_fixture(0);let n = d.grid[0];
        d.code = [get_sr(80, 1), get_sr(20, 2), mov(5, 0),
            e("device_load 0, i32, x, r4, u10_u11, r5, unsigned"), e("wait 0"),
            e("imadd r3, r2, r4, r1"), store(2, 8, 3), e("wait 0"), e("stop")].concat();
        d.uniforms = vec![(8, 0x1000), (9, 0), (10, 0), (11, 16)];
        let prepared = PreparedDispatch::with_m2_core_map(&d, &cores).unwrap();
        let legacy = PreparedDispatch::new(&d).unwrap();
        let wgs: Vec<_> = [11, 0, 8, 7, 3, 11].into_iter().map(|x| [x, 0, 0]).collect();
        crate::par::set_local_threads(4);
        for stride in [n, n * 2] {
            let mut mem = Memory::default();
            mem.map(0x1000, vec![0x5a; 64 * stride as usize * 4]);
            mem.map(1u64 << 36, stride.to_le_bytes().to_vec());
            let before = mem.read_bytes(0x1000, 64 * stride as usize * 4).unwrap();
            for (batch, supplied) in [(prepared.trace_workgroups(&mem, &wgs).unwrap(), true),
                (legacy.trace_workgroups(&mem, &wgs).unwrap(), false)] {
                for (i, traces) in batch.iter().enumerate() {
                    assert_eq!(traces.len(), 4);
                    for (g, tr) in traces.iter().enumerate() {
                        let core = if supplied { cores[wgs[i][0] as usize] } else { 0 };
                        let first = core as u64 * stride as u64 + wgs[i][0] as u64 * 128 + g as u64 * 32;
                        let segment = (0x1000 + first * 4) >> 6;
                        assert_eq!(tr.mem, vec![(3, vec![(1u64 << 36) >> 6]), (6, vec![segment, segment + 1])]);
                        assert!(tr.skipped.is_empty() && tr.warnings.is_empty());
                    }
                }
            }
            assert_eq!(mem.read_bytes(0x1000, before.len()).unwrap(), before);
            assert_eq!(mem.read(1u64 << 36, 4).unwrap(), stride as u64);
            let fresh = trace_workgroups_with_m2_core_map(&d, &mem, &wgs, &cores).unwrap();
            let cached = prepared.trace_workgroups(&mem, &wgs).unwrap();
            for (a, b) in fresh.iter().flatten().zip(cached.iter().flatten()) {
                assert_eq!(a.pcs, b.pcs);assert_eq!(a.mem, b.mem);assert_eq!(a.masked, b.masked);
            }
        }
        crate::par::set_local_threads(0);
    }

    #[test]
    fn m2_core_map_prepared_validation_rejects_before_tracing() {
        let (d, mem, mut cores) = m2_map_fixture(0);cores[1] = 16;
        assert!(PreparedDispatch::with_m2_core_map(&d, &cores).err().unwrap().contains("spans clusters"));
        assert!(trace_workgroups_with_m2_core_map(&d, &mem, &[[0; 3]], &cores).unwrap_err().contains("spans clusters"));
        cores[1] = 1;let p = PreparedDispatch::with_m2_core_map(&d, &cores).unwrap();
        assert!(p.trace_workgroups(&mem, &[[0; 3], [12, 0, 0]]).unwrap_err().contains("outside dispatch"));
        assert_eq!(mem.read(0x1000, 4).unwrap(), 0x5a5a5a5a);
    }

    fn sr_group(mask: u32) -> Group {
        let mut st = CoreState::new(8, None);
        for t in 0..32 {
            st.exec[t] = mask & (1 << t) != 0;
            st.set_reg32(4, t, 0xaabbccdd).unwrap();
        }
        Group { st, ctx: GroupCtx { wg_size: [8, 4, 2], wg_id: [3, 2, 1], group: 1, core: 19, slot: 7, uniforms: vec![],
            launch: crate::cost::SrLaunch { tg_bytes: 513, regs: 65, context: Some(47), pipeline: Some(0x1080), preamble: false } },
            pc: 0, done: false, steps: 0, trace: None, stack: vec![], pending: vec![], lane_regs: vec![] }
    }

    #[test]
    fn cached_special_registers_match_geometry_and_independent_mask_oracles() {
        for sr in (0u32..256).filter(|&sr| sr != 113) { // hardware time-varying SR113 remains unmodelled
            for half in [false, true] {
                let dest = if half { Operand::Reg16(8, 0) } else { Operand::Reg32(4, 0) };
                let args = SpecialRegister::new(&vec![("SR", sr as u64)], &[dest]);
                for mask in [0u32, u32::MAX, 0x55555555, 0x80808080, 0x00002001, 0xffff0000, 0x00001fff, 0x80000000] {
                    let mut g = sr_group(mask);
                    let mut stats = RunStats::default();
                    special_register(&mut g, &args, &mut stats).unwrap();
                    assert_eq!(stats.placement_read, sr == 17 || sr == 20);
                    for t in 0..32 {
                        let mut expected = 0xaabbccdd;
                        if mask & (1 << t) != 0 {
                            let q = t / 4 * 4;
                            let value = match sr {
                                56 => (q..t).filter(|&i| mask & (1 << i) != 0).count() as u32,
                                57 => (q..q + 4).filter(|&i| mask & (1 << i) != 0).count() as u32,
                                58 => (0..t).filter(|&i| mask & (1 << i) != 0).count() as u32,
                                59 => mask.count_ones(),
                                _ => g.ctx.sr(sr, t).unwrap(),
                            };
                            expected = if half { 0xaabb0000 | (value & 0xffff) } else { value };
                        }
                        assert_eq!(g.st.get_reg32(4, t).unwrap() as u32, expected, "sr={sr} half={half} mask={mask:x} lane={t}");
                    }
                }
            }
        }
    }

    #[test]
    fn cached_sr_metadata_does_not_cache_launch_values_or_bypass_destination_errors() {
        let args = SpecialRegister::new(&vec![("SR", 21)], &[Operand::Reg32(4, 0)]);
        let mut g = sr_group(1);
        let mut stats = RunStats::default();
        for context in [47, 12] {
            g.ctx.launch.context = Some(context);
            special_register(&mut g, &args, &mut stats).unwrap();
            assert_eq!(g.st.get_reg32(4, 0).unwrap(), context as i128);
        }
        let unknown = SpecialRegister::new(&vec![("SR", 113)], &[Operand::Reg32(4, 0)]);
        let mut empty = sr_group(0);
        special_register(&mut empty, &unknown, &mut stats).unwrap(); // inactive lanes never resolve unknown/fallback values
        let invalid = SpecialRegister::new(&vec![("SR", 7)], &[]);
        assert!(special_register(&mut empty, &invalid, &mut stats).unwrap_err().contains("unexpected destination"));
    }

    fn dispatch(code: Vec<u8>, grid: [u32; 3], wg: [u32; 3]) -> Dispatch {
        Dispatch { code, grid, wg, shared_bytes: 0, uniforms: vec![], num_registers: 32,
            max_steps: 1000, placement: 0, launch: Default::default() }
    }

    fn local_atomic_case(op: u64, ret: bool, source: u32, compare: u32) -> (Dispatch, Memory) {
        use crate::kernels::{mov, store, tg, tg_atomic_add};
        let mut code = mov(2, 0);
        code.extend(mov(3, 10));
        code.extend(mov(4, 0xaabbccdd));
        code.extend(mov(5, 0));
        code.extend(mov(source, 3));
        if source < 31 { code.extend(mov(source + 1, compare)); }
        code.extend(tg(false, "i32", 3, Some(5), 0, 1));
        let atomic = tg_atomic_add(4, 5, source, ret);
        let n = isa::opcode_to_number(&atomic);
        let desc = isa::get_instruction_descriptor(n).unwrap();
        // Mesa's local-atomic op field is bits 24..27. Preserve the original
        // 10-byte length/control bits instead of allowing encode_fields to shorten it.
        let encoded = (n & !(15u128 << 24)) | (op as u128) << 24;
        assert_eq!(fget(&desc.decode_fields(encoded), "op"), Some(op));
        assert_eq!(desc.decode_size(encoded), atomic.len());
        code.extend((0..atomic.len()).map(|i| (encoded >> (8 * i)) as u8));
        code.extend(tg(true, "i32", 8, Some(5), 0, 1));
        code.extend(store(8, 8, 2));
        code.extend(mov(2, 1));
        code.extend(store(4, 8, 2));
        code.extend([0x88, 0]);
        let mut d = dispatch(code, [1; 3], [1; 3]);
        d.shared_bytes = 4;
        d.uniforms = vec![(8, 0x1000), (9, 0)];
        let mut mem = Memory::default();
        mem.map(0x1000, vec![0xcd; 8]);
        (d, mem)
    }

    #[test]
    fn unknown_threadgroup_atomic_is_rejected_even_with_an_empty_mask() {
        for op in 11..=15 {
            let args = ThreadgroupAtomic::new(&vec![("op", op)]);
            for mask in [0, 0xaaaaaaaa, u32::MAX] {
                let mut g = sr_group(mask);
                let mut memory = vec![0x5a; 2048];
                assert!(tg_atomic(&mut g, &args, &mut memory).unwrap_err()
                    .contains("termination observed, memory side effects not modeled"));
                assert_eq!(memory, vec![0x5a; 2048]);
            }
        }
    }

    #[test]
    fn threadgroup_non_cas_does_not_read_an_unused_compare_register() {
        // Synthetic register-boundary regression; ordinary op semantics are established
        // by the retained integer-atomic suite, not by a new hardware run here.
        for (op, expected) in [(0, 13), (1, 7), (2, 3), (4, 3), (5, 3), (6, 10), (7, 10), (8, 2), (9, 11), (10, 9)] {
            for ret in [false, true] {
                let (d, mut mem) = local_atomic_case(op, ret, 31, 0);
                run(&d, &mut mem).unwrap();
                assert_eq!(mem.words(0x1000, 2).unwrap(), vec![expected, if ret { 10 } else { 0xaabbccdd }], "op={op} ret={ret}");
            }
        }
    }

    #[test]
    fn threadgroup_cas_still_requires_and_uses_the_compare_register() {
        for (compare, expected) in [(10, 3), (11, 10)] {
            let (d, mut mem) = local_atomic_case(3, true, 30, compare);
            run(&d, &mut mem).unwrap();
            assert_eq!(mem.words(0x1000, 2).unwrap(), vec![expected, 10]);
        }
        let (d, mut mem) = local_atomic_case(3, true, 31, 0);
        assert!(run(&d, &mut mem).unwrap_err().contains("register"));
        assert_eq!(mem.words(0x1000, 2).unwrap(), vec![0xcdcdcdcd; 2]);
    }

    #[test]
    fn prepared_tracing_preserves_data_geometry_and_memory_isolation() {
        use crate::kernels::{atomic_add, e, e2, get_sr, mov};
        let mut code = mov(3, 0);
        code.extend(mov(5, 0));
        code.extend(mov(6, 1));
        code.extend(e("device_load 0, i32, x, r4, u8_u9, r5, unsigned"));
        code.extend(e("wait 0"));
        code.extend(get_sr(0, 7));
        code.extend(e("iadd r4, r4, r7"));
        code.extend(e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1"));
        let mut lp = e("iadd r3, 1, r3.sx");
        lp.extend(e("while_icmp r0l, ult, r3, r4, 1"));
        lp.extend(e(&format!("jmp_exec_any pc-{}", lp.len())));
        code.extend(lp);
        code.extend(e("pop_exec r0l, 1"));
        code.extend(atomic_add(8, 5, 6)); // mutate only the private workgroup snapshot
        code.extend([0x88, 0, 0xff]); // unreachable malformed byte is never decoded
        let mut d = dispatch(code, [512, 1, 1], [32, 1, 1]);
        d.uniforms = vec![(8, 0x1000), (9, 0)];
        let prepared = PreparedDispatch::new(&d).unwrap();
        let wgs: Vec<_> = (0..16).map(|x| [x, 0, 0]).collect();
        let mut lengths = vec![];
        crate::par::set_local_threads(4);
        for trips in [2u32, 5] {
            let mut mem = Memory::default();
            mem.map(0x1000, trips.to_le_bytes().to_vec());
            let batch = prepared.trace_workgroups(&mem, &wgs).unwrap();
            for (i, wg) in wgs.iter().enumerate() {
                let fresh = trace_workgroups(&d, &mem, &[*wg]).unwrap();
                let (a, b) = (&batch[i][0], &fresh[0][0]);
                assert_eq!(a.pcs, b.pcs);
                assert_eq!(a.mem, b.mem);
                assert_eq!(a.masked, b.masked);
                assert_eq!(a.skipped, b.skipped);
                assert_eq!(a.warnings, b.warnings);
                if i > 0 { assert!(a.pcs.len() > batch[i - 1][0].pcs.len()); }
            }
            assert_eq!(mem.read(0x1000, 4).unwrap(), trips as u64);
            lengths.push(batch[0][0].pcs.len());
        }
        crate::par::set_local_threads(0);
        assert!(lengths[1] > lengths[0]);
    }

    #[test]
    fn step_limit_survives_barrier_resumes_and_is_per_group() {
        // Two barrier variants then stop: each group executes exactly three instructions.
        let mut d = dispatch(vec![0x68, 0, 0x78, 0, 0x88, 0], [64, 1, 1], [64, 1, 1]);
        d.max_steps = 3;
        let stats = run(&d, &mut Memory::default()).unwrap();
        assert_eq!(stats.instructions, 6);
        d.max_steps = 2;
        assert!(run(&d, &mut Memory::default()).unwrap_err().contains("step limit"));
        assert!(trace_workgroups(&d, &Memory::default(), &[[0; 3]]).unwrap_err().contains("step limit"));
        d.max_steps = 0;
        assert!(run(&d, &mut Memory::default()).unwrap_err().contains("step limit"));
    }

    #[test]
    fn reject_truncated_instructions_and_invalid_taken_jumps() {
        use crate::kernels::{e, get_sr};
        for n in 1..4 {
            let d = dispatch(get_sr(80, 2)[..n].to_vec(), [32, 1, 1], [32, 1, 1]);
            assert!(run(&d, &mut Memory::default()).is_err(), "truncated get_sr: {n} bytes");
        }
        for text in ["jmp_exec_any pc-2", "jmp_exec_any pc+100", "jmp_exec_any pc+1"] {
            let d = dispatch(e(text), [32, 1, 1], [32, 1, 1]);
            assert!(run(&d, &mut Memory::default()).unwrap_err().contains("jump at"));
        }
        // An untaken jump does not inspect its target, nor decode unreachable data.
        let mut code = e("jmp_exec_none pc+100");
        code.extend([0x88, 0, 0xff]);
        let d = dispatch(code, [32, 1, 1], [32, 1, 1]);
        assert_eq!(run(&d, &mut Memory::default()).unwrap().instructions, 2);
    }

    #[test]
    fn partial_workgroups_mask_every_axis() {
        use crate::kernels::{e, get_sr, mov, store};
        for (grid, wg) in [([5u32, 3, 3], [4u32, 2, 2]), ([5, 7, 3], [4, 5, 2]), ([1, 1, 1], [8, 8, 8])] {
            let padded: [u32; 3] = std::array::from_fn(|i| grid[i].div_ceil(wg[i]) * wg[i]);
            let mut code = get_sr(80, 3);
            code.extend(get_sr(81, 4));
            code.extend(get_sr(82, 5));
            code.extend(mov(6, padded[0]));
            code.extend(mov(7, padded[0] * padded[1]));
            code.extend(e("imadd r3, r4, r6, r3"));
            code.extend(e("imadd r3, r5, r7, r3"));
            code.extend(mov(2, 1));
            code.extend(store(2, 8, 3));
            code.extend([0x88, 0]);
            let mut d = dispatch(code, grid, wg);
            d.uniforms = vec![(8, 0x1000), (9, 0)];
            let n = padded.iter().product::<u32>() as usize;
            let mut mem = Memory::default();
            mem.map(0x1000, vec![0; 4 * n]);
            run(&d, &mut mem).unwrap();
            let words = mem.words(0x1000, n).unwrap();
            for z in 0..padded[2] { for y in 0..padded[1] { for x in 0..padded[0] {
                let i = (x + padded[0] * (y + padded[1] * z)) as usize;
                assert_eq!(words[i], (x < grid[0] && y < grid[1] && z < grid[2]) as u32,
                    "grid={grid:?} wg={wg:?} at {x},{y},{z}");
            } } }
        }
    }

    #[test]
    fn invalid_dispatches_return_errors() {
        for wg in [[0, 1, 1], [1025, 1, 1], [u32::MAX, 2, 1]] {
            let d = dispatch(vec![], [1, 1, 1], wg);
            assert!(run(&d, &mut Memory::default()).is_err());
            assert!(trace_workgroups(&d, &Memory::default(), &[[0; 3]]).is_err());
        }
        let mut d = dispatch(vec![], [1; 3], [1; 3]);
        d.uniforms = vec![(256, 0)];
        assert!(run(&d, &mut Memory::default()).is_err());
        d.uniforms.clear();
        d.grid = [0, 1, 1];
        assert_eq!(run(&d, &mut Memory::default()).unwrap().instructions, 0);
    }

    /// An 8-byte device atomic (bytes from the LLVM AGX backend) with fields changed through its descriptor.
    fn atomic_with(bytes: [u8; 8], fields: &[(&'static str, i64)]) -> Vec<u8> {
        let n = crate::isa::opcode_to_number(&bytes);
        let d = crate::isa::get_instruction_descriptor(n).expect("atomic decodes");
        let mut f: std::collections::HashMap<&'static str, i64> =
            d.decode_fields(n).into_iter().map(|(k, v)| (k, v as i64)).collect();
        for &(k, v) in fields {
            assert!(f.contains_key(k), "no field {k}");
            f.insert(k, v);
        }
        let x = d.encode_fields(&f).expect("atomic encodes");
        (0..8).map(|i| (x >> (8 * i)) as u8).collect()
    }

    // B24 on the M2 Max: u64 min at an address 4 mod 8 is a silent no-op, and ops 11-15 end the SIMD group with no
    // memory effect whatever bits 45-46 hold.
    #[test]
    fn device_atomic_corners_follow_the_hardware() {
        use crate::kernels::{e, mov, store};
        // r2_r3 = u2_u3; atomic cmpxchg u64 min (mode 3), no result, at u0_u1
        let umin = [e("mov r2, u2"), e("mov r3, u3"),
            vec![0xd5, 0x00, 0x00, 0x8d, 0x00, 0x30, 0x04, 0x00], e("stop")].concat();
        for (offset, want) in [(0u64, [5u64, 100]), (4, [100, 100])] {
            let mut mem = Memory::default();
            mem.map(0x1000, [100u64, 100].iter().flat_map(|v| v.to_le_bytes()).collect());
            let va = 0x1000 + offset;
            let d = Dispatch { code: umin.clone(), grid: [1, 1, 1], wg: [1, 1, 1], shared_bytes: 0,
                uniforms: vec![(0, va as u32), (1, 0), (2, 5), (3, 0)], num_registers: 8, max_steps: 100,
                placement: 0, launch: Default::default() };
            run(&d, &mut mem).unwrap();
            let got = [mem.read(0x1000, 8).unwrap(), mem.read(0x1008, 8).unwrap()];
            assert_eq!(got, want, "u64 min at offset {offset}");
        }
        // atomic add r1 at u0_u1, as op 11 with each variant: neither it nor the store after it runs
        let add = [0x15, 0x00, 0x00, 0x8d, 0x00, 0x50, 0x02, 0x00];
        for variant in [0, 1, 2, 3] {
            let code = [mov(1, 7), mov(5, 0), mov(6, 77), atomic_with(add, &[("op", 11), ("u5", variant)]),
                store(6, 8, 5), e("wait 0"), e("stop")].concat();
            let mut mem = Memory::default();
            mem.map(0x1000, vec![0; 8]);
            mem.map(0x2000, vec![0; 4]);
            let d = Dispatch { code, grid: [1, 1, 1], wg: [1, 1, 1], shared_bytes: 0,
                uniforms: vec![(0, 0x1000), (1, 0), (8, 0x2000), (9, 0)], num_registers: 8, max_steps: 100,
                placement: 0, launch: Default::default() };
            run(&d, &mut mem).unwrap_or_else(|e| panic!("variant {variant}: {e}"));
            assert_eq!(mem.read(0x1000, 4).unwrap(), 0, "variant {variant}: no memory effect");
            assert_eq!(mem.read(0x2000, 4).unwrap(), 0, "variant {variant}: the group ended");
        }
    }

    #[test]
    fn atomic_float_edges() {
        for (a, b, want) in [
            (0x3f80_0000, 0x3f80_0000, 0x4000_0000),
            (0x3f80_0000, 0x3380_0000, 0x3f80_0000), // tie to even
            (0x3f80_0001, 0x3380_0000, 0x3f80_0002),
            (0x8000_0001, 0x8000_0000, 0x8000_0000), // signed input flush
            (0x0080_0001, 0x8080_0000, 0), // subnormal result flush
            (0x8080_0001, 0x0080_0000, 0x8000_0000),
            (0x7f80_0000, 0xff80_0000, 0x7fc0_0000),
            (0xff80_0001, 0, 0x7fc0_0000),
            (0x7f7f_ffff, 0x7f7f_ffff, 0x7f80_0000),
        ] { assert_eq!(atomic_fadd(a, b), want, "{a:08x} + {b:08x}"); }
    }

    #[test]
    fn bulk_memory_matches_byte_reads() {
        let mut m = Memory::default();
        m.map(100, vec![1, 2, 3, 4]);
        m.map(104, vec![5, 6, 7, 8]);
        assert_eq!(m.read_bytes(100, 8).unwrap(), (1..=8).collect::<Vec<u8>>());
        assert!(m.read_bytes(100, 9).is_err());
        assert_eq!(m.read_bytes(999, 0).unwrap(), Vec::<u8>::new());
        // Partial overlap in an earlier region wins, just as the old byte-wise dump did.
        m.map(98, vec![9; 12]);
        let expected: Vec<_> = (98..110).map(|a| m.read(a, 1).unwrap() as u8).collect();
        assert_eq!(m.read_bytes(98, 12).unwrap(), expected);
        let before = m.clone();
        m.write(100, 1, 42).unwrap();
        assert_eq!(before.read(100, 1).unwrap(), 1); // copy-on-write unchanged
        assert_eq!(m.read_bytes(100, 4).unwrap(), vec![42, 2, 3, 4]);
    }

    #[test]
    fn memory_bounds_do_not_wrap() {
        let mut m = Memory::default();
        m.map(u64::MAX - 3, vec![1, 2, 3, 4]);
        assert_eq!(m.read(u64::MAX - 3, 4).unwrap(), 0x0403_0201);
        assert_eq!(m.read_bytes(u64::MAX - 3, 4).unwrap(), vec![1, 2, 3, 4]);
        assert!(m.read_bytes(u64::MAX, 2).is_err());
        assert!(m.read(0, 1).is_err());
        assert!(m.read(u64::MAX - 3, 8).is_err());
        assert!(m.write(u64::MAX - 3, 9, 0).is_err());
    }
}

