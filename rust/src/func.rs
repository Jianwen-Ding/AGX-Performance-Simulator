//! Functional executor: runs every SIMD group of a compute dispatch over real buffers.
//!
//! Instructions the applegpu-derived emulator (`isa::compiled`) implements run through it; this module adds
//! what a dispatch needs around it: device memory mapped at virtual addresses, the harness uniforms, special
//! registers per thread (`cost::GroupCtx`), device loads / stores of every format and mask, threadgroup memory per
//! workgroup, device and threadgroup atomics, barriers (whole workgroup, and 0x78 over aligned blocks of 4
//! groups), and the exec-mask jumps. Semantics follow the playground findings (vector loads index format
//! elements; 8/16-bit loads write the low half; threadgroup byte = base + 4*size*index + size*component; atomics
//! return the old value; cmpxchg's source holds the new value, source+1 the compare value).

use crate::cost::{self, is_barrier78, GroupCtx, Trace};
use crate::fsem;
use crate::isa::{self, compiled::Compiled, exec::CoreState, fget, Desc, Fields, Operand};

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
    ops: Vec<Operand>,
    /// the value operands (registers, uniforms, immediates) the semantic helpers take, computed once
    vals: Vec<Operand>,
    /// what step_group does with it, decided once from the name instead of on every execution
    kind: OpKind,
}

#[derive(Clone, Copy)]
enum OpKind {
    Stop,
    Jmp { any: bool },
    Call,
    Ret,
    TgBarrier,
    Nop,
    Wait,
    GetSr,
    DevMem { load: bool },
    Stack { load: bool },
    UniformStore,
    TgMem { load: bool },
    Atomic,
    FloatArith,
    CmpSel { float: bool },
    Ballot { float: bool, quad: bool },
    MaskCtl { kind: &'static str, float: bool },
    TgAtomic,
    Round,
    Intl,
    Transcendental,
    Convert,
    QuadShuffle,
    CrossLane,
    Generic,
}

fn op_kind(name: &'static str) -> OpKind {
    use OpKind::*;
    match name {
        "stop" | "trap" | "end_group" => Stop,
        "nop6" | "fixed18" => Nop,
        "jmp_exec_any" => Jmp { any: true },
        "jmp_exec_none" => Jmp { any: false },
        "call" => Call,
        "ret" => Ret,
        "threadgroup_barrier" => TgBarrier,
        "wait" => Wait,
        "memory_barrier" => Nop,
        "get_sr" => GetSr,
        // the .TODO encodings differ from the plain ones only in control bits (Mesa emits them for Vulkan-memory-model
        // atomic loads / stores, between memory_barrier sequences); on one sequentially simulated memory they read and
        // write exactly as the plain access does
        "device_load" | "device_store" | "device_load.TODO" | "device_store.TODO" => DevMem { load: name.starts_with("device_load") },
        "stack_load" | "stack_store" => Stack { load: name == "stack_load" },
        "uniform_store" => UniformStore,
        "threadgroup_load" | "threadgroup_store" => TgMem { load: name == "threadgroup_load" },
        "atomic" => Atomic,
        "fadd32" | "fmul32" | "fmadd32" | "fadd16" | "fmul16" | "fmadd16" => FloatArith,
        "icmpsel" | "fcmpsel" => CmpSel { float: name == "fcmpsel" },
        "icmp_ballot" | "fcmp_ballot" | "icmp_quad_ballot" | "fcmp_quad_ballot" => Ballot { float: name.starts_with('f'), quad: name.contains("quad") },
        "if_icmp" | "if_fcmp" | "else_icmp" | "else_fcmp" | "while_icmp" | "while_fcmp" => {
            MaskCtl { kind: name.split('_').next().unwrap(), float: name.ends_with("fcmp") }
        }
        "threadgroup_atomic" => TgAtomic,
        "floor" | "ceil" | "trunc" | "rint" => Round,
        "intl" => Intl,
        "rcp" | "rsqrt" | "rsqrt_special" | "log2" | "exp2" | "sin_pt_1" | "sin_pt_2" => Transcendental,
        "convert" => Convert,
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
    /// recorded dynamic path in the cost model's format (pcs, memory segments / threadgroup words per access)
    trace: Option<Trace>,
    /// per-lane private stack (scratch) memory for stack_store / stack_load, grown on demand
    stack: Vec<Vec<u8>>,
    /// device loads issued on scoreboard slot != 0 whose register writes land at the `wait` on that slot:
    /// (slot, [(32-bit register?, register, component, lane, value)])
    pending: Vec<(u64, Vec<(bool, usize, usize, usize, i128)>)>,
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
    let mut cache: Vec<Option<Info>> = (0..d.code.len()).map(|_| None).collect();
    let dims: Vec<u32> = (0..3).map(|i| d.grid[i].div_ceil(d.wg[i])).collect();
    let mut stats = RunStats::default();
    for wz in 0..dims[2] {
        for wy in 0..dims[1] {
            for wx in 0..dims[0] {
                run_workgroup(d, &mut cache, mem, [wx, wy, wz], false, &mut stats)?;
            }
        }
    }
    Ok(stats)
}

/// Dynamic path of every SIMD group of the given workgroups (each run alone from `mem` as it is, with its own copy),
/// for the cost model: exact control flow and addresses, whatever the data.
pub fn trace_workgroups(d: &Dispatch, mem: &Memory, wgs: &[[u32; 3]]) -> Result<Vec<Vec<Trace>>, String> {
    crate::par::map(wgs, |&wg| {
        let mut cache: Vec<Option<Info>> = (0..d.code.len()).map(|_| None).collect();
        let mut m = mem.clone();
        run_workgroup(d, &mut cache, &mut m, wg, true, &mut RunStats::default())
    })
    .into_iter()
    .collect()
}

fn wg_linear(d: &Dispatch, [wx, wy, wz]: [u32; 3]) -> u64 {
    let dims: Vec<u64> = (0..3).map(|i| d.grid[i].div_ceil(d.wg[i]) as u64).collect();
    wx as u64 + dims[0] * (wy as u64 + dims[1] * wz as u64)
}

/// (sr20 core id, sr17 hardware slot) of SIMD group `g` of a workgroup under `d.placement`. The hardware gives
/// every workgroup some core (SR20 = cluster << 4 | core) and its groups slots that equal the group index only
/// while the core holds nothing else (playground sr-geometry-core-id, sr17-sr32).
fn placement(d: &Dispatch, wg: [u32; 3], g: u32) -> (u32, u32) {
    if d.placement == 0 {
        return (0, g);
    }
    let w = wg_linear(d, wg);
    let gpw = (d.wg[0] * d.wg[1] * d.wg[2]).div_ceil(32);
    let several = (0..3).any(|i| d.grid[i] > d.wg[i]);
    let core = 1 + ((w * 7 + d.placement as u64) % 50) as u32;
    let slot = if several { g + gpw * (1 + ((w + d.placement as u64) % 3) as u32) } else { g };
    (core, slot)
}

fn run_workgroup(d: &Dispatch, cache: &mut [Option<Info>], mem: &mut Memory, [wx, wy, wz]: [u32; 3], record: bool,
                 stats: &mut RunStats) -> Result<Vec<Trace>, String> {
    let code = &d.code;
    let threads = d.wg[0] * d.wg[1] * d.wg[2];
    let gpw = threads.div_ceil(32);
    let mut tg = vec![0u8; d.shared_bytes.max(16) as usize];
    let mut groups: Vec<Group> = (0..gpw)
        .map(|g| {
            let (core, slot) = placement(d, [wx, wy, wz], g);
            let ctx = GroupCtx { wg_size: d.wg, wg_id: [wx, wy, wz], group: g, core, slot, uniforms: d.uniforms.clone(), launch: d.launch };
            let mut st = CoreState::new(d.num_registers, None);
            let lanes = ctx.lanes();
            for t in 0..32 {
                // lanes beyond the grid (partial workgroups at the edge) do not exist
                let gx = wx * d.wg[0] + ((32 * g + t as u32) % d.wg[0]);
                let exists = t < lanes && gx < d.grid[0];
                st.exec[t] = exists;
                let _ = st.set_reg16(0, t, if exists { 0 } else { 1 });
            }
            for &(u, v) in &d.uniforms {
                st.uniforms[2 * u] = (v & 0xffff) as u16;
                st.uniforms[2 * u + 1] = (v >> 16) as u16;
            }
            Group { st, ctx, pc: 0, done: false, trace: if record { Some(Trace::default()) } else { None }, stack: vec![], pending: vec![] }
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
            mem.writer = wg_linear(d, [wx, wy, wz]) << 16 | gi as u64;
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

/// Link-register base for call / ret: the value `call` stores is `call_base() + offset of the next instruction`.
/// 0x2000 on G14C (hardware probes C1a-C1c, playground novel-results/2026-10-17-call-ret).
fn call_base() -> i128 {
    std::env::var("AGX_CALL_BASE").ok().and_then(|v| i128::from_str_radix(v.trim_start_matches("0x"), if v.starts_with("0x") { 16 } else { 10 }).ok()).unwrap_or(0x2000)
}

fn decode_at(code: &[u8], pc: usize, nregs: usize) -> Result<Info, String> {
    if is_barrier78(code, pc) {
        return Ok(Info { desc: None, size: 2, fields: vec![], op: None, ops: vec![], vals: vec![], kind: OpKind::Nop });
    }
    let n0 = isa::opcode_to_number(&code[pc..code.len().min(pc + 16)]);
    let desc = isa::get_instruction_descriptor(n0).ok_or_else(|| format!("undecodable instruction at {}", pc))?;
    let size = desc.decode_size(n0);
    let n = desc.mask_instr(n0);
    let fields = desc.decode_fields(n);
    let ops = if desc.name == "stop" || desc.name == "trap" { vec![] } else { desc.fields_to_operands(&fields)? };
    let op = Compiled::new(desc, &fields, &ops, nregs);
    let vals = vals(&ops);
    Ok(Info { desc: Some(desc), size, fields, op: Some(op), ops, vals, kind: op_kind(desc.name) })
}

fn step_group(g: &mut Group, code: &[u8], cache: &mut [Option<Info>], mem: &mut Memory, tg: &mut [u8], max_steps: u64,
              stats: &mut RunStats) -> Result<Stop, String> {
    let nregs = g.st.reg16s.len(); // 16-bit registers, as Compiled::new expects
    let mut steps = 0u64;
    while g.pc < code.len() {
        steps += 1;
        stats.instructions += 1;
        if steps > max_steps {
            return Err("step limit (non-terminating loop?)".into());
        }
        if cache[g.pc].is_none() {
            cache[g.pc] = Some(decode_at(code, g.pc, nregs)?);
        }
        let info = cache[g.pc].as_ref().unwrap();
        if let Some(tr) = g.trace.as_mut() {
            // as cost::trace_group records it: every pc (barriers, jumps and the stop included) and, before the
            // access executes, the device segments / threadgroup words / atomic words of the active lanes
            tr.pcs.push(g.pc);
            let at = (tr.pcs.len() - 1) as u32;
            let touched = match info.desc.map(|d| d.name) {
                Some("device_load" | "device_store" | "device_load.TODO" | "device_store.TODO") => cost::mem_segments(&info.fields, &g.st),
                Some(n) if n.starts_with("threadgroup_") && n != "threadgroup_barrier" => cost::tg_words(&info.fields, &g.st),
                Some("atomic") => cost::atomic_words(&info.fields, &g.st),
                _ => None,
            };
            if let Some(w) = touched {
                tr.mem.push((at, w));
            }
        }
        let Some(desc) = info.desc else {
            g.pc += 2; // 0x78 barrier
            return Ok(Stop::Barrier(2));
        };
        let f = &info.fields;
        let x = |k: &str| fget(f, k).unwrap_or(0);
        let v = &info.vals;
        match info.kind {
            OpKind::Stop => return Ok(Stop::Done),
            OpKind::Jmp { any: want_any } => {
                let any = g.st.exec.iter().any(|&e| e);
                let take = if want_any { any } else { !any };
                let mut off = x("off") as i64;
                if off & (1 << 31) != 0 {
                    off -= 1 << 32;
                }
                g.pc = if take { (g.pc as i64 + off) as usize } else { g.pc + info.size };
                continue;
            }
            // call / ret (applegpu pseudocode; no hardware evidence yet, see the playground's remaining-work.md). The link
            // register is r1; the stored value is AGX_CALL_BASE + the byte offset of the next instruction. The base is 0x2000, measured
            // on G14C (stage 1 probes). A target outside the program is an error, never a jump.
            OpKind::Call => {
                if info.size != 6 {
                    return Err("call (register form): semantics unknown, not modelled".into());
                }
                let mut off = x("off") as i64;
                if off & (1 << 31) != 0 {
                    off -= 1 << 32;
                }
                let target = g.pc as i64 + off;
                if target < 0 || target as usize >= code.len() {
                    return Err(format!("call at {} targets {} outside the program (0..{})", g.pc, target, code.len()));
                }
                let link = call_base() + (g.pc + info.size) as i128;
                for t in 0..32 {
                    if g.st.exec[t] {
                        g.st.set_reg32(1, t, link)?;
                    }
                }
                g.pc = target as usize;
                continue;
            }
            OpKind::Ret => {
                let r = x("reg32") as usize;
                let mut tgt: Option<i128> = None;
                for t in 0..32 {
                    if g.st.exec[t] {
                        let v = g.st.get_reg32(r, t)? & 0xffff_ffff;
                        match tgt {
                            None => tgt = Some(v),
                            Some(p) if p != v => return Err(format!("ret at {}: active lanes disagree on the target (divergent ret not modelled)", g.pc)),
                            _ => {}
                        }
                    }
                }
                let Some(v) = tgt else {
                    g.pc += info.size; // no active lane: falls through
                    continue;
                };
                // ret jumps to code start + register (hardware fault of probe C2a: `ret r1` with the raw link, which
                // is code_off + pc + 6, fetched past the end of the code buffer). The link is therefore NOT a ret target.
                let target = v;
                if target < 0 || target as usize >= code.len() {
                    return Err(format!("ret at {} targets {} outside the program (0..{})", g.pc, target, code.len()));
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
            OpKind::GetSr => {
                let sr = x("SR") as u32;
                stats.placement_read |= sr == 17 || sr == 20;
                // 32-bit or 16-bit destination (Mesa writes e.g. sr58 to a half register); a half takes the low 16 bits
                let (dr, half) = match info.ops.first() {
                    Some(Operand::Reg32(r, _)) => (*r as usize, false),
                    Some(Operand::Reg16(r, _)) => (*r as usize, true),
                    _ => return Err("get_sr: unexpected destination".into()),
                };
                // SR58 / SR59 (SR56 / SR57 within the quad) are the active thread's index among the active threads and the number of active threads
                // (Mesa aggregates atomics with them under divergence); with every lane active they equal the launch
                // geometry's lane index / lane count, which is what the recorded suite covers
                // SR56 / SR57 are the same within the lane's quad (hardware: nested ifs, playground sr-divergence)
                let active = g.st.exec.iter().filter(|&&e| e).count() as u32;
                let mut rank = 0u32;
                for t in 0..32 {
                    if g.st.exec[t] {
                        let q = t & !3;
                        let v = match sr {
                            56 => (q..t).filter(|&u| g.st.exec[u]).count() as u32,
                            57 => (q..q + 4).filter(|&u| g.st.exec[u]).count() as u32,
                            58 => rank,
                            59 => active,
                            // AGX_SR_DEFAULT=V: unmodelled selectors read V (for checking a program's control flow / stores
                            // offline, e.g. a 256-selector scan, before it runs on the GPU)
                            _ => match (g.ctx.sr(sr, t), std::env::var("AGX_SR_DEFAULT").ok().and_then(|v| v.parse::<u32>().ok())) {
                                (Some(v), _) | (None, Some(v)) => v,
                                (None, None) => return Err(format!("get_sr: special register {} not modelled", sr)),
                            },
                        };
                        rank += 1;
                        if half { g.st.set_reg16(dr, t, v as i128)? } else { g.st.set_reg32(dr, t, v as i128)? }
                    }
                }
            }
            OpKind::DevMem { load } => device_mem(g, f, mem, load)?,
            OpKind::Stack { load } => stack_mem(g, f, load)?,
            OpKind::UniformStore => uniform_store(g, f)?,
            OpKind::TgMem { load } => tg_mem(g, f, tg, load)?,
            // Measured only with Mesa's u5=2: ops 11..15 end the whole issuing group,
            // including inactive lanes, without touching memory.
            OpKind::Atomic if x("op") >= 11 && x("u5") == 2 => return Ok(Stop::Done),
            OpKind::Atomic => device_atomic(g, f, mem)?,
            OpKind::FloatArith => fsem::float_arith(desc.name, f, v, &mut g.st)?,
            OpKind::CmpSel { float } => fsem::cmpsel(float, f, v, &mut g.st)?,
            OpKind::Ballot { float, quad } => fsem::ballot(float, quad, f, v, &mut g.st)?,
            OpKind::MaskCtl { kind, float } => fsem::mask_ctl(kind, float, f, v, &mut g.st)?,
            OpKind::TgAtomic => tg_atomic(g, f, tg)?,
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
        st.exec[t] = t == 0;
        let _ = st.set_reg16(0, t, if t == 0 { 0 } else { 1 });
    }
    for &(u, v) in uniforms {
        st.uniforms[2 * u] = (v & 0xffff) as u16;
        st.uniforms[2 * u + 1] = (v >> 16) as u16;
    }
    let mut g = Group { st, ctx, pc: 0, done: false, trace: None, stack: vec![], pending: vec![] };
    let mut cache: Vec<Option<Info>> = (0..code.len()).map(|_| None).collect();
    let mut tg = vec![0u8; 16];
    let mut stats = RunStats::default();
    match step_group(&mut g, code, &mut cache, mem, &mut tg, 1_000_000, &mut stats)? {
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

fn device_mem(g: &mut Group, f: &Fields, mem: &mut Memory, load: bool) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    // fields read once, not per lane (fget searches the field list by name)
    let (f_at, f_a, f_ot, f_o, f_ou) = (x("At"), x("A"), x("Ot"), x("O"), x("Ou"));
    let fmt = x("F") | x("Fx") << 3;
    if fmt > 7 {
        return Err(format!("device memory format {} not implemented", fmt));
    }
    let item = item_size(fmt);
    let shift = if x("s") == 3 { 2 } else { x("s") };
    let mask = x("mask");
    let (r, rt) = (x("R") as usize, x("Rt") != 0);
    // scoreboard slot (applegpu `g`, Mesa scoreboard bit 0): a load on slot 1 writes its registers only at `wait 1`;
    // read before that, the destination keeps its old value (coverage mem_load device_load_1_*: `wait 0` leaves
    // the preloaded 0xdeadbeef). Slot-0 loads complete at once (every program waits on slot 0 before reading).
    let slot = x("g");
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
                if slot != 0 {
                    deferred.push((rt, r, k, t, v));
                } else {
                    put(&mut g.st, rt, r, k, t, v)?;
                }
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

fn tg_mem(g: &mut Group, f: &Fields, tg: &mut [u8], load: bool) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    // fields read once, not per lane (fget searches the field list by name)
    let (f_at, f_a, f_ot, f_o) = (x("At"), x("A"), x("Ot"), x("O"));
    let item = item_size(x("F")) as usize;
    let mask = x("mask");
    let (r, rt) = (x("R") as usize, x("Rt") != 0);
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

fn device_atomic(g: &mut Group, f: &Fields, mem: &mut Memory) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    // fields read once, not per lane (fget searches the field list by name)
    let (f_at, f_a, f_ot, f_o, f_ou, f_op) = (x("At"), x("A"), x("Ot"), x("O"), x("Ou"), x("op"));
    let r = (x("R") | x("Rx") << 6) as usize;
    let ret = x("Rt") != 0;
    let s = x("S") as usize;
    let variant = x("u5");
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
            // Rt returns a changed flag, rather than the old memory value.
            let old = mem.read(va, 8)?;
            let v = g.st.get_reg64(s / 2, t)? as u64;
            let new = if variant & 2 != 0 { old.max(v) } else { old.min(v) };
            mem.write(va, 8, new)?;
            (new != old) as u32
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

fn tg_atomic(g: &mut Group, f: &Fields, tg: &mut [u8]) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    // fields read once, not per lane (fget searches the field list by name)
    let (f_at, f_a, f_ot, f_o, f_op) = (x("At"), x("A"), x("Ot"), x("O"), x("op"));
    let r = x("R") as usize;
    let ret = x("Ra") != 0;
    let s = x("S") as usize;
    // Recorded sub + gap bit 6 suppresses the access and returns zero.
    if f_op == 1 && x("u1") == 1 {
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
        let cmp = g.st.get_reg32(s / 2 + 1, t)? as u32;
        tg[b..b + 4].copy_from_slice(&atomic_op(f_op, old, v, cmp)?.to_le_bytes());
        if ret {
            g.st.set_reg32(r / 2, t, old as i128)?;
        }
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

