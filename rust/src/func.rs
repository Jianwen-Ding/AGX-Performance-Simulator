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
        self.regions.iter().position(|(b, d)| va >= *b && va + len <= *b + d.len() as u64).map(|i| (i, (va - self.regions[i].0) as usize))
    }
    pub fn read(&self, va: u64, len: usize) -> Result<u64, String> {
        let (i, o) = self.find(va, len as u64).ok_or_else(|| format!("read of unmapped address {:#x}", va))?;
        let d = &self.regions[i].1;
        let mut v = 0u64;
        for k in 0..len {
            v |= (d[o + k] as u64) << (8 * k);
        }
        Ok(v)
    }
    pub fn write(&mut self, va: u64, len: usize, v: u64) -> Result<(), String> {
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
        (0..n).map(|k| self.read(va + 4 * k as u64, 4).map(|x| x as u32)).collect()
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
        "stop" | "trap" => Stop,
        "jmp_exec_any" => Jmp { any: true },
        "jmp_exec_none" => Jmp { any: false },
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

/// u8norm / 16-bit normalised formats are converted to fp16 by the hardware; not implemented yet.
fn plain_format(fmt: u64) -> bool {
    matches!(fmt, 0 | 1 | 2)
}

/// u8norm (format 4) load: v / 255 rounded to the destination register's precision (fp16 for 16-bit halves).
fn unorm8(v: u64, rt: bool) -> i128 {
    let x = v as f64 / 255.0;
    if rt { (x as f32).to_bits() as i128 } else { crate::isa::fma::f64_to_f16(x.to_bits(), false) as i128 }
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
            let ctx = GroupCtx { wg_size: d.wg, wg_id: [wx, wy, wz], group: g, core, slot, uniforms: d.uniforms.clone() };
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
                // SR58 / SR59 are the active thread's index among the active threads and the number of active threads
                // (Mesa aggregates atomics with them under divergence); with every lane active they equal the launch
                // geometry's lane index / lane count, which is what the recorded suite covers
                let active = g.st.exec.iter().filter(|&&e| e).count() as u32;
                let mut rank = 0u32;
                for t in 0..32 {
                    if g.st.exec[t] {
                        let v = match sr {
                            58 => rank,
                            59 => active,
                            _ => g.ctx.sr(sr, t).ok_or_else(|| format!("get_sr: special register {} not modelled", sr))?,
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
            OpKind::Atomic => device_atomic(g, f, mem)?,
            OpKind::FloatArith => fsem::float_arith(desc.name, f, v, &mut g.st)?,
            OpKind::CmpSel { float } => fsem::cmpsel(float, f, v, &mut g.st)?,
            OpKind::Ballot { float, quad } => fsem::ballot(float, quad, f, v, &mut g.st)?,
            OpKind::MaskCtl { kind, float } => fsem::mask_ctl(kind, float, f, v, &mut g.st)?,
            OpKind::TgAtomic => tg_atomic(g, f, tg)?,
            OpKind::Round => fsem::round_op(desc.name, v, &mut g.st)?,
            OpKind::Intl => fsem::intl(v, &mut g.st)?,
            OpKind::Transcendental => fsem::transcendental(desc.name, v, &mut g.st)?,
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
    let ctx = GroupCtx { wg_size: [1, 1, 1], wg_id: [0, 0, 0], group: 0, core: 0, slot: 0, uniforms: uniforms.to_vec() };
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
    if !plain_format(fmt) && !(fmt == 4 && load) {
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
                let v = if fmt == 4 { unorm8(v, rt) } else { v as i128 };
                if slot != 0 {
                    deferred.push((rt, r, k, t, v));
                } else {
                    put(&mut g.st, rt, r, k, t, v)?;
                }
            } else {
                let v = get(&g.st, rt, r, k, t)?;
                mem.store(va, item as usize, v as u64)?;
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
        let old = mem.read(va, 4)? as u32;
        let v = g.st.get_reg32(s / 2, t)? as u32;
        let cmp = g.st.get_reg32(s / 2 + 1, t)? as u32;
        mem.write(va, 4, atomic_op(f_op, old, v, cmp)? as u64)?;
        if ret {
            g.st.set_reg32(r / 2, t, old as i128)?;
        }
    }
    Ok(())
}

fn tg_atomic(g: &mut Group, f: &Fields, tg: &mut [u8]) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    // fields read once, not per lane (fget searches the field list by name)
    let (f_at, f_a, f_ot, f_o, f_op) = (x("At"), x("A"), x("Ot"), x("O"), x("op"));
    let r = x("R") as usize;
    let ret = x("Ra") != 0;
    let s = x("S") as usize;
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
