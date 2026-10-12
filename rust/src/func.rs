//! Functional executor: runs every SIMD group of a compute dispatch over real buffers.
//!
//! Instructions the applegpu-derived emulator (`isa::compiled`) implements run through it; this module adds
//! what a dispatch needs around it: device memory mapped at virtual addresses, the harness uniforms, special
//! registers per thread (`cost::GroupCtx`), device loads / stores of every format and mask, threadgroup memory per
//! workgroup, device and threadgroup atomics, barriers (whole workgroup, and 0x78 over aligned blocks of 4
//! groups), and the exec-mask jumps. Semantics follow the playground findings (vector loads index format
//! elements; 8/16-bit loads write the low half; threadgroup byte = base + 4*size*index + size*component; atomics
//! return the old value; cmpxchg's source holds the new value, source+1 the compare value).

use crate::cost::{is_barrier78, GroupCtx};
use crate::fsem;
use crate::isa::{self, compiled::Compiled, exec::CoreState, fget, Desc, Fields, Operand};

/// Sparse byte-addressed memory made of mapped regions.
#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub regions: Vec<(u64, Vec<u8>)>,
}

impl Memory {
    pub fn map(&mut self, va: u64, bytes: Vec<u8>) {
        self.regions.push((va, bytes));
    }
    fn find(&self, va: u64, len: u64) -> Option<(usize, usize)> {
        self.regions.iter().position(|(b, d)| va >= *b && va + len <= *b + d.len() as u64).map(|i| (i, (va - self.regions[i].0) as usize))
    }
    pub fn read(&self, va: u64, len: usize) -> Result<u64, String> {
        let (i, o) = self.find(va, len as u64).ok_or_else(|| format!("read of unmapped address {:#x}", va))?;
        let mut v = 0u64;
        for k in 0..len {
            v |= (self.regions[i].1[o + k] as u64) << (8 * k);
        }
        Ok(v)
    }
    pub fn write(&mut self, va: u64, len: usize, v: u64) -> Result<(), String> {
        let (i, o) = self.find(va, len as u64).ok_or_else(|| format!("write to unmapped address {:#x}", va))?;
        for k in 0..len {
            self.regions[i].1[o + k] = (v >> (8 * k)) as u8;
        }
        Ok(())
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
}

#[derive(Debug, Default)]
pub struct RunStats {
    pub instructions: u64,
    pub groups: u64,
}

struct Info {
    desc: Option<&'static Desc>,
    size: usize,
    fields: Fields,
    op: Option<Compiled>,
    ops: Vec<Operand>,
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
    let code = &d.code;
    let mut cache: Vec<Option<Info>> = (0..code.len()).map(|_| None).collect();
    let dims: Vec<u32> = (0..3).map(|i| d.grid[i].div_ceil(d.wg[i])).collect();
    let threads = d.wg[0] * d.wg[1] * d.wg[2];
    let gpw = threads.div_ceil(32);
    let mut stats = RunStats::default();
    for wz in 0..dims[2] {
        for wy in 0..dims[1] {
            for wx in 0..dims[0] {
                let mut tg = vec![0u8; d.shared_bytes.max(16) as usize];
                let mut groups: Vec<Group> = (0..gpw)
                    .map(|g| {
                        let ctx = GroupCtx { wg_size: d.wg, wg_id: [wx, wy, wz], group: g, core: 0, slot: g, uniforms: d.uniforms.clone() };
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
                        Group { st, ctx, pc: 0, done: false }
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
                        match step_group(g, code, &mut cache, mem, &mut tg, d.max_steps, &mut stats)? {
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
            }
        }
    }
    Ok(stats)
}

fn decode_at(code: &[u8], pc: usize, nregs: usize) -> Result<Info, String> {
    if is_barrier78(code, pc) {
        return Ok(Info { desc: None, size: 2, fields: vec![], op: None, ops: vec![] });
    }
    let n0 = isa::opcode_to_number(&code[pc..code.len().min(pc + 16)]);
    let desc = isa::get_instruction_descriptor(n0).ok_or_else(|| format!("undecodable instruction at {}", pc))?;
    let size = desc.decode_size(n0);
    let n = desc.mask_instr(n0);
    let fields = desc.decode_fields(n);
    let ops = if desc.name == "stop" || desc.name == "trap" { vec![] } else { desc.fields_to_operands(&fields)? };
    let op = Compiled::new(desc, &fields, &ops, nregs);
    Ok(Info { desc: Some(desc), size, fields, op: Some(op), ops })
}

fn step_group(g: &mut Group, code: &[u8], cache: &mut [Option<Info>], mem: &mut Memory, tg: &mut [u8], max_steps: u64,
              stats: &mut RunStats) -> Result<Stop, String> {
    let nregs = g.st.reg16s.len() / 2;
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
        let Some(desc) = info.desc else {
            g.pc += 2; // 0x78 barrier
            return Ok(Stop::Barrier(2));
        };
        let f = &info.fields;
        let x = |k: &str| fget(f, k).unwrap_or(0);
        match desc.name {
            "stop" | "trap" => return Ok(Stop::Done),
            "jmp_exec_any" | "jmp_exec_none" => {
                let any = g.st.exec.iter().any(|&e| e);
                let take = if desc.name == "jmp_exec_any" { any } else { !any };
                let mut off = x("off") as i64;
                if off & (1 << 31) != 0 {
                    off -= 1 << 32;
                }
                g.pc = if take { (g.pc as i64 + off) as usize } else { g.pc + info.size };
                continue;
            }
            "threadgroup_barrier" => {
                g.pc += info.size;
                return Ok(Stop::Barrier(1));
            }
            "wait" | "memory_barrier" => {}
            "get_sr" => {
                let sr = x("SR") as u32;
                let Some(Operand::Reg32(dr, _)) = info.ops.first() else { return Err("get_sr: unexpected destination".into()) };
                for t in 0..32 {
                    if g.st.exec[t] {
                        let v = g.ctx.sr(sr, t).ok_or_else(|| format!("get_sr: special register {} not modelled", sr))?;
                        g.st.set_reg32(*dr as usize, t, v as i128)?;
                    }
                }
            }
            "device_load" | "device_store" => device_mem(g, f, mem, desc.name == "device_load")?,
            "threadgroup_load" | "threadgroup_store" => tg_mem(g, f, tg, desc.name == "threadgroup_load")?,
            "atomic" => device_atomic(g, f, mem)?,
            "fadd32" | "fmul32" | "fmadd32" | "fadd16" | "fmul16" | "fmadd16" => fsem::float_arith(desc.name, f, &vals(&info.ops), &mut g.st)?,
            "icmpsel" | "fcmpsel" => fsem::cmpsel(desc.name == "fcmpsel", f, &vals(&info.ops), &mut g.st)?,
            "icmp_ballot" | "fcmp_ballot" | "icmp_quad_ballot" | "fcmp_quad_ballot" => {
                fsem::ballot(desc.name.starts_with('f'), desc.name.contains("quad"), f, &vals(&info.ops), &mut g.st)?
            }
            "if_icmp" | "if_fcmp" | "else_icmp" | "else_fcmp" | "while_icmp" | "while_fcmp" => {
                let kind = desc.name.split('_').next().unwrap();
                fsem::mask_ctl(kind, desc.name.ends_with("fcmp"), f, &vals(&info.ops), &mut g.st)?
            }
            "threadgroup_atomic" => tg_atomic(g, f, tg)?,
            "floor" | "ceil" | "trunc" | "rint" => fsem::round_op(desc.name, &vals(&info.ops), &mut g.st)?,
            "intl" => fsem::intl(&vals(&info.ops), &mut g.st)?,
            "convert" => fsem::convert(f, &vals(&info.ops), &mut g.st)?,
            n if n.starts_with("quad_shuffle") => fsem::quad_shuffle(n, &vals(&info.ops), &mut g.st)?,
            n if (n.starts_with("simd_") || n.starts_with("quad_")) && !n.starts_with("simd_shuffle") && !n.starts_with("simd_matrix") => {
                fsem::cross_lane(n, &vals(&info.ops), &mut g.st)?
            }
            _ => {
                info.op.as_ref().unwrap().exec(&mut g.st).map_err(|e| format!("{} ({}): {}", desc.name, isa::disassemble_normalized(desc, 0).unwrap_or_default(), e))?;
            }
        }
        g.pc += info.size;
    }
    Ok(Stop::Done)
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
    let fmt = x("F") | x("Fx") << 3;
    if !plain_format(fmt) && !(fmt == 4 && load) {
        return Err(format!("device memory format {} not implemented", fmt));
    }
    let item = item_size(fmt);
    let shift = if x("s") == 3 { 2 } else { x("s") };
    let mask = x("mask");
    let (r, rt) = (x("R") as usize, x("Rt") != 0);
    // per-lane element address
    let mut addr = [0i128; 32];
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let base = if x("At") != 0 { g.st.ureg64((x("A") >> 1) as usize)? } else { g.st.get_reg64((x("A") >> 1) as usize, t)? };
        let mut off: i128 = if x("Ot") != 0 {
            ext(x("O") as i128, 16, true)
        } else {
            let o = g.st.get_reg32((x("O") >> 1) as usize, t)?;
            if x("Ou") == 0 { ext(o, 32, true) } else { o }
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
        for t in 0..32 {
            if !g.st.exec[t] {
                continue;
            }
            let va = (addr[t] + c as i128 * item as i128) as u64;
            if load {
                let v = mem.read(va, item as usize)?;
                let v = if fmt == 4 { unorm8(v, rt) } else { v as i128 };
                put(&mut g.st, rt, r, k, t, v)?;
            } else {
                let v = get(&g.st, rt, r, k, t)?;
                mem.write(va, item as usize, v as u64)?;
            }
        }
        k += 1;
    }
    Ok(())
}

fn tg_mem(g: &mut Group, f: &Fields, tg: &mut [u8], load: bool) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    let item = item_size(x("F")) as usize;
    let mask = x("mask");
    let (r, rt) = (x("R") as usize, x("Rt") != 0);
    let mut addr = [0usize; 32];
    for t in 0..32 {
        if g.st.exec[t] {
            let base = if x("At") == 0 { g.st.get_reg16(x("A") as usize, t)? as usize } else { 0 };
            let idx = if x("Ot") != 0 { x("O") as usize } else { g.st.get_reg16(x("O") as usize, t)? as usize };
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
    let r = (x("R") | x("Rx") << 6) as usize;
    let ret = x("Rt") != 0;
    let s = x("S") as usize;
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let base = if x("At") != 0 { g.st.ureg64((x("A") >> 1) as usize)? } else { g.st.get_reg64((x("A") >> 1) as usize, t)? };
        let off: i128 = if x("Ot") != 0 {
            ext(x("O") as i128, 16, true)
        } else {
            let o = g.st.get_reg32((x("O") >> 1) as usize, t)?;
            if x("Ou") == 0 { ext(o, 32, true) } else { o }
        };
        let va = (base + 4 * off) as u64;
        let old = mem.read(va, 4)? as u32;
        let v = g.st.get_reg32(s / 2, t)? as u32;
        let cmp = g.st.get_reg32(s / 2 + 1, t)? as u32;
        mem.write(va, 4, atomic_op(x("op"), old, v, cmp)? as u64)?;
        if ret {
            g.st.set_reg32(r / 2, t, old as i128)?;
        }
    }
    Ok(())
}

fn tg_atomic(g: &mut Group, f: &Fields, tg: &mut [u8]) -> Result<(), String> {
    let x = |k: &str| fget(f, k).unwrap_or(0);
    let r = x("R") as usize;
    let ret = x("Ra") != 0;
    let s = x("S") as usize;
    for t in 0..32 {
        if !g.st.exec[t] {
            continue;
        }
        let base = if x("At") == 0 { g.st.get_reg16(x("A") as usize, t)? as usize } else { 0 };
        let idx = if x("Ot") != 0 { x("O") as usize } else { g.st.get_reg16(x("O") as usize, t)? as usize };
        let b = base + 16 * idx;
        if b + 4 > tg.len() {
            return Err(format!("threadgroup atomic outside the declared memory ({} bytes)", tg.len()));
        }
        let old = u32::from_le_bytes([tg[b], tg[b + 1], tg[b + 2], tg[b + 3]]);
        let v = g.st.get_reg32(s / 2, t)? as u32;
        let cmp = g.st.get_reg32(s / 2 + 1, t)? as u32;
        tg[b..b + 4].copy_from_slice(&atomic_op(x("op"), old, v, cmp)?.to_le_bytes());
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
