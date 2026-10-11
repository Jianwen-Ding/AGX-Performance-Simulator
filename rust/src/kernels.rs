//! Kernel generators shared by the model and the verifier: ports of `cost_hw.counted/kernel/mem_kernel`,
//! `run_bench.loop`, `bench_ops.OPS`, `agx_loop.store`, `run_drm_tg_layout.tg`,
//! `run_agx_timestamps.mov_imm` and `run_drm_opcode_describe.get_sr`.

use crate::isa::{self, desc_by_name, opcode_to_number, Operand};
use crate::pyrandom::PyRandom;
use std::cell::RefCell;
use std::collections::HashMap;

pub const DEFAULT_CONSTS: [u32; 4] = [1, 0x3f800000, 5, 7];
pub const MEM_CONSTS: [u32; 4] = [1, 0x3f800000, 32, 7];

thread_local! {
    static ENC: RefCell<HashMap<(String, Option<String>), Vec<u8>>> = RefCell::new(HashMap::new());
}

/// agx_loop.encode (memoised): assemble one line and require the round trip.
pub fn e(text: &str) -> Vec<u8> {
    enc(text, None)
}

pub fn e2(text: &str, want: &str) -> Vec<u8> {
    enc(text, Some(want))
}

fn enc(text: &str, want: Option<&str>) -> Vec<u8> {
    let key = (text.to_string(), want.map(str::to_string));
    if let Some(b) = ENC.with(|c| c.borrow().get(&key).cloned()) {
        return b;
    }
    let b = isa::encode(text, want).unwrap_or_else(|err| panic!("{}", err));
    ENC.with(|c| c.borrow_mut().insert(key, b.clone()));
    b
}

/// run_agx_timestamps.mov_imm: `mov_imm rN, imm32`, the control's own `6209` form.
pub fn mov(reg: u32, value: u32) -> Vec<u8> {
    let mut b = ((0x62u32 | 2 << 7 | (2 * reg) << 9) as u16).to_le_bytes().to_vec();
    b.extend_from_slice(&value.to_le_bytes());
    b
}

/// run_drm_opcode_describe.get_sr
pub fn get_sr(selector: u32, reg: u32) -> Vec<u8> {
    (0x72u32 | 2 << 7 | (2 * reg) << 9 | (selector & 63) << 16 | (selector >> 6) << 26).to_le_bytes().to_vec()
}

/// cost_hw.counted: counted loop around `body`; r24..r27 preloaded with `consts`, r5..r12 with 1.5f.
pub fn counted(body: &[u8], trips: u32, init: bool, consts: [u32; 4]) -> Vec<u8> {
    let mut head = mov(3, 0);
    head.extend(mov(4, trips));
    if init {
        for (k, v) in consts.iter().enumerate() {
            head.extend(mov(24 + k as u32, *v));
        }
        for r in 5..13 {
            head.extend(mov(r, 0x3fc00000));
        }
    }
    head.extend(e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1"));
    let mut lp = body.to_vec();
    lp.extend(e("iadd r3, 1, r3.sx"));
    lp.extend(e("while_icmp r0l, ult, r3, r4, 1"));
    let j = format!("jmp_exec_any pc-{}", lp.len());
    lp.extend(e2(&j, &j));
    head.extend(lp);
    head.extend(e("pop_exec r0l, 1"));
    head.extend(e("iadd r2, r5, 0"));
    head
}

/// agx_loop.store: `device_store 0, i32, x, r<reg>, u<base>_u<base+1>, r<index>, unsigned`, made by
/// editing the R/A/O fields of the control's own store and requiring an exact decode round trip.
pub fn store(reg: u32, base_uniform: u32, index_reg: u32) -> Vec<u8> {
    let rb = [0x45u8, 0x11, 0x2c, 0x0e, 0x00, 0xc0, 0x12, 0x00];
    let n = opcode_to_number(&rb);
    let d = isa::get_instruction_descriptor(n).expect("store control decodes");
    let mut f: HashMap<&'static str, i64> = d.decode_fields(n).into_iter().map(|(k, v)| (k, v as i64)).collect();
    f.insert("R", 2 * reg as i64);
    f.insert("A", 2 * base_uniform as i64);
    f.insert("O", 2 * index_reg as i64);
    let x = d.encode_fields(&f).expect("store encodes");
    let m = isa::get_instruction_descriptor(x).expect("store decodes");
    let back: HashMap<&'static str, i64> = m.decode_fields(x).into_iter().map(|(k, v)| (k, v as i64)).collect();
    assert!(std::ptr::eq(m, d) && back == f && m.decode_size(x) == 8, "store encoding mismatch");
    (0..8).map(|i| (x >> (8 * i)) as u8).collect()
}

/// run_drm_tg_layout.tg: `threadgroup_{load,store} <fmt>, x, value, base, index` (raw encoding, checked
/// against its disassembly).
pub fn tg(load: bool, fmt: &str, reg: u32, base: Option<u32>, index: i64, mask: u32) -> Vec<u8> {
    let f: u128 = match fmt {
        "i8" => 0,
        "i16" => 1,
        "i32" => 2,
        _ => panic!("tg format"),
    };
    let wide = fmt == "i32";
    let r = 2 * reg as u128;
    let (a, at): (u128, u128) = match base {
        None => (0, 0b10),
        Some(b) => (2 * b as u128, 0),
    };
    let (o, ot) = ((index & 0xffff) as u128, 1u128);
    let raw = (if load { 0x69 } else { 0x29 }) | (if wide { 1 << 8 } else { 0 }) | (r & 63) << 9 | 1 << 15 | (a & 63) << 16
        | at << 22 | f << 24 | (o & 63) << 28 | ot << 34 | (mask as u128) << 36 | (o >> 6) << 48 | (a >> 6) << 58
        | (r >> 6) << 60;
    let b: Vec<u8> = (0..8).map(|i| (raw >> (8 * i)) as u8).collect();
    let (d, n, size) = isa::decode_one(&b).expect("tg decodes");
    let text = isa::disassemble_normalized(d, n).unwrap();
    let comps: String = "xyzw".chars().enumerate().filter(|(k, _)| mask >> k & 1 != 0).map(|(_, c)| c).collect();
    let nreg = mask.count_ones();
    let regs = if wide { (0..nreg).map(|k| format!("r{}", reg + k)).collect::<Vec<_>>().join("_") } else { format!("r{}l", reg) };
    let want = format!(
        "threadgroup_{} {}, {}, {}, {}, {}",
        if load { "load" } else { "store" },
        fmt,
        comps,
        regs,
        match base {
            None => "0".to_string(),
            Some(b) => format!("r{}l", b),
        },
        index
    );
    assert!(size == 8 && text == want, "tg encoding mismatch: {:?} != {:?}", text, want);
    b
}

/// bench_ops.OPS (insertion order matters: random kernels draw from it).
pub const OPS: &[(&str, &str, &str)] = &[
    ("iadd", "iadd r{d}, r{d}, r20", "int"),
    ("isub", "isub r{d}, r{d}, r20", "int"),
    ("iadd_lsl2", "iadd r{d}, r{d}, r20, lsl 2", "int"),
    ("imadd", "imadd r{d}, r{d}, r21, r20", "int"),
    ("imul", "imadd r{d}, r{d}, r21, 0", "int"),
    ("iadd64", "iadd r{d}_r{d1}, r{d}_r{d1}, r20_r21", "int64"),
    ("imadd_wide", "imadd r{d}_r{d1}, r{d}, r21, r20_r21", "int64"),
    ("and", "and r{d}, r{d}, r20", "int"),
    ("xor", "xor r{d}, r{d}, r20", "int"),
    ("bfi", "bfi r{d}, r{d}, r20, r22", "bitfield"),
    ("bfeil", "bfeil r{d}, r{d}, r20, r22", "bitfield"),
    ("extr", "extr r{d}, r{d}, r20, r22", "bitfield"),
    ("shlhi", "shlhi r{d}, r{d}, r20, r22", "bitfield"),
    ("shrhi", "shrhi r{d}, r{d}, r20, r22", "bitfield"),
    ("asr", "asr r{d}, r{d}, r22", "bitfield"),
    ("asrh", "asrh r{d}, r{d}, r22", "bitfield"),
    ("bitrev", "bitrev r{d}, r{d}", "bitfield"),
    ("popcount", "popcount r{d}, r{d}", "bitfield"),
    ("ffs", "ffs r{d}, r{d}", "bitfield"),
    ("intl", "intl r{d}, r{d}, r20", "bitfield"),
    ("icmpsel", "icmpsel ult, r{d}, r{d}, r20, r{d}, r21", "select"),
    ("fcmpsel", "fcmpsel lt, r{d}, r{d}, r20, r{d}, r21", "select"),
    ("mov_imm", "mov_imm r{d}, 305419896, 0b0", "move"),
    ("iadd_imm", "iadd r{d}, r{d}, 1", "int"),
    ("iadd_cache", "iadd r{d}, r{d}, r20.cache", "int"),
    ("imadd_cache", "imadd r{d}, r{d}, r21.cache, r20.cache", "int"),
    ("fadd32", "fadd32 r{d}, r{d}, r20", "float"),
    ("fmul32", "fmul32 r{d}, r{d}, r21", "float"),
    ("fmadd32", "fmadd32 r{d}, r{d}, r21, r20", "float"),
    ("fadd32_imm", "fadd32 r{d}, r{d}, 0.125", "float"),
    ("fadd32_cache", "fadd32 r{d}, r{d}, r20.cache", "float"),
    ("fadd32_cache2", "fadd32 r{d}, r{d}.cache, r20.cache", "float"),
    ("fmul32_cache", "fmul32 r{d}, r{d}, r21.cache", "float"),
    ("fmadd32_imm", "fmadd32 r{d}, r{d}, 0.5, 0.125", "float"),
    ("fmadd32_cache", "fmadd32 r{d}, r{d}, r21.cache, r20.cache", "float"),
    ("fadd16", "fadd16 r{d}l, r{d}l, r20l", "half"),
    ("fmul16", "fmul16 r{d}l, r{d}l, r21l", "half"),
    ("fmadd16", "fmadd16 r{d}l, r{d}l, r21l, r20l", "half"),
    ("convert_f2i", "convert f_to_u32, r{d}, r{d}, rtz", "convert"),
    ("convert_i2f", "convert u32_to_f, r{d}, r{d}, rte", "convert"),
    ("floor", "floor r{d}, r{d}", "sfu"),
    ("rint", "rint r{d}, r{d}", "sfu"),
    ("rcp", "rcp r{d}, r{d}", "sfu"),
    ("rsqrt", "rsqrt r{d}, r{d}", "sfu"),
    ("rsqrt_special", "rsqrt_special r{d}, r{d}", "sfu"),
    ("log2", "log2 r{d}, r{d}", "sfu"),
    ("exp2", "exp2 r{d}, r{d}", "sfu"),
    ("sin_pt_1", "sin_pt_1 r{d}, r{d}", "sfu"),
    ("sin_pt_2", "sin_pt_2 r{d}, r{d}", "sfu"),
    ("simd_shuffle", "simd_shuffle r{d}, r{d}, r22l", "simd"),
    ("simd_shuffle_down", "simd_shuffle_down r{d}, r{d}, r22l", "simd"),
    ("simd_shuffle_xor", "simd_shuffle_xor r{d}, r{d}, r22l", "simd"),
    ("quad_shuffle", "quad_shuffle r{d}, r{d}, r22l", "simd"),
    ("simd_iadd", "simd_iadd r{d}, r{d}", "simd"),
    ("simd_prefix_iadd", "simd_prefix_iadd r{d}, r{d}", "simd"),
    ("simd_fadd", "simd_fadd r{d}, r{d}", "simd"),
    ("simd_min.u", "simd_min.u r{d}, r{d}", "simd"),
    ("quad_iadd", "quad_iadd r{d}, r{d}", "simd"),
    ("icmp_ballot", "icmp_ballot r{d}, ult, r{d}, r20", "simd"),
    ("fcmp_ballot", "fcmp_ballot r{d}, lt, r{d}, r20", "simd"),
];

pub fn op(name: &str) -> (&'static str, &'static str) {
    OPS.iter().find(|o| o.0 == name).map(|o| (o.1, o.2)).unwrap_or_else(|| panic!("unknown bench op {}", name))
}

fn consts_of(cls: &str) -> [u32; 4] {
    match cls {
        "int" | "move" | "convert" | "sfu" | "simd" => [1, 3, 5, 7],
        "int64" => [1, 0, 5, 7],
        "bitfield" => [0x5a5a1234, 3, 5, 7],
        "select" => [100, 7, 5, 7],
        "float" => [0, 0x3f800000, 5, 7],
        "half" => [0, 0x3c00, 5, 7],
        _ => panic!("class {}", cls),
    }
}

fn init_of(cls: &str, r: u32) -> u32 {
    match cls {
        "float" | "sfu" | "convert" => 0x3fc00000,
        "half" => 0x3e00,
        _ => 0x01234567 + r,
    }
}

fn subst_consts(t: &str) -> String {
    t.replace("r20", "r24").replace("r21", "r25").replace("r22", "r26")
}

/// run_bench.step_bytes
pub fn step_bytes(name: &str, d: u32) -> Vec<u8> {
    let (t, _) = op(name);
    let t = subst_consts(t).replace("{d1}", &(d + 1).to_string()).replace("{d}", &d.to_string());
    e(&t)
}

/// run_bench.loop: `steps` steps of one benchmark op over `nchains` register chains, counted `trips` times.
pub fn bench_loop(name: &str, nchains: u32, steps: u32, trips: u32) -> Vec<u8> {
    let (_, cls) = op(name);
    let step = if cls == "int64" { 2 } else { 1 };
    let regs: Vec<u32> = (0..nchains).map(|k| 5 + step * k).collect();
    let mut head = mov(3, 0);
    head.extend(mov(4, trips));
    for (k, v) in consts_of(cls).iter().enumerate() {
        head.extend(mov(24 + k as u32, *v));
    }
    for &r in &regs {
        head.extend(mov(r, init_of(cls, r)));
    }
    if cls == "int64" {
        for &r in &regs {
            head.extend(mov(r + 1, 0));
        }
    }
    head.extend(e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1"));
    let mut lp = vec![];
    for k in 0..steps {
        lp.extend(step_bytes(name, regs[(k % nchains) as usize]));
    }
    lp.extend(e("iadd r3, 1, r3.sx"));
    lp.extend(e("while_icmp r0l, ult, r3, r4, 1"));
    let j = format!("jmp_exec_any pc-{}", lp.len());
    lp.extend(e2(&j, &j));
    head.extend(lp);
    head.extend(e("pop_exec r0l, 1"));
    head.extend(e("iadd r2, r5, 0"));
    head
}

/// cost_hw.POOL: benchmark forms usable in random kernels.
pub fn pool() -> Vec<&'static str> {
    OPS.iter().filter(|o| o.2 != "int64" && !o.1.contains("{d1}")).map(|o| o.0).collect()
}

/// cost_hw.rand_instr
pub fn rand_instr(rng: &mut PyRandom, ops: &[&'static str]) -> (&'static str, String) {
    let name = *rng.choice(ops);
    let (t, _) = op(name);
    let d = rng.randint(5, 12);
    let s = rng.randint(5, 12);
    let mut t = subst_consts(t);
    let first = t.find("{d}").unwrap();
    if let Some(off) = t[first + 3..].find("{d}") {
        let second = first + 3 + off;
        t = format!("{}{{s}}{}", &t[..second], &t[second + 3..]);
    }
    (name, t.replace("{d}", &d.to_string()).replace("{s}", &s.to_string()))
}

/// cost_hw.kernel: `length` random benchmark instructions over r5..r12.
pub fn kernel(seed: u64, length: usize) -> (Vec<String>, Vec<u8>) {
    let mut rng = PyRandom::new(seed);
    let pool = pool();
    let mut lines = vec![];
    let mut body = vec![];
    for _ in 0..length {
        let (_, t) = rand_instr(&mut rng, &pool);
        body.extend(e(&t));
        lines.push(t);
    }
    (lines, body)
}

/// cost_hw.mem_kernel: random ALU forms mixed with device/threadgroup loads and stores and waits.
pub fn mem_kernel(seed: u64, length: usize) -> (Vec<String>, Vec<u8>) {
    let p_mem = 0.15;
    let mut rng = PyRandom::new(seed);
    let pool = pool();
    let mut lines = vec![];
    let mut body = vec![];
    for _ in 0..length {
        let x = rng.random();
        let (t, b);
        if x < p_mem {
            let kind = *rng.choice(&["dload", "dload", "dstore", "tgload", "tgstore"]);
            let d = rng.randint(13, 15) as u32;
            let src = rng.randint(5, 12) as u32;
            let slot = rng.randint(0, 7);
            match kind {
                "dload" => {
                    let idx = *rng.choice(&[26, 27]);
                    t = format!("device_load 0, i32, x, r{}, u8_u9, r{}, unsigned", d, idx);
                    b = e(&t);
                }
                "dstore" => {
                    b = store(src, 8, 26);
                    t = format!("device_store r{} -> word 32", src);
                }
                "tgload" => {
                    b = tg(true, "i32", d, None, slot, 1);
                    t = format!("threadgroup_load r{} <- slot {}", d, slot);
                }
                _ => {
                    b = tg(false, "i32", src, None, slot, 1);
                    t = format!("threadgroup_store r{} -> slot {}", src, slot);
                }
            }
        } else if x < p_mem + 0.06 {
            b = e("wait 0");
            t = "wait 0".into();
        } else {
            let (_, tt) = rand_instr(&mut rng, &pool);
            b = e(&tt);
            t = tt;
        }
        lines.push(t);
        body.extend(b);
    }
    lines.push("wait 0".into());
    body.extend(e("wait 0"));
    (lines, body)
}

/// The decoded descriptor name of every instruction (debug aid).
pub fn names(code: &[u8]) -> Vec<String> {
    let mut out = vec![];
    let mut pc = 0;
    while pc < code.len() {
        match isa::decode_one(&code[pc..]) {
            Some((d, _, size)) => {
                out.push(d.name.to_string());
                pc += size;
            }
            None => break,
        }
    }
    let _ = (desc_by_name, Operand::Null);
    out
}

/// mem_probes.mem_loop: counted loop (trip count in r29) of `steps` device loads or stores, `wait 0`
/// after every 8. Index register j (r5..r12) starts at `data + lane*s + j*jstride + warp*w` (lane =
/// r1 & 31, or the global index r1 when `glob`; warp = r30); with `advance`, every index advances by
/// that many words after each wait and wraps with `mask` (table words 0..mask).
#[allow(clippy::too_many_arguments)]
pub fn mem_loop(steps: u32, store_kind: bool, s: u32, jstride: u32, w: u32, data: u32, advance: u32, mask: u32, glob: bool) -> Vec<u8> {
    let mut h = mov(3, 0);
    h.extend(e("iadd r4, r29, 0"));
    if glob && w == u32::MAX {
        return mem_loop_private(h, steps, s, jstride, data, advance, mask);
    }
    h.extend(e(if glob { "iadd r23, r1, 0" } else { "and r23, r1, 31" }));
    h.extend(mov(21, s));
    h.extend(mov(22, w));
    for k in 0..8u32 {
        let r = 5 + k;
        h.extend(mov(27, data + k * jstride));
        h.extend(e("imadd r27, r30, r22, r27"));
        h.extend(e(&format!("imadd r{}, r23, r21, r27", r)));
        if mask != 0 {
            h.extend(mov(27, mask));
            h.extend(e(&format!("and r{}, r{}, r27", r, r)));
        }
    }
    if advance != 0 {
        h.extend(mov(21, advance));
        h.extend(mov(22, mask));
    }
    for r in 13..21 {
        h.extend(mov(r, 0x5a5a5a5a));
    }
    h.extend(e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1"));
    let mut body = vec![];
    for k in 0..steps {
        let (ri, rd) = (5 + k % 8, 13 + k % 8);
        if store_kind {
            body.extend(store(rd, 8, ri));
        } else {
            body.extend(e(&format!("device_load 0, i32, x, r{}, u8_u9, r{}, unsigned", rd, ri)));
        }
        if k % 8 == 7 || k == steps - 1 {
            body.extend(e("wait 0"));
            if advance != 0 {
                for r2 in 5..13 {
                    body.extend(e(&format!("iadd r{}, r{}, r21", r2, r2)));
                    body.extend(e(&format!("and r{}, r{}, r22", r2, r2)));
                }
            }
        }
    }
    body.extend(e("iadd r3, 1, r3.sx"));
    body.extend(e("while_icmp r0l, ult, r3, r4, 1"));
    let j = format!("jmp_exec_any pc-{}", body.len());
    body.extend(e2(&j, &j));
    h.extend(body);
    h.extend(e("pop_exec r0l, 1"));
    h
}

/// mem_probes.mem_loop(private=True): each workgroup streams its own window (register-pair base
/// r24_r25 = u8_u9 + 4*(data + SR0*(mask+1)), index = (SR51*s + j*J + k*advance) & mask).
fn mem_loop_private(mut h: Vec<u8>, steps: u32, s: u32, jstride: u32, data: u32, advance: u32, mask: u32) -> Vec<u8> {
    h.extend(e("iadd r24, u8, 0"));
    h.extend(e("iadd r25, u9, 0"));
    h.extend(get_sr(0, 26));
    h.extend(mov(27, mask.wrapping_add(1)));
    h.extend(mov(28, data));
    h.extend(e("imadd r26, r26, r27, r28"));
    h.extend(e("iadd r24_r25, r24_r25, r26, lsl 2"));
    h.extend(get_sr(51, 23));
    h.extend(mov(21, s));
    for k in 0..8u32 {
        let r = 5 + k;
        h.extend(mov(27, k * jstride));
        h.extend(e(&format!("imadd r{}, r23, r21, r27", r)));
        h.extend(mov(27, mask));
        h.extend(e(&format!("and r{}, r{}, r27", r, r)));
    }
    h.extend(mov(21, advance));
    h.extend(mov(22, mask));
    for r in 13..21 {
        h.extend(mov(r, 0x5a5a5a5a));
    }
    h.extend(e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1"));
    let mut body = vec![];
    for k in 0..steps {
        body.extend(e(&format!("device_load 0, i32, x, r{}, r24_r25, r{}, unsigned", 13 + k % 8, 5 + k % 8)));
        if k % 8 == 7 || k == steps - 1 {
            body.extend(e("wait 0"));
            for r2 in 5..13 {
                body.extend(e(&format!("iadd r{}, r{}, r21", r2, r2)));
                body.extend(e(&format!("and r{}, r{}, r22", r2, r2)));
            }
        }
    }
    body.extend(e("iadd r3, 1, r3.sx"));
    body.extend(e("while_icmp r0l, ult, r3, r4, 1"));
    let j = format!("jmp_exec_any pc-{}", body.len());
    body.extend(e2(&j, &j));
    h.extend(body);
    h.extend(e("pop_exec r0l, 1"));
    h
}
