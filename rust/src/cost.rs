//! Port of `scripts/agx_cost.py`: AGX (G14C) cycle cost model and simulator. No GPU access.
//!
//! Units: hardware numbers in this repository are in *reference units* (a dependent 8-byte
//! `iadd x, x, 1` step = 2 units); 1 unit = 1.049 real cycles. Every parameter is converted to real
//! cycles, the simulation runs in 0.1-cycle ticks, and `to_ref` converts results back.
//!
//! The port is exact: same decode classification, same dynamic trace (applegpu emulator semantics),
//! same issue/arbitration/token-bucket rules, the same float operations in the same order and
//! Python's round-half-to-even for every tick conversion.

use crate::isa::{self, exec, fget, Desc, Operand};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

pub const REAL_CYCLES_PER_UNIT: f64 = 1.049;
const ISSUE_PER_CORE: i64 = 4;
const RESULT_CACHE: usize = 8;
const RF_WRITEBACK: f64 = 18.0;
const RF_WRITEBACK16: f64 = 18.0;
pub const TOKSCALE: i64 = 100;
pub const TICKS: i64 = 10;
const SFU_SPACING: f64 = 2.0;
const SFU_GROUP_BLOCK: f64 = 2.0;
const SFU_WB_GAP: f64 = 2.0;
const NEG_INF: i64 = -1_000_000_000;

const ALU: &[(&str, &str)] = &[
    ("iadd", "iadd"), ("imadd", "imadd"), ("bitop", "and"), ("bfi", "bfi"), ("bfeil", "bfeil"), ("extr", "extr"),
    ("shlhi", "shlhi"), ("shrhi", "shrhi"), ("asr", "asr"), ("asrh", "asrh"), ("bitrev", "bitrev"),
    ("popcount", "popcount"), ("ffs", "ffs"), ("intl", "intl"), ("iunop", "bitrev"), ("icmpsel", "icmpsel"),
    ("fcmpsel", "fcmpsel"), ("mov_imm", "mov_imm"), ("fadd32", "fadd32"), ("fmul32", "fmul32"), ("fmadd32", "fmadd32"),
    ("fadd16", "fadd16"), ("fmul16", "fmul16"), ("fmadd16", "fmadd16"), ("convert", "convert_f2i"), ("floor", "floor"),
    ("ceil", "floor"), ("trunc", "floor"), ("rint", "rint"), ("rcp", "rcp"), ("rsqrt", "rsqrt"),
    ("rsqrt_special", "rsqrt_special"), ("log2", "log2"), ("exp2", "exp2"), ("sin_pt_1", "sin_pt_1"),
    ("sin_pt_2", "sin_pt_2"), ("funop", "floor"), ("dfdx", "sin_pt_1"), ("dfdy", "sin_pt_1"),
    ("simd_shuffle", "simd_shuffle"), ("simd_shuffle_down", "simd_shuffle_down"), ("simd_shuffle_up", "simd_shuffle_down"),
    ("simd_shuffle_xor", "simd_shuffle_xor"), ("simd_shuffle_rotate_up", "simd_shuffle_down"),
    ("simd_shuf_op", "simd_shuffle"), ("quad_shuffle", "quad_shuffle"), ("quad_shuffle_down", "quad_shuffle"),
    ("quad_shuffle_up", "quad_shuffle"), ("quad_shuffle_xor", "quad_shuffle"), ("quad_shuffle_rotate_up", "quad_shuffle"),
    ("icmp_ballot", "icmp_ballot"), ("fcmp_ballot", "fcmp_ballot"), ("icmp_quad_ballot", "icmp_ballot"),
    ("fcmp_quad_ballot", "fcmp_ballot"), ("get_sr", "get_sr"), ("TODO.sr_thing", "get_sr"),
];
const UNMEASURED_ALU: &[&str] = &[
    "iunop", "ceil", "trunc", "funop", "dfdx", "dfdy", "simd_shuffle_up", "simd_shuffle_rotate_up", "simd_shuf_op",
    "quad_shuffle_down", "quad_shuffle_up", "quad_shuffle_xor", "quad_shuffle_rotate_up", "icmp_quad_ballot",
    "fcmp_quad_ballot", "TODO.sr_thing",
];
const LOADS: &[&str] = &[
    "device_load", "device_load.TODO", "threadgroup_load", "stack_load", "async_load", "texture_sample", "texture_load",
    "iter", "iterproj", "TODO.ldcf", "TODO.ld/st_tile",
];
const STORES: &[&str] = &[
    "device_store", "device_store.TODO", "threadgroup_store", "stack_store", "uniform_store", "image_write",
    "image_write_block", "async_store", "st_var", "st_var_final", "tex_state_store", "sampler_state_store",
];
const MEASURED_MEMORY: &[&str] = &["device_load", "device_store", "threadgroup_load", "threadgroup_store"];
pub const CONTROL: &[&str] = &[
    "if_icmp", "if_fcmp", "else_icmp", "else_fcmp", "while_icmp", "while_fcmp", "pop_exec", "push_exec", "jmp_exec_any",
    "jmp_exec_none", "stop", "trap", "TODO.unk51", "no_var",
];
const BARRIERS: &[&str] = &["threadgroup_barrier", "memory_barrier", "wait_pix", "signal_pix", "TODO.unkE800"];
const UNIT_GROUPS: &[&[&str]] = &[
    &["iadd", "isub", "iadd_lsl2", "iadd_imm", "iadd_cache"], &["imadd", "imul", "imadd_cache"],
    &["fadd32", "fadd32_imm", "fadd32_cache", "fadd32_cache2"], &["fmul32", "fmul32_cache"],
    &["fmadd32", "fmadd32_imm", "fmadd32_cache"], &["and", "xor"],
];
const SHARED_UNITS: &[(&str, &[&str])] = &[
    ("sfu", &["rcp", "rsqrt", "rsqrt_special", "log2", "exp2", "sin_pt_1", "sin_pt_2", "floor", "rint"]),
    ("imadbf", &["imadd", "imul", "imadd_cache", "bfi", "bfeil", "extr", "shlhi", "shrhi", "asr", "asrh", "bitrev",
                 "popcount", "ffs", "intl", "convert_f2i", "convert_i2f"]),
];
const SFU_SLOW: &[(&str, f64)] = &[("rcp", 3.0), ("rsqrt", 4.0), ("rsqrt_special", 4.0)];
const CORE_SFU_WB_GAP: f64 = 1.0;
/// Core structure from the core-scaling probe (results/2026-10-11-core-scaling): 4 schedulers per core,
/// each issuing at most one instruction per cycle; SIMD group g runs on scheduler g % 4. Schedulers
/// {0, 2} and {1, 3} form two halves; each half has 2 32-bit register-operand reads per cycle and one
/// unit for the transcendental / integer-multiply / bit-field / convert / 64-bit-add class. An ALU
/// instruction's issue interval blocks its scheduler.
pub const SCHEDULERS: usize = 4;
const STEAL: bool = true;
const HALF_READ_PORTS: f64 = 2.0;
/// A 32-bit source written by the same SIMD group at most this many real cycles earlier is forwarded and
/// takes no register-file read port (one dependent chain per group reads fewer ports than eight).
const FORWARD_WINDOW: f64 = 4.0;
/// Per-half unit occupancy in real cycles (reference units at 32 groups x 1.049).
const HALF_UNIT_SLOW: &[(&str, f64)] = &[("rcp", 2.94 * 1.049), ("rsqrt", 3.92 * 1.049), ("rsqrt_special", 3.92 * 1.049)];
const HALF_UNIT_FAST: f64 = 1.96 * 1.049;
/// 64-bit add and 32x32->64 multiply-add occupy the same half unit (no sfu-pipe rules otherwise).
const HALF_UNIT_WIDE: &[(&str, f64)] = &[("iadd64", 3.92 * 1.049), ("imadd_wide", 3.92 * 1.049)];

fn lookup<'a, T: Copy>(table: &'a [(&str, T)], k: &str) -> Option<T> {
    table.iter().find(|(n, _)| *n == k).map(|(_, v)| *v)
}

/// agx_cost.REDUCE: `^(simd|quad)_(prefix_)?(iadd|fadd|fmul|and|or|xor|min|max)`
pub fn is_reduce(name: &str) -> bool {
    let rest = match name.strip_prefix("simd_").or_else(|| name.strip_prefix("quad_")) {
        Some(r) => r,
        None => return false,
    };
    let rest = rest.strip_prefix("prefix_").unwrap_or(rest);
    ["iadd", "fadd", "fmul", "and", "or", "xor", "min", "max"].iter().any(|p| rest.starts_with(p))
}

/// Python `int(round(x))` (round half to even).
#[inline]
pub fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

#[inline]
fn tk(c: f64) -> i64 {
    pyround(c * TICKS as f64)
}

#[derive(Clone, Debug)]
pub struct Row {
    pub lat: f64,
    pub ii: f64,
    pub ipc: f64,
    pub tmpl: String,
}

/// Model parameters (agx_cost.Params).
#[derive(Clone, Debug)]
pub struct Params {
    pub rows: HashMap<String, Row>,
    pub cal: Value,
    pub unit_cap: HashMap<String, f64>,
    pub unit_bucket: HashMap<String, String>,
    pub shared_of: HashMap<String, String>,
    pub load_fast: f64,
    pub load_latency: f64,
    pub defer_fast: f64,
    pub defer_slow: f64,
    pub defer_delayed_extra: f64,
    pub tg_defer: f64,
    pub load_ii: f64,
    pub dev_store_issue: (f64, f64),
    pub dev_store_done: (f64, f64),
    pub tg_load_lat: (f64, f64),
    pub tg_load_issue: f64,
    pub tg_store_issue: (f64, f64),
    pub space_switch: f64,
    pub quiet_load: f64,
    pub quiet_tg: f64,
    pub quiet_device_store: f64,
    pub wait_idle: f64,
    pub wait_drain: f64,
    pub store_after_wait: f64,
    pub wait_idle_big: f64,
    pub branch: Option<f64>,
    pub allow_unknown: bool,
}

pub fn repo_root() -> PathBuf {
    if let Ok(r) = std::env::var("AGX_SIM_REPO") {
        return PathBuf::from(r);
    }
    // rust/ lives at the repository root
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent().map(Path::to_path_buf).unwrap_or(here)
}

pub fn default_bench() -> PathBuf {
    repo_root().join("results/2026-10-11-instruction-bench/bench.json.gz")
}

pub fn default_calib() -> PathBuf {
    repo_root().join("results/2026-10-11-cost-model/calibration.json")
}

pub fn read_gz_json(path: &Path) -> Result<Value, String> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut s = String::new();
    flate2::read::GzDecoder::new(f).read_to_string(&mut s).map_err(|e| format!("{}: {}", path.display(), e))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {}", path.display(), e))
}

pub fn read_json(path: &Path) -> Result<Value, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {}", path.display(), e))
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64()
}

impl Params {
    pub fn new() -> Result<Self, String> {
        Self::load(&default_bench(), &default_calib(), None)
    }

    pub fn load(bench: &Path, calib: &Path, load_latency: Option<f64>) -> Result<Self, String> {
        let k = REAL_CYCLES_PER_UNIT;
        let b = read_gz_json(bench)?;
        let ops = b.get("ops").and_then(Value::as_object).ok_or("bench: no ops")?;
        let mut rows = HashMap::new();
        for (name, v) in ops {
            let g = |a: &str, b: &str| v.get(a).and_then(|x| x.get(b)).and_then(num).ok_or(format!("bench {}: {}.{}", name, a, b));
            rows.insert(name.clone(), Row {
                lat: g("latency", "cycles_per_step")? * k,
                ii: (g("simd_issue", "cycles_per_step")? * k).max(1.0),
                ipc: g("gpu", "lanes_per_cycle_per_core")? / 32.0 / k,
                tmpl: v.get("template").and_then(Value::as_str).unwrap_or("").to_string(),
            });
        }
        let cal = if calib.exists() { read_json(calib)? } else { Value::Object(Default::default()) };
        let c = |key: &str, d: f64| cal.get(key).and_then(num).unwrap_or(d);
        let measured = |key: &str, d: f64| cal.get("measured").and_then(|m| m.get(key)).and_then(num).unwrap_or(d);
        rows.insert("get_sr".into(), Row {
            lat: c("get_sr_latency", 3.0) * k,
            ii: (measured("get_sr_issue_cycles", 2.0) * k).max(1.0),
            ipc: 4.0,
            tmpl: "get_sr".into(),
        });
        let mut unit_cap: HashMap<String, f64> = rows.iter().map(|(k, v)| (k.clone(), v.ipc)).collect();
        let mut unit_bucket: HashMap<String, String> = rows.keys().map(|k| (k.clone(), k.clone())).collect();
        let mut shared_of: HashMap<String, String> = HashMap::new();
        for (u, ks) in SHARED_UNITS {
            for kk in *ks {
                shared_of.insert(kk.to_string(), u.to_string());
            }
        }
        if cal.get("merge_sfu_imad").map(|v| v.as_bool().unwrap_or(false) || v.as_f64().map(|x| x != 0.0).unwrap_or(false)).unwrap_or(false) {
            for v in shared_of.values_mut() {
                *v = "sfu".into();
            }
        }
        for grp in UNIT_GROUPS {
            let best = grp.iter().filter_map(|k| rows.get(*k)).map(|r| r.ipc).fold(f64::NEG_INFINITY, f64::max);
            for kk in *grp {
                if rows.contains_key(*kk) {
                    unit_cap.insert(kk.to_string(), best);
                    unit_bucket.insert(kk.to_string(), grp[0].to_string());
                }
            }
        }
        // SIMD/quad reductions: per-core rate measured with 32 SIMD groups (the 4-group bench is latency
        // bound for them); only ever raises the cap.
        let scaling = repo_root().join("results/2026-10-11-core-scaling/grid.json");
        if let Ok(sc) = read_json(&scaling) {
            for r in sc["rows"].as_array().into_iter().flatten() {
                let (Some(op), Some(t)) = (r["op"].as_str(), r["core_cycles_per_step"].as_f64()) else { continue };
                if is_reduce(op) && r["chains"].as_u64() == Some(8) && r["groups"].as_u64() == Some(32) && unit_cap.contains_key(op) {
                    let cap = 32.0 / (t * k);
                    let cur = unit_cap[op];
                    unit_cap.insert(op.to_string(), cur.max(cap));
                }
            }
        }
        let g = |key: &str, d: f64| c(key, d) * k;
        let load_latency = match load_latency {
            Some(l) if l != 0.0 => l,
            _ => c("device_load_slow_ref", 62.0) * k,
        };
        Ok(Params {
            load_fast: c("device_load_fast_ref", 41.6) * k,
            load_latency,
            defer_fast: c("device_load_defer_fast_ref", 31.0) * k,
            defer_slow: c("device_load_defer_slow_ref", 56.0) * k,
            defer_delayed_extra: c("device_load_defer_delayed_extra_ref", 5.0) * k,
            tg_defer: c("tg_load_defer_ref", 20.0) * k,
            load_ii: (c("device_load_extra_inflight_ref", 2.6) * k).max(1.0),
            dev_store_issue: (g("device_store_fast_issue_ref", 3.1), g("device_store_slow_issue_ref", 35.7)),
            dev_store_done: (g("device_store_done_fast_ref", 18.3), g("device_store_done_slow_ref", 4.7)),
            tg_load_lat: (g("tg_load_fast_ref", 19.4), g("tg_load_slow_ref", 38.4)),
            tg_load_issue: g("tg_load_issue_ref", 21.0),
            tg_store_issue: (g("tg_store_fast_issue_ref", 17.1), g("tg_store_slow_issue_ref", 36.2)),
            space_switch: g("mem_space_switch_ref", 13.3),
            quiet_load: g("mem_quiet_load_ref", 20.0),
            quiet_tg: g("mem_quiet_tg_ref", 20.0),
            quiet_device_store: g("mem_quiet_device_store_ref", 33.6),
            wait_idle: g("wait_idle_ref", 4.89),
            wait_drain: g("wait_drain_ref", 17.0),
            store_after_wait: g("store_after_wait_ref", 10.6),
            wait_idle_big: g("wait_idle_big_ref", 10.9),
            branch: cal.get("branch_taken_real").and_then(num),
            allow_unknown: false,
            rows,
            cal,
            unit_cap,
            unit_bucket,
            shared_of,
        })
    }

    pub fn row(&self, name: &str) -> Option<&Row> {
        self.rows.get(name)
    }

    fn cal_f(&self, key: &str, d: f64) -> f64 {
        self.cal.get(key).and_then(num).unwrap_or(d)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Alu,
    Load,
    Store,
    TgAtomic,
    Wait,
    Control,
    Barrier,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Alu => "alu",
            Kind::Load => "load",
            Kind::Store => "store",
            Kind::TgAtomic => "tg_atomic",
            Kind::Wait => "wait",
            Kind::Control => "control",
            Kind::Barrier => "barrier",
        }
    }
}

/// One decoded instruction with its cost class (agx_cost.decode record).
#[derive(Clone, Debug)]
pub struct Instr {
    pub pc: usize,
    pub size: usize,
    pub name: String,
    pub text: String,
    /// register halves, sorted and unique (Python sets)
    pub dst: Vec<u32>,
    pub src: Vec<u32>,
    pub reads32: usize,
    /// 32-bit register numbers read (the `reads32` set)
    pub src32: Vec<u32>,
    pub unit: Option<String>,
    pub kind: Kind,
    pub unmeasured: bool,
    pub slot: u64,
    pub cache: bool,
    pub uniform_src: bool,
    pub loop_bytes: Option<usize>,
    pub wait_run: usize,
}

fn set_of(v: impl IntoIterator<Item = u32>) -> Vec<u32> {
    let mut s: Vec<u32> = v.into_iter().collect();
    s.sort_unstable();
    s.dedup();
    s
}

fn regs_of(o: &Operand) -> Vec<u32> {
    let mut v = vec![];
    o.regs_of(&mut v);
    set_of(v)
}

/// The 2-byte `78 09` barrier over aligned blocks of 4 SIMD groups (playground barrier finding), which
/// applegpu does not decode.
pub fn is_barrier78(code: &[u8], pc: usize) -> bool {
    pc + 1 < code.len() && code[pc] == 0x78 && code[pc + 1] == 0x09
}

/// Mesa's stack_adjust (agx_pack.c AGX_OPCODE_STACK_ADJUST): opcode 0xb5 shared with stack_store.
pub fn is_stack_adjust(n: u128) -> bool {
    let fixed: u128 = 0x10100b5 | 1 << 26 | 2 << 36 | 1 << 47;
    let size_bits: u128 = 0xf << 20 | 0xf << 32 | 0xff << 56;
    n & ((1u128 << 64) - 1) & !size_bits == fixed
}

/// agx_cost.decode
pub fn decode(code: &[u8], p: &Params) -> Result<Vec<Instr>, String> {
    let mut out = vec![];
    let mut pc = 0;
    while pc < code.len() {
        if is_barrier78(code, pc) {
            out.push(Instr {
                pc, size: 2, name: "barrier_4groups".into(), text: "0x78 barrier (aligned 4 SIMD groups)".into(), dst: vec![], src: vec![],
                reads32: 0, src32: vec![], unit: None, kind: Kind::Barrier, unmeasured: false, slot: 0, cache: false, uniform_src: false,
                loop_bytes: None, wait_run: 0,
            });
            pc += 2;
            continue;
        }
        let n0 = isa::opcode_to_number(&code[pc..code.len().min(pc + 16)]);
        let d = isa::get_instruction_descriptor(n0).ok_or_else(|| format!("undecodable instruction at {}", pc))?;
        let size = d.decode_size(n0);
        let n = if size >= 16 { n0 } else { n0 & ((1u128 << (8 * size)) - 1) };
        let ops = d.decode_operands(n)?;
        let mut name = d.name.to_string();
        let mut text = isa::disassemble_normalized(d, n)?;
        let f = d.decode_fields(n);
        let regops: Vec<(&Operand, Vec<u32>)> = ops.iter().map(|o| (o, regs_of(o))).filter(|(_, r)| !r.is_empty()).collect();
        let union = |it: &[(&Operand, Vec<u32>)]| set_of(it.iter().flat_map(|(_, r)| r.iter().copied()));
        let mut rec = Instr {
            pc, size, name: String::new(), text: String::new(), dst: vec![], src: vec![], reads32: 0, src32: vec![], unit: None,
            kind: Kind::Alu, unmeasured: false, slot: 0, cache: false, uniform_src: false, loop_bytes: None, wait_run: 0,
        };
        let nm = d.name;
        if nm == "stack_store" && is_stack_adjust(n) {
            rec.kind = Kind::Control;
            name = "stack_adjust".into();
            text = format!("stack_adjust #{} (words)", ((n >> 20) & 0xf) | ((n >> 32) & 0xf) << 4 | (n >> 56) << 8);
        } else if LOADS.contains(&nm) {
            rec.kind = Kind::Load;
            rec.slot = fget(&f, "g").unwrap_or(0);
            rec.unmeasured = !MEASURED_MEMORY.contains(&nm);
            if !regops.is_empty() {
                rec.dst = regops[0].1.clone();
                rec.src = union(&regops[1..]);
            }
        } else if STORES.contains(&nm) {
            rec.kind = Kind::Store;
            rec.slot = fget(&f, "g").unwrap_or(0);
            rec.unmeasured = !MEASURED_MEMORY.contains(&nm);
            rec.src = union(&regops);
        } else if nm == "atomic" {
            let ret = fget(&f, "Rt").unwrap_or(0) != 0;
            let r = (fget(&f, "R").unwrap_or(0) | (fget(&f, "Rx").unwrap_or(0) << 6)) as u32;
            let width = if fget(&f, "op") == Some(3) { 4 } else { 2 };
            let s = fget(&f, "S").unwrap_or(0) as u32;
            let dst = if ret { vec![r, r + 1] } else { vec![] };
            let mut src: Vec<u32> = union(&regops).into_iter().filter(|h| !dst.contains(h)).collect();
            src.extend(s..s + width);
            rec.kind = if ret { Kind::Load } else { Kind::Store };
            rec.slot = fget(&f, "g").unwrap_or(0);
            rec.dst = set_of(dst);
            rec.src = set_of(src);
            rec.unmeasured = true;
        } else if nm == "threadgroup_atomic" {
            let ret = fget(&f, "Ra").unwrap_or(0) != 0;
            let r = (fget(&f, "R").unwrap_or(0) | (fget(&f, "Rx").unwrap_or(0) << 6)) as u32;
            let dst = if ret { vec![r, r + 1] } else { vec![] };
            rec.src = union(&regops).into_iter().filter(|h| !dst.contains(h)).collect();
            rec.dst = set_of(dst);
            rec.kind = Kind::TgAtomic;
            rec.unmeasured = true;
        } else if nm == "wait" {
            rec.kind = Kind::Wait;
            rec.slot = match ops.first() {
                Some(o) => isa::py_str(o)?.parse::<u64>().map_err(|e| e.to_string())?,
                None => 0,
            };
        } else if CONTROL.contains(&nm) {
            rec.kind = Kind::Control;
            let mut src = union(
                &regops.iter().filter(|(o, _)| !matches!(o, Operand::Reg16(0, _))).cloned().collect::<Vec<_>>(),
            );
            if !["jmp_exec_any", "jmp_exec_none", "stop", "trap"].contains(&nm) {
                src.push(0);
            }
            rec.src = set_of(src);
            rec.dst = if nm != "stop" && nm != "trap" { vec![0] } else { vec![] };
            rec.uniform_src = ops.iter().any(Operand::is_ureg);
        } else if BARRIERS.contains(&nm) {
            rec.kind = Kind::Barrier;
            rec.unmeasured = nm != "threadgroup_barrier";
        } else if lookup(ALU, nm).is_some() || is_reduce(nm) {
            rec.kind = Kind::Alu;
            if !regops.is_empty() {
                rec.dst = regops[0].1.clone();
                rec.src = union(&regops[1..]);
            }
            if nm == "get_sr" {
                rec.src = vec![];
            }
            let wide = set_of(
                regops
                    .iter()
                    .skip(1)
                    .filter(|(o, _)| matches!(o, Operand::Reg32(..) | Operand::Reg64(..)))
                    .flat_map(|(_, r)| r.iter().map(|h| h / 2)),
            );
            rec.reads32 = wide.len();
            rec.src32 = wide.clone();
            rec.cache = regops.iter().skip(1).any(|(o, _)| o.flags() & isa::operand::F_CACHE != 0);
            let imm = ops.iter().filter(|o| matches!(o, Operand::Imm(..))).count();
            let key = if is_reduce(nm) {
                rec.unmeasured = p.row(nm).is_none();
                if p.row(nm).is_some() {
                    nm.to_string()
                } else if nm.starts_with("quad") {
                    "quad_iadd".into()
                } else if nm.contains("prefix") {
                    "simd_prefix_iadd".into()
                } else {
                    "simd_iadd".into()
                }
            } else {
                let mut key = lookup(ALU, nm).unwrap().to_string();
                rec.unmeasured = UNMEASURED_ALU.contains(&nm);
                let first4 = regops.first().map(|(_, r)| r.len() == 4).unwrap_or(false);
                if nm == "iadd" && first4 {
                    key = "iadd64".into();
                }
                if nm == "imadd" && first4 {
                    key = "imadd_wide".into();
                }
                if imm > 0 && rec.reads32 <= 1 && p.row(&format!("{}_imm", key)).is_some() {
                    key += "_imm";
                } else if rec.cache && p.row(&format!("{}_cache", key)).is_some() {
                    key += "_cache";
                }
                key
            };
            rec.unit = Some(key);
        } else {
            if !p.allow_unknown {
                return Err(format!("agx_cost: no cost class for {:?} at pc {} ({})", nm, pc, text));
            }
            rec.kind = Kind::Alu;
            rec.unit = Some("mov_imm".into());
            rec.unmeasured = true;
            if !regops.is_empty() {
                rec.dst = regops[0].1.clone();
                rec.src = union(&regops[1..]);
            }
        }
        rec.name = name;
        rec.text = text;
        out.push(rec);
        pc += size;
    }
    mark_loops(&mut out);
    Ok(out)
}

/// agx_cost.mark_loops: innermost backward-jump loop size and consecutive-wait run length.
pub fn mark_loops(instrs: &mut [Instr]) {
    let mut loops = vec![];
    for r in instrs.iter() {
        if r.name == "jmp_exec_any" {
            if let Some(i) = r.text.find("pc-") {
                let digits: String = r.text[i + 3..].chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(off) = digits.parse::<usize>() {
                    loops.push((r.pc as i64 - off as i64, (r.pc + r.size) as i64));
                }
            }
        }
    }
    for r in instrs.iter_mut() {
        r.loop_bytes = loops
            .iter()
            .filter(|(a, b)| *a <= r.pc as i64 && (r.pc as i64) < *b)
            .map(|(a, b)| (b - a) as usize)
            .min();
    }
    let mut i = 0;
    while i < instrs.len() {
        if instrs[i].name == "wait" {
            let mut j = i;
            while j < instrs.len() && instrs[j].name == "wait" {
                j += 1;
            }
            for k in i..j {
                instrs[k].wait_run = j - i;
            }
            i = j;
        } else {
            i += 1;
        }
    }
}

pub const TABLE_VA: i128 = 0x1100100000;

#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub pcs: Vec<usize>,
    pub skipped: BTreeSet<String>,
    pub warnings: Vec<String>,
    /// device loads/stores: (index into `pcs`, distinct 64-byte segments touched by the active lanes)
    pub mem: Vec<(u32, Vec<u64>)>,
}

/// Where one SIMD group sits in a dispatch: enough to answer `get_sr` for the geometry registers
/// (playground experiments/registers/sr-geometry-core-alt.md) and to give r1 the global X index (the
/// direct-DRM harness prefix convention). Threads are linearised x fastest; SIMD group `group` holds
/// linear local indices 32*group .. 32*group+31.
#[derive(Clone, Debug)]
pub struct GroupCtx {
    pub wg_size: [u32; 3],
    pub wg_id: [u32; 3],
    pub group: u32,
    pub core: u32,
    pub slot: u32,
    /// 32-bit uniform registers preset (index, value), e.g. buffer addresses in u8_u9
    pub uniforms: Vec<(usize, u32)>,
}

impl GroupCtx {
    pub fn threads(&self) -> u32 {
        self.wg_size[0] * self.wg_size[1] * self.wg_size[2]
    }
    /// active lanes of this group (a partial last group has fewer)
    pub fn lanes(&self) -> usize {
        (self.threads().saturating_sub(32 * self.group)).min(32) as usize
    }
    fn local(&self, i: u32) -> [u32; 3] {
        let [x, y, _] = self.wg_size;
        [i % x, (i / x) % y, i / (x * y)]
    }
    /// Special register `sr` for lane `t`, or None when not modelled.
    pub fn sr(&self, sr: u32, t: usize) -> Option<u32> {
        let l = self.threads();
        let i = 32 * self.group + t as u32;
        let loc = self.local(i);
        Some(match sr {
            0..=2 => self.wg_id[sr as usize],
            4..=6 => self.wg_size[sr as usize - 4],
            8..=10 => self.wg_size[sr as usize - 8],
            7 => (l + 31) / 32,
            15 => 16,
            17 => self.slot,
            20 => self.core,
            24 | 25 => 2,
            48..=50 => loc[sr as usize - 48],
            51 => i,
            52 | 58 => i % 32,
            53 => i / 32,
            56 => i % 4,
            57 => 4.min(l - 4 * (i / 4)),
            59 => 32.min(l - 32 * (i / 32)),
            80..=82 => self.wg_id[sr as usize - 80] * self.wg_size[sr as usize - 80] + loc[sr as usize - 80],
            _ => return None,
        })
    }
}

/// Word addresses (byte / 4) of a threadgroup load / store / atomic for every active lane and component:
/// byte = base + 4 * size * index + size * component (playground threadgroup-addressing finding).
fn tg_words(f: &isa::Fields, st: &exec::CoreState) -> Option<Vec<u64>> {
    let g = |k: &str| fget(f, k);
    let item: i128 = match g("F") {
        Some(1 | 3 | 6 | 7) => 2,
        Some(2) | None => 4,
        _ => 1,
    };
    let mask = g("mask").unwrap_or(1);
    let mut w = vec![];
    for t in 0..exec::SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let base = if g("At")? == 0 { st.get_reg16(g("A")? as usize, t).ok()? } else { 0 };
        let idx = if g("Ot")? != 0 { g("O")? as i128 } else { st.get_reg16(g("O")? as usize, t).ok()? };
        for c in 0..4 {
            if mask & (1 << c) != 0 {
                w.push(((base + 4 * item * idx + item * c) / 4) as u64);
            }
        }
    }
    Some(w)
}

/// Word addresses of a device `atomic` (32-bit) for every active lane.
fn atomic_words(f: &isa::Fields, st: &exec::CoreState) -> Option<Vec<u64>> {
    let g = |k: &str| fget(f, k);
    let mut w = vec![];
    for t in 0..exec::SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let base = if g("At")? != 0 { st.ureg64((g("A")? >> 1) as usize).ok()? } else { st.get_reg64((g("A")? >> 1) as usize, t).ok()? };
        let off: i128 = if g("Ot")? != 0 {
            isa::desc::sign_extend(g("O")? as i128, 16)
        } else {
            let o = st.get_reg32((g("O")? >> 1) as usize, t).ok()?;
            if g("Ou")? == 0 { isa::desc::sign_extend(o, 32) } else { o }
        };
        w.push(((base + 4 * off) >> 2) as u64);
    }
    Some(w)
}

/// Per-lane byte addresses of a device_load / device_store (same formula as the emulator's
/// device_load), for the active lanes; None if an operand cannot be read.
fn mem_segments(f: &isa::Fields, st: &exec::CoreState) -> Option<Vec<u64>> {
    let g = |k: &str| fget(f, k);
    let mut segs = vec![];
    let fmt = g("F")?;
    let mut shift = g("s")?;
    if shift == 3 {
        shift = 2;
    }
    let bit_packed = matches!(fmt, 8 | 12 | 13);
    if bit_packed {
        shift = 2;
    }
    let item: i128 = match fmt {
        1 | 3 | 6 | 7 => 2,
        2 => 4,
        _ => 1,
    };
    let mask = if bit_packed { 0xF } else { g("mask")? };
    for t in 0..exec::SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let base = if g("At")? != 0 { st.ureg64((g("A")? >> 1) as usize).ok()? } else { st.get_reg64((g("A")? >> 1) as usize, t).ok()? };
        let mut off: i128 = if g("Ot")? != 0 {
            isa::desc::sign_extend(g("O")? as i128, 16)
        } else {
            let o = st.get_reg32((g("O")? >> 1) as usize, t).ok()?;
            if g("Ou")? == 0 { isa::desc::sign_extend(o, 32) } else { o }
        };
        off <<= shift;
        let a = base & !(item - 1);
        for i in 0..4 {
            if mask & (1 << i) != 0 {
                segs.push(((a + (off + i) * item) >> 6) as u64);
            }
        }
    }
    segs.sort_unstable();
    segs.dedup();
    Some(segs)
}

struct PcInfo {
    desc: &'static Desc,
    size: usize,
    fields: isa::Fields,
    ops: Vec<Operand>,
    rs: Vec<Vec<u32>>,
    /// halves of rs[1:] (sources for the taint rules)
    uses: Vec<u32>,
    /// first non-empty rs (the destination for the taint rules)
    dst: Vec<u32>,
    op: isa::compiled::Compiled,
    stop: bool,
    /// 0 = not a jump, 1 = jmp_exec_any, 2 = jmp_exec_none
    jump: u8,
    off: i64,
    /// control instruction whose condition is checked for uniforms / tainted sources
    check: bool,
    ureg: bool,
}

/// agx_cost.trace: dynamic pc sequence for one SIMD group (applegpu emulator; uniforms and memory
/// read as zero; instructions it cannot execute are skipped and taint their destination).
pub fn trace(code: &[u8], active: usize, max_steps: usize) -> Result<Trace, String> {
    trace_impl(code, active, max_steps, false, None)
}

/// `trace` for one SIMD group of a dispatch: `get_sr` returns the group's geometry registers, r1 holds
/// the global X index, uniforms are preset, and exec starts with the group's real lanes active.
pub fn trace_group(code: &[u8], ctx: &GroupCtx, max_steps: usize) -> Result<Trace, String> {
    trace_impl(code, ctx.lanes(), max_steps, false, Some(ctx))
}

/// `trace` driven by the reference interpreter (`isa::exec::exec`) instead of the compiled one.
pub fn trace_reference(code: &[u8], active: usize, max_steps: usize) -> Result<Trace, String> {
    trace_impl(code, active, max_steps, true, None)
}

fn trace_impl(code: &[u8], active: usize, max_steps: usize, reference: bool, ctx: Option<&GroupCtx>) -> Result<Trace, String> {
    let mut st = exec::CoreState::new(128, Some(exec::Memory { base: TABLE_VA, words: vec![0; 65536] }));
    for t in 0..exec::SIMD_WIDTH {
        st.set_reg16(0, t, if t < active { 0 } else { 1 })?;
        st.set_reg32(1, t, t as i128)?;
        st.exec[t] = t < active;
    }
    if let Some(c) = ctx {
        for t in 0..exec::SIMD_WIDTH {
            st.set_reg32(1, t, c.sr(80, t).unwrap_or(0) as i128)?;
        }
        for &(u, v) in &c.uniforms {
            st.uniforms[2 * u] = (v & 0xffff) as u16;
            st.uniforms[2 * u + 1] = (v >> 16) as u16;
        }
    }
    let mut out = Trace::default();
    let mut tainted = vec![false; 1024];
    let mut cache: Vec<Option<PcInfo>> = (0..code.len()).map(|_| None).collect();
    let mut pc: i64 = 0;
    while pc < code.len() as i64 {
        if pc < 0 {
            return Err(format!("trace: negative pc {}", pc));
        }
        let upc = pc as usize;
        if is_barrier78(code, upc) {
            out.pcs.push(upc);
            pc += 2;
            continue;
        }
        if cache[upc].is_none() {
            let n0 = isa::opcode_to_number(&code[upc..code.len().min(upc + 16)]);
            let desc = isa::get_instruction_descriptor(n0).ok_or_else(|| format!("trace: undecodable instruction at {}", upc))?;
            let size = desc.decode_size(n0);
            let n = desc.mask_instr(n0);
            let fields = desc.decode_fields(n);
            let stop = desc.name == "stop" || desc.name == "trap";
            let ops = if stop { vec![] } else { desc.fields_to_operands(&fields)? };
            let rs: Vec<Vec<u32>> = ops.iter().map(regs_of).collect();
            let uses = set_of(rs.iter().skip(1).flatten().copied());
            let dst = rs.iter().find(|r| !r.is_empty()).cloned().unwrap_or_default();
            let jump = match desc.name {
                "jmp_exec_any" => 1,
                "jmp_exec_none" => 2,
                _ => 0,
            };
            let mut off = fget(&fields, "off").unwrap_or(0) as i64;
            if off & (1 << 31) != 0 {
                off -= 1 << 32;
            }
            let check = CONTROL.contains(&desc.name) && !["jmp_exec_any", "jmp_exec_none", "pop_exec"].contains(&desc.name);
            let ureg = ops.iter().any(Operand::is_ureg);
            let op = isa::compiled::Compiled::new(desc, &fields, &ops, st.reg16s.len());
            cache[upc] = Some(PcInfo { desc, size, fields, ops, rs, uses, dst, op, stop, jump, off, check, ureg });
        }
        let info = cache[upc].as_ref().unwrap();
        out.pcs.push(upc);
        if out.pcs.len() > max_steps {
            return Err("trace step limit (non-terminating loop?)".into());
        }
        if info.stop {
            break;
        }
        if info.check && ((info.ureg && ctx.is_none()) || info.uses.iter().any(|&h| tainted[h as usize])) {
            let w = format!("pc {}: {} condition depends on a uniform or a skipped result (trip count unknown)", upc, info.desc.name);
            if !out.warnings.contains(&w) {
                out.warnings.push(w);
            }
        }
        if info.jump != 0 {
            let any = st.exec.iter().any(|&e| e);
            let take = if info.jump == 1 { any } else { !any };
            pc = if take { pc + info.off } else { pc + info.size as i64 };
            continue;
        }
        let dname = info.desc.name;
        if dname == "device_load" || dname == "device_store" {
            if let Some(sg) = mem_segments(&info.fields, &st) {
                out.mem.push(((out.pcs.len() - 1) as u32, sg));
            }
        } else if dname.starts_with("threadgroup_") && dname != "threadgroup_barrier" {
            if let Some(w) = tg_words(&info.fields, &st) {
                out.mem.push(((out.pcs.len() - 1) as u32, w));
            }
        } else if dname == "atomic" {
            if let Some(w) = atomic_words(&info.fields, &st) {
                out.mem.push(((out.pcs.len() - 1) as u32, w));
            }
        }
        if let (Some(c), "get_sr") = (ctx, dname) {
            let sr = fget(&info.fields, "SR").unwrap_or(0) as u32;
            if let (Some(Operand::Reg32(d, _)), Some(_)) = (info.ops.first(), c.sr(sr, 0)) {
                for t in 0..exec::SIMD_WIDTH {
                    if st.exec[t] {
                        st.set_reg32(*d as usize, t, c.sr(sr, t).unwrap() as i128)?;
                    }
                }
                for &h in &info.dst {
                    tainted[h as usize] = false;
                }
                pc += info.size as i64;
                continue;
            }
        }
        let r = if reference { exec::exec(info.desc, &info.fields, &info.ops, &mut st) } else { info.op.exec(&mut st) };
        match r {
            Ok(()) => {
                for &h in &info.dst {
                    tainted[h as usize] = false;
                }
                if info.uses.iter().any(|&h| tainted[h as usize]) {
                    for &h in &info.dst {
                        tainted[h as usize] = true;
                    }
                }
            }
            Err(_) => {
                out.skipped.insert(info.desc.name.to_string());
                for &h in &info.dst {
                    tainted[h as usize] = true;
                }
            }
        }
        let _ = &info.rs;
        pc += info.size as i64;
    }
    Ok(out)
}

/// agx_cost.fit_branch: taken-branch penalty so an empty counted-loop trip costs the measured 35.25 units.
pub fn fit_branch(p: &mut Params) -> Result<f64, String> {
    p.branch = Some(0.0);
    let mut c = [0.0; 2];
    for (i, t) in [200u32, 1000].into_iter().enumerate() {
        let code = crate::kernels::counted(&[], t, true, crate::kernels::DEFAULT_CONSTS);
        let ins = decode(&code, p)?;
        let tr = trace(&code, 32, 500_000)?;
        c[i] = simulate(&ins, &tr.pcs, p, 1, false)?.0;
    }
    let target = p
        .cal
        .get("measured")
        .and_then(|m| m.get("loop_trip_cycles_empty"))
        .and_then(Value::as_f64)
        .unwrap_or(35.25)
        * REAL_CYCLES_PER_UNIT;
    let b = (target - (c[1] - c[0]) / 800.0).max(0.0);
    p.branch = Some(b);
    Ok(b)
}

#[derive(Clone, Debug, Default)]
pub struct PcStats {
    pub n: u64,
    pub stall: f64,
    pub issue: f64,
    pub branch: f64,
}

pub fn to_ref(real_cycles: f64) -> f64 {
    real_cycles / REAL_CYCLES_PER_UNIT
}

/// Pre-resolved per-instruction simulation data (all in ticks).
struct SimInstr {
    kind: Kind,
    pc: usize,
    src: Vec<u32>,
    dst: Vec<u32>,
    dst_regs: Vec<u32>,
    ndst: usize,
    slot: usize,
    unit: Option<usize>,
    bucket: usize,
    cost: f64,
    need_port: f64,
    src32: Vec<u32>,
    reduce: bool,
    lat: i64,
    ii: i64,
    sfu_op: bool,
    sfu_slow: Option<i64>,
    core_slow: Option<i64>,
    core_fast_cost: i64,
    /// 64-bit forms on the half unit: occupancy in ticks
    wide_cost: Option<i64>,
    tgop: bool,
    tg_name: bool,
    /// device atomic (load kind when it returns, store kind otherwise)
    atomic: bool,
    /// 0 = not a barrier, 1 = whole workgroup (0x68), 2 = aligned 4 SIMD groups (0x78)
    barrier: u8,
    mem_issue0: i64,
    mem_done: (i64, i64),
    is_jmp_any: bool,
    ctl_bump: bool,
    /// exec-mask instructions other than the loop's while/jmp_exec_any
    exec_stack: bool,
}

struct Bucket {
    shared: bool,
    cap: f64,
    init: f64,
    /// refill ceiling: 2*T*TOKSCALE for shared units, (cap+1)*T*TOKSCALE otherwise
    capmax: f64,
}

#[inline]
fn pymin(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Device-memory hierarchy (results/2026-10-11-memory-hierarchy): per-core L1 and L2, a shared system-level
/// cache and DRAM, with measured latencies and per-core bandwidths. Bandwidths are in 64-byte segments per
/// real cycle per core; latencies beyond L2 are in ns and scale with `clock_mhz`.
#[derive(Clone, Debug)]
pub struct MemModel {
    pub clock_mhz: f64,
    /// cores sharing the SLC and DRAM (all cores run the same kernel on disjoint data)
    pub active_cores: f64,
    pub l1_bytes: usize,
    pub l2_bytes: usize,
    pub slc_bytes: usize,
    /// extra load latency over an L1 hit: L2 in real cycles; SLC and DRAM total latency in ns
    pub l2_extra: f64,
    pub slc_ns: f64,
    pub dram_ns: f64,
    pub l1_load_rate: f64,
    pub l1_load_min: f64,
    pub l1_store_rate: f64,
    pub l1_store_min: f64,
    pub l2_rate: f64,
    pub slc_rate: f64,
    pub dram_gbps: f64,
    /// replay every group's accesses once before timing (steady state of a long-running loop)
    pub warm: bool,
    /// the dispatch's device data already sits in the system-level cache (written by the CPU or a previous
    /// dispatch just before, and small enough): first touches hit the SLC instead of DRAM. agx_dispatch sets it
    /// when the estimated GPU-wide footprint fits `slc_bytes`.
    pub resident_slc: bool,
    /// device atomics (no return, distinct words): per-core occupancy max(atomic_min, atomic_per_seg x 32-byte
    /// sectors) reference cycles; lanes on one word serialise GPU-wide at atomic_same_word_rate per cycle
    pub atomic_min: f64,
    pub atomic_per_seg: f64,
    pub atomic_same_word_rate: f64,
    /// threadgroup memory, one SIMD group back to back: reference cycles per load / store plus one per extra
    /// bank-conflict way (core: max(1, conflict degree) per instruction)
    pub tg_group_load: f64,
    pub tg_group_store: f64,
}

impl Default for MemModel {
    fn default() -> Self {
        let k = REAL_CYCLES_PER_UNIT;
        MemModel {
            clock_mhz: 1398.0,
            active_cores: 38.0,
            l1_bytes: 8 << 10,
            l2_bytes: 256 << 10,
            slc_bytes: 16 << 20,
            l2_extra: 90.0,
            slc_ns: 349.0,
            dram_ns: 388.0,
            l1_load_rate: 2.0 / k,
            l1_load_min: 2.0,
            l1_store_rate: 1.02 / k,
            l1_store_min: 2.0,
            l2_rate: 0.73 / k,
            slc_rate: 0.5 / k,
            dram_gbps: 410.0,
            warm: true,
            resident_slc: false,
            atomic_min: 6.9,
            atomic_per_seg: 1.7,
            atomic_same_word_rate: 2.0,
            tg_group_load: 31.0,
            tg_group_store: 24.0,
        }
    }
}

/// Least-recently-used set of 64-byte segments.
struct Lru {
    cap: usize,
    stamp: u64,
    at: HashMap<u64, u64>,
    order: std::collections::BTreeMap<u64, u64>,
}

impl Lru {
    fn new(bytes: usize) -> Self {
        Lru { cap: (bytes / 64).max(1), stamp: 0, at: HashMap::new(), order: Default::default() }
    }
    fn contains(&self, s: u64) -> bool {
        self.at.contains_key(&s)
    }
    fn touch(&mut self, s: u64) {
        self.stamp += 1;
        if let Some(old) = self.at.insert(s, self.stamp) {
            self.order.remove(&old);
        }
        self.order.insert(self.stamp, s);
        if self.at.len() > self.cap {
            let (&o, &victim) = self.order.iter().next().unwrap();
            self.order.remove(&o);
            self.at.remove(&victim);
        }
    }
}

/// agx_cost.simulate: `groups` identical SIMD groups on one core; returns real cycles (and per-pc stats).
pub fn simulate(instrs: &[Instr], pcs: &[usize], p: &Params, groups: usize, detail: bool)
    -> Result<(f64, Option<HashMap<usize, PcStats>>), String> {
    let seqs: Vec<&[usize]> = (0..groups).map(|_| pcs).collect();
    simulate_groups(instrs, &seqs, None, p, detail, None)
}

/// One core running one SIMD group per entry of `seqs` (dynamic pc sequences, e.g. from `trace_group`).
/// With `mem` (each group's `Trace::mem` and a `MemModel`), device loads and stores go through the
/// memory hierarchy; without it they use the L1-hit rules only (as `simulate`).
/// `wg_of`: workgroup of each group (barrier scope; None = all groups in one workgroup). Groups of a
/// workgroup are listed in SIMD-group order.
pub fn simulate_groups(instrs: &[Instr], seqs_pcs: &[&[usize]], mem: Option<(&[&[(u32, Vec<u64>)]], &MemModel)>, p: &Params,
    detail: bool, wg_of: Option<&[u32]>) -> Result<(f64, Option<HashMap<usize, PcStats>>), String> {
    let groups = seqs_pcs.len();
    let wg: Vec<u32> = wg_of.map(|w| w.to_vec()).unwrap_or_else(|| vec![0; groups]);
    let local: Vec<usize> = (0..groups).map(|i| (0..i).filter(|&j| wg[j] == wg[i]).count()).collect();
    let mut bar_count = vec![0u64; groups];
    let mut bar_arrive: Vec<Option<i64>> = vec![None; groups];
    let mut bar_release: HashMap<(u32, usize, u64), i64> = HashMap::new();
    let p_branch = match p.branch {
        Some(b) => b,
        None => {
            let mut q = p.clone();
            fit_branch(&mut q)?
        }
    };
    let t_ = TICKS;
    let w = tk(p.cal_f("rf_writeback_real", RF_WRITEBACK));
    let w16 = tk(p.cal_f("rf_writeback16_real", RF_WRITEBACK16));
    let nc = p.cal.get("result_cache_entries").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(RESULT_CACHE);
    let br = tk(p_branch);
    let cl = tk(p.cal_f("control_latency", 2.0));
    let ci = tk(p.cal_f("control_issue_cycles", 1.0));
    let wait_idle = tk(p.wait_idle);
    let wait_drain = tk(p.wait_drain);
    let quiet_tg = tk(p.quiet_tg);
    let quiet_load = tk(p.quiet_load);
    let quiet_dstore = tk(p.quiet_device_store);
    let tg_lat0 = tk(p.tg_load_lat.0);
    let tg_issue0 = tk(p.tg_load_issue);
    let load_fast = tk(p.load_fast);
    let load_ii = tk(p.load_ii);
    let space_switch = tk(p.space_switch);
    let tg_defer = tk(p.tg_defer);
    let defer_fast = tk(p.defer_fast);
    let defer_extra = tk(p.defer_delayed_extra);
    let store_after_wait = tk(p.store_after_wait);
    let load_latency = tk(p.load_latency);
    let sfu_spacing = tk(SFU_SPACING);
    let sfu_wb_gap = tk(SFU_WB_GAP);
    let sfu_group_block = tk(SFU_GROUP_BLOCK);
    let core_wb_gap = tk(CORE_SFU_WB_GAP);
    let fwd = tk(FORWARD_WINDOW);
    // barrier release after the last arrival (barrier_loop probe: 13.0 / 7.1 reference units back to back)
    let bar68 = tk(13.0 * REAL_CYCLES_PER_UNIT);
    let bar78 = tk(7.1 * REAL_CYCLES_PER_UNIT);
    // a group arrives once its ALU results have drained (4 iadds before each barrier: 42 vs 13 units)
    let bar_drain = tk(24.0 * REAL_CYCLES_PER_UNIT);
    let ctl_sched = tk(std::env::var("AGX_CTL_SCHED").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(3.0) * REAL_CYCLES_PER_UNIT);

    // ---- resolve units and buckets
    let mut bucket_ids: HashMap<String, usize> = HashMap::new();
    let mut buckets: Vec<Bucket> = vec![];
    let mut bucket_of = |u: &str| -> (usize, f64, f64) {
        let sh = p.shared_of.get(u).map(String::as_str);
        let (name, cost, cap) = if matches!(sh, Some("sfu") | Some("imadbf")) {
            ("sfu_core_free".to_string(), 0.0, 1.0) // SFU_PIPE with the core-level pipe
        } else if let Some(s) = sh {
            (s.to_string(), pyround((t_ * TOKSCALE) as f64 / p.unit_cap[u].max(1e-3)) as f64, 1.0)
        } else {
            (p.unit_bucket[u].clone(), (t_ * TOKSCALE) as f64, p.unit_cap[u])
        };
        let id = *bucket_ids.entry(name.clone()).or_insert_with(|| {
            let shared = SHARED_UNITS.iter().any(|(s, _)| *s == name);
            let rcap = if shared { 1.0 } else { p.unit_cap.get(&name).copied().unwrap_or(1.0) };
            let capmax = if shared { (2 * t_ * TOKSCALE) as f64 } else { (rcap + 1.0) * t_ as f64 * TOKSCALE as f64 };
            buckets.push(Bucket { shared, cap: rcap, init: (cap + 1.0) * t_ as f64 * TOKSCALE as f64, capmax });
            buckets.len() - 1
        });
        (id, cost, cap)
    };
    let mut by_pc: HashMap<usize, usize> = HashMap::new();
    let mut sim: Vec<SimInstr> = vec![];
    let mut max_half = 0u32;
    for r in instrs {
        let sfu_op = r.unit.as_ref().map(|u| matches!(p.shared_of.get(u).map(String::as_str), Some("sfu") | Some("imadbf"))).unwrap_or(false);
        let (bucket, cost, unit, need_port) = match &r.unit {
            Some(u) => {
                let (b, c, _) = bucket_of(u);
                (b, c, Some(0usize), (r.reads32 as f64).min(HALF_READ_PORTS + 2.0) * t_ as f64)
            }
            None => (0, 0.0, None, 0.0),
        };
        let row = r.unit.as_ref().map(|u| p.rows.get(u).ok_or_else(|| format!("KeyError: {}", u))).transpose()?;
        let (lat, ii) = match row {
            Some(row) => (tk(row.lat), tk(row.ii)),
            None => (0, 0),
        };
        let u = r.unit.as_deref().unwrap_or("");
        let is_tg = r.name.starts_with("threadgroup");
        let (mem_issue0, mem_done) = if r.name == "threadgroup_store" {
            (tk(p.tg_store_issue.0), (0, 0))
        } else if r.name == "threadgroup_load" {
            // P.mem['threadgroup_load'] has no 'done' entry (never a store)
            (tk(p.tg_load_issue), (0, 0))
        } else {
            (tk(p.dev_store_issue.0), (tk(p.dev_store_done.0), tk(p.dev_store_done.1)))
        };
        let core_fast_cost = tk(HALF_UNIT_FAST);
        for &h in r.src.iter().chain(r.dst.iter()) {
            max_half = max_half.max(h);
        }
        by_pc.insert(r.pc, sim.len());
        sim.push(SimInstr {
            kind: r.kind,
            pc: r.pc,
            src: r.src.clone(),
            dst: r.dst.clone(),
            dst_regs: set_of(r.dst.iter().map(|h| h / 2)),
            ndst: r.dst.len(),
            slot: r.slot as usize,
            unit,
            bucket,
            cost,
            need_port,
            src32: r.src32.clone(),
            reduce: is_reduce(&r.name),
            lat,
            ii,
            sfu_op,
            sfu_slow: lookup(SFU_SLOW, u).map(tk),
            core_slow: lookup(HALF_UNIT_SLOW, u).map(tk),
            core_fast_cost,
            wide_cost: lookup(HALF_UNIT_WIDE, u).map(tk),
            tgop: r.name == "threadgroup_load",
            atomic: r.name == "atomic",
            barrier: match r.name.as_str() {
                "threadgroup_barrier" => 1,
                "barrier_4groups" => 2,
                _ => 0,
            },
            tg_name: is_tg,
            mem_issue0,
            mem_done,
            is_jmp_any: r.name == "jmp_exec_any",
            exec_stack: matches!(r.name.as_str(), "if_icmp" | "if_fcmp" | "else_icmp" | "else_fcmp" | "pop_exec" | "push_exec" | "jmp_exec_none"),
            ctl_bump: r.src.iter().any(|&h| h != 0),
        });
    }
    let mut seqs: Vec<Vec<usize>> = vec![];
    for pcs in seqs_pcs {
        seqs.push(pcs.iter().map(|pc| by_pc.get(pc).copied().ok_or_else(|| format!("KeyError: pc {}", pc))).collect::<Result<_, _>>()?);
    }
    // ---- memory hierarchy state (one core)
    struct MemState {
        l1: Lru,
        l2: Lru,
        slc: Lru,
        /// tokens (segments) for l1 load, l1 store, l2, slc, dram, and their rates per tick / caps
        tok: [f64; 5],
        rate: [f64; 5],
        cap: [f64; 5],
        extra: [i64; 4],
        cur: Vec<usize>,
        /// L1 load / store pipelines: busy until (ticks); a load's or store's segments pass in order
        l1_free: [i64; 2],
        /// threadgroup-memory and device-atomic pipelines: busy until (ticks)
        tg_free: i64,
        at_free: i64,
        tg_group_free: Vec<i64>,
    }
    let mut ms: Option<MemState> = mem.map(|(_, m)| {
        let per_tick = |segs_per_cycle: f64| segs_per_cycle / TICKS as f64;
        let dram = m.dram_gbps * 1e9 / m.active_cores / (m.clock_mhz * 1e6) / 64.0;
        let rate = [per_tick(m.l1_load_rate), per_tick(m.l1_store_rate), per_tick(m.l2_rate), per_tick(m.slc_rate), per_tick(dram)];
        let cap = [64.0, 64.0, 64.0, 64.0, 64.0];
        let ns = |x: f64| tk((x * m.clock_mhz / 1000.0 - 50.0).max(0.0));
        MemState {
            l1: Lru::new(m.l1_bytes),
            l2: Lru::new(m.l2_bytes),
            slc: Lru::new((m.slc_bytes as f64 / m.active_cores) as usize),
            tok: cap,
            rate,
            cap,
            extra: [0, tk(m.l2_extra), ns(m.slc_ns), ns(m.dram_ns)],
            cur: vec![0; groups],
            l1_free: [0, 0],
            tg_free: 0,
            at_free: 0,
            tg_group_free: vec![0; groups],
        }
    });
    let memv: Vec<&[(u32, Vec<u64>)]> = mem.map(|(v, _)| v.to_vec()).unwrap_or_default();
    if let (Some(m), Some((_, mm))) = (ms.as_mut(), mem) {
        if mm.resident_slc && !mm.warm {
            let mut distinct: std::collections::HashSet<u64> = Default::default();
            for (gi, v) in memv.iter().enumerate() {
                for (step, segs) in v.iter() {
                    let si = &sim[seqs[gi][*step as usize]];
                    let dev = !si.atomic && !si.tg_name && matches!(si.kind, Kind::Load | Kind::Store);
                    if dev {
                        distinct.extend(segs.iter().copied());
                    }
                }
            }
            for &sg in &distinct {
                m.slc.touch(sg);
            }
        }
        if mm.warm {
            let longest = memv.iter().map(|v| v.len()).max().unwrap_or(0);
            for i in 0..longest {
                for v in &memv {
                    if let Some((_, segs)) = v.get(i) {
                        for &sg in segs {
                            m.l1.touch(sg);
                            m.l2.touch(sg);
                            m.slc.touch(sg);
                        }
                    }
                }
            }
        }
    }
    let nhalf = (max_half as usize + 1).max(2);
    let nreg = nhalf / 2 + 1;
    let nslots = sim.iter().map(|s| s.slot + 1).max().unwrap_or(1).max(2);

    // ---- per-group state
    let g = groups;
    let mut nxt = vec![0usize; g];
    let mut free = vec![0i64; g];
    let mut ready = vec![vec![0i64; nhalf]; g];
    let mut present = vec![vec![false; nhalf]; g]; // defaultdict keys
    let mut present_list: Vec<Vec<u32>> = vec![vec![]; g];
    let mut crossed = vec![vec![i64::MIN; nhalf]; g];
    let mut slots: Vec<Vec<Option<i64>>> = vec![vec![None; nslots]; g]; // max of the pending list
    let mut rcache: Vec<Vec<u32>> = vec![Vec::with_capacity(nc + 1); g];
    let mut wtime: Vec<Vec<Option<(i64, bool)>>> = vec![vec![None; nreg]; g];
    let mut last_w: Vec<Vec<u8>> = vec![vec![0; nreg]; g];
    let mut tok: Vec<Option<f64>> = vec![None; buckets.len()];
    let mut refilling: Vec<usize> = vec![];
    let mut in_refill: Vec<bool> = vec![false; buckets.len()];
    let mut port = [HALF_READ_PORTS * t_ as f64; 2];
    let mut sched_free = [0i64; SCHEDULERS];
    let mut tick_issued = vec![false; groups];
    // an idle scheduler may issue a ready group of a ring neighbour (idle-slot probe: slots 1, 3 and 4 relieve
    // scheduler 0, slot 2 does not)
    let npass = if STEAL && groups > 4 { 2 } else { 1 };
    let mut issue_tok: i64 = ISSUE_PER_CORE * t_;
    let mut last_alu = vec![NEG_INF; g];
    let mut alu_done = vec![NEG_INF; g];
    let mut last_wait = vec![NEG_INF; g];
    let mut last_idle_wait = vec![NEG_INF; g];
    let mut sfu_iss = vec![NEG_INF; g];
    let mut sfu_wb = vec![NEG_INF; g];
    let mut sfu_slow = vec![NEG_INF; g];
    let (mut core_wb, mut core_slow, mut core_fast) = ([NEG_INF; 2], [NEG_INF; 2], [NEG_INF; 2]);
    let mut alu_since_mem = vec![false; g];
    // device loads of a group complete in order (a quiet-time delay holds back the whole batch);
    // only with the memory model, so the fitted L1 rules are unchanged without it
    let inorder_loads = true;
    let mut last_load_done = vec![NEG_INF; g];
    let mut last_space: Vec<u8> = vec![0; g]; // 0 None, 1 'tg', 2 'device_load', 3 'device_store'
    struct Pending {
        slot: usize,
        done: i64,
        lat: i64,
        ins: usize,
        /// latency beyond an L1 hit (memory hierarchy): a load issued after ALU work waits for done - extra
        extra: i64,
    }
    let mut pending: Vec<Vec<Pending>> = (0..g).map(|_| vec![]).collect();
    let mut last_t: i64 = 0;
    let mut stats: Vec<(u64, i64, i64, i64)> = vec![(0, 0, 0, 0); sim.len()];
    let mut wake: Vec<i64> = Vec::with_capacity(g);
    let mut sr_cache: Vec<Option<i64>> = vec![None; g];
    let mut blk_cache: Vec<Option<i64>> = vec![None; g];
    let mut order: Vec<usize> = Vec::with_capacity(g);

    #[inline]
    fn touch(ready: &mut [i64], present: &mut [bool], list: &mut Vec<u32>, h: u32) -> usize {
        let i = h as usize;
        if !present[i] {
            present[i] = true;
            list.push(h);
        }
        let _ = ready;
        i
    }

    let mut t: i64 = 0;
    let mut done = 0usize;
    let npcs: Vec<usize> = seqs.iter().map(Vec::len).collect();
    while done < g {
        let mut issued = 0;
        order.clear();
        let k0 = if g == 1 { 0 } else { (((t / t_) as u64).wrapping_mul(2654435761) >> 7) as usize % g };
        order.extend((k0..g).chain(0..k0));
        wake.clear();
        for flag in tick_issued.iter_mut() {
            *flag = false;
        }
        for pass in 0..npass {
        for oi in 0..order.len() {
            let gi = order[oi];
            if nxt[gi] >= npcs[gi] || tick_issued[gi] {
                continue;
            }
            if free[gi] > t {
                wake.push(free[gi]);
                continue;
            }
            let ri = seqs[gi][nxt[gi]];
            let r = &sim[ri];
            // src_ready. It depends only on this group's ready / wtime / rcache, which change only when
            // this group issues, so it is computed once per instruction and group (same value as
            // recomputing it every tick; the defaultdict insertions it makes are idempotent).
            let sr = match sr_cache[gi] {
                Some(v) => v,
                None => {
                    let mut sr = 0i64;
                    let mut any = false;
                    for &h in &r.src {
                        let i = touch(&mut ready[gi], &mut present[gi], &mut present_list[gi], h);
                        let mut v = ready[gi][i];
                        let reg = (h / 2) as usize;
                        if let Some((t0, narrow)) = wtime[gi][reg] {
                            if !rcache[gi].contains(&(reg as u32)) {
                                v = v.max(t0 + if narrow { w16 } else { w });
                            }
                        }
                        if !any || v > sr {
                            sr = v;
                            any = true;
                        }
                    }
                    sr_cache[gi] = Some(sr);
                    sr
                }
            };
            if sr > t {
                wake.push(sr);
                continue;
            }
            if r.kind == Kind::Wait {
                if let Some(m) = slots[gi][r.slot] {
                    if m > t {
                        wake.push(m);
                        continue;
                    }
                }
                if last_wait[gi] + wait_idle > t {
                    wake.push(last_wait[gi] + wait_idle);
                    continue;
                }
            }
            let home = gi % SCHEDULERS;
            let half = home % 2;
            let mut need = 0.0;
            // pass 0: the group's own scheduler; pass 1: a ring-neighbour scheduler that issued nothing
            let sch = if pass == 0 {
                home
            } else {
                if sched_free[home] <= t {
                    continue;
                }
                match [(home + 1) % SCHEDULERS, (home + SCHEDULERS - 1) % SCHEDULERS].into_iter().find(|&n| sched_free[n] <= t) {
                    Some(n) => n,
                    None => continue,
                }
            };
            if sched_free[sch] > t {
                wake.push(sched_free[sch]);
                continue;
            }
            if issue_tok < t_ {
                wake.push(t + 1);
                continue;
            }
            if r.unit.is_some() {
                let b = r.bucket;
                if tok[b].is_none() {
                    tok[b] = Some(buckets[b].init);
                    if buckets[b].init != buckets[b].capmax {
                        in_refill[b] = true;
                        refilling.push(b);
                    }
                }
                need = r.need_port;
                if need > 0.0 {
                    let fresh = r.src32.iter().filter(|&&reg| matches!(wtime[gi].get(reg as usize), Some(Some((t0, _))) if t - t0 <= fwd)).count();
                    need = (need - (fresh as i64 * t_) as f64).max(0.0);
                }
                if tok[b].unwrap() < r.cost || port[half] < need {
                    wake.push(t + 1);
                    continue;
                }
            }
            if r.ndst == 1 || r.ndst == 2 {
                // partial-write hazard: max over hazard registers of t0 + 2 cycles (blocks while > t);
                // per-group state only, cached like src_ready
                let h = match blk_cache[gi] {
                    Some(v) => v,
                    None => {
                        let wid = r.ndst as u8;
                        let mut h = i64::MIN;
                        for &reg in &r.dst_regs {
                            let lw = last_w[gi][reg as usize];
                            if lw != 0 && ((lw == 1 && wid == 2) || (lw == 2 && wid == 1)) {
                                let (t0, _) = wtime[gi][reg as usize].unwrap();
                                h = h.max(t0 + 2 * t_);
                            }
                        }
                        blk_cache[gi] = Some(h);
                        h
                    }
                };
                if h > t {
                    wake.push(h);
                    continue;
                }
            }
            if r.sfu_op {
                let l_ = r.lat;
                let mut gate = (sfu_iss[gi] + sfu_spacing).max(sfu_wb[gi] + sfu_wb_gap - l_).max(if r.sfu_slow.is_some() {
                    sfu_slow[gi]
                } else {
                    NEG_INF
                });
                if g > 1 {
                    gate = gate.max(core_wb[0] + core_wb_gap - l_).max(if r.core_slow.is_some() { core_slow[half] } else { core_fast[half] });
                }
                if gate > t {
                    wake.push(gate);
                    continue;
                }
            }
            if g > 1 && r.wide_cost.is_some() {
                let gate = core_fast[half].max(core_slow[half]);
                if gate > t {
                    wake.push(gate);
                    continue;
                }
            }
            // barrier: wait until every group of the scope has arrived (after draining its ALU work)
            if r.barrier != 0 {
                if bar_arrive[gi].is_none() {
                    bar_arrive[gi] = Some(t.max(alu_done[gi] + bar_drain));
                }
                let scope = if r.barrier == 1 { 0 } else { 1 + local[gi] / 4 };
                let key = (wg[gi], scope, bar_count[gi]);
                let rel = match bar_release.get(&key) {
                    Some(&x) => Some(x),
                    None => {
                        let mut latest = bar_arrive[gi].unwrap();
                        let mut all = true;
                        for j in 0..groups {
                            if wg[j] != wg[gi] || (r.barrier == 2 && local[j] / 4 != local[gi] / 4) || nxt[j] >= npcs[j] {
                                continue;
                            }
                            if bar_count[j] > bar_count[gi] {
                                continue;
                            }
                            match bar_arrive[j] {
                                Some(a) if bar_count[j] == bar_count[gi] => latest = latest.max(a),
                                _ => {
                                    all = false;
                                    break;
                                }
                            }
                        }
                        if all {
                            let x = latest + if r.barrier == 1 { bar68 } else { bar78 };
                            bar_release.insert(key, x);
                            Some(x)
                        } else {
                            None
                        }
                    }
                };
                match rel {
                    None => continue,
                    Some(x) if x > t => {
                        wake.push(x);
                        continue;
                    }
                    _ => {}
                }
            }
            // device memory: segments of this dynamic instruction, level of each, bandwidth tokens
            let mut mem_charge: Option<([f64; 5], i64, Vec<u64>)> = None;
            if let Some(m) = ms.as_mut() {
                let dev = !r.atomic && ((r.kind == Kind::Load && !r.tgop && !r.tg_name) || (r.kind == Kind::Store && !r.tg_name));
                if dev && !memv.is_empty() {
                    let mv = memv[gi];
                    while m.cur[gi] < mv.len() && (mv[m.cur[gi]].0 as usize) < nxt[gi] {
                        m.cur[gi] += 1;
                    }
                    if m.cur[gi] < mv.len() && mv[m.cur[gi]].0 as usize == nxt[gi] {
                        let segs = &mv[m.cur[gi]].1;
                        let store = r.kind == Kind::Store;
                        let mut need = [0.0; 5];
                        let n = segs.len() as f64;
                        if store {
                            need[1] = n.max(1.0).max(m_l1_store_min(mem));
                        } else {
                            need[0] = n.max(m_l1_load_min(mem));
                        }
                        let mut level = 0usize;
                        for &sg in segs {
                            let lv = if m.l1.contains(sg) { 0 } else if m.l2.contains(sg) { 1 } else if m.slc.contains(sg) { 2 } else { 3 };
                            if lv >= 1 {
                                need[1 + lv] += 1.0;
                            }
                            level = level.max(lv);
                        }
                        let mut until = 0i64;
                        for i in 2..5 {
                            let want = need[i].min(m.cap[i]);
                            if need[i] > 0.0 && m.tok[i] < want {
                                until = until.max(((want - m.tok[i]) / m.rate[i]).ceil() as i64);
                            }
                        }
                        if until > 0 {
                            wake.push(t + until.max(1));
                            continue;
                        }
                        mem_charge = Some((need, if store { 0 } else { m.extra[level] }, segs.clone()));
                    }
                }
            }
            // ---- issue at tick t
            issue_tok -= t_;
            let mut pipe_extra = 0i64;
            if let (Some(m), Some((_, mm))) = (ms.as_mut(), mem) {
                if (r.tg_name && r.barrier == 0) || r.atomic {
                    let mv = memv[gi];
                    while m.cur[gi] < mv.len() && (mv[m.cur[gi]].0 as usize) < nxt[gi] {
                        m.cur[gi] += 1;
                    }
                    if m.cur[gi] < mv.len() && mv[m.cur[gi]].0 as usize == nxt[gi] {
                        let words = &mv[m.cur[gi]].1;
                        let k = REAL_CYCLES_PER_UNIT;
                        let (occ, occ_min, free) = if r.atomic {
                            let mut ws = words.clone();
                            ws.sort_unstable();
                            let mut same = 0usize;
                            let mut run = 0usize;
                            for i in 0..ws.len() {
                                run = if i > 0 && ws[i] == ws[i - 1] { run + 1 } else { 1 };
                                same = same.max(run);
                            }
                            // 32-byte sectors; lanes on one word (a shared counter) serialise GPU-wide
                            let mut secs: Vec<u64> = ws.iter().map(|w| w >> 3).collect();
                            secs.dedup();
                            let contention = if same > 1 { same as f64 * mm.active_cores / mm.atomic_same_word_rate } else { 0.0 };
                            let c = (mm.atomic_min).max(mm.atomic_per_seg * secs.len() as f64).max(contention);
                            (tk(c * k), tk(mm.atomic_min * k), &mut m.at_free)
                        } else {
                            // bank conflicts: 32 banks of 4 bytes; lanes on the same word share one access
                            let mut per_bank: HashMap<u64, Vec<u64>> = HashMap::new();
                            for &w in words {
                                let e = per_bank.entry(w % 32).or_default();
                                if !e.contains(&w) {
                                    e.push(w);
                                }
                            }
                            let degree = per_bank.values().map(Vec::len).max().unwrap_or(1).max(1);
                            // the group's own threadgroup accesses are serial
                            let own = if r.kind == Kind::Store { mm.tg_group_store } else { mm.tg_group_load } + (degree - 1) as f64;
                            let gstart = t.max(m.tg_group_free[gi]);
                            m.tg_group_free[gi] = gstart + tk(own * k);
                            let gextra = m.tg_group_free[gi] - t - tk(own * k);
                            (tk(degree as f64 * k), tk(k) - gextra.max(0), &mut m.tg_free)
                        };
                        let start = t.max(*free);
                        *free = start + occ;
                        pipe_extra = (*free - t - occ_min).max(0);
                    }
                }
            }
            if let (Some(m), Some((need, _, segs))) = (ms.as_mut(), mem_charge.as_ref()) {
                for i in 2..5 {
                    m.tok[i] -= need[i].min(m.cap[i]);
                }
                // L1 pipeline: this instruction's segments pass after those already queued
                let k = if need[1] > 0.0 { 1 } else { 0 };
                let occ = (need[k] / m.rate[k]).round() as i64;
                let occ_min = ((if k == 1 { m_l1_store_min(mem) } else { m_l1_load_min(mem) }) / m.rate[k]).round() as i64;
                let start = t.max(m.l1_free[k]);
                m.l1_free[k] = start + occ;
                pipe_extra = (m.l1_free[k] - t - occ_min).max(0);
                for &sg in segs {
                    m.l1.touch(sg);
                    m.l2.touch(sg);
                    m.slc.touch(sg);
                }
            }
            let mem_extra = mem_charge.as_ref().map(|c| c.1).unwrap_or(0) + pipe_extra;
            if r.unit.is_some() {
                let b = r.bucket;
                tok[b] = Some(tok[b].unwrap() - r.cost);
                if !in_refill[b] && tok[b] != Some(buckets[b].capmax) {
                    in_refill[b] = true;
                    refilling.push(b);
                }
                port[half] -= need;
            }
            let mut branch = 0i64;
            let ii: i64;
            match r.kind {
                Kind::Alu => {
                    let lat = r.lat;
                    let mut iiv = r.ii;
                    if r.sfu_op {
                        sfu_iss[gi] = t;
                        sfu_wb[gi] = sfu_wb[gi].max(t + lat);
                        if let Some(s) = r.sfu_slow {
                            sfu_slow[gi] = t + s;
                        }
                        iiv = sfu_group_block;
                        if g > 1 {
                            core_wb[0] = core_wb[0].max(t + lat);
                            if let Some(s) = r.core_slow {
                                core_slow[half] = t + s;
                            } else {
                                core_fast[half] = t + r.core_fast_cost;
                            }
                        }
                    }
                    if g > 1 {
                        if let Some(c) = r.wide_cost {
                            core_fast[half] = t + c;
                        }
                    }
                    ii = iiv;
                    for &h in &r.dst {
                        let i = touch(&mut ready[gi], &mut present[gi], &mut present_list[gi], h);
                        ready[gi][i] = t + lat;
                    }
                    last_alu[gi] = t;
                    alu_done[gi] = alu_done[gi].max(t + lat);
                    alu_since_mem[gi] = true;
                    for pi in 0..pending[gi].len() {
                        let newdone = t + pending[gi][pi].lat;
                        if newdone > pending[gi][pi].done {
                            pending[gi][pi].done = newdone;
                            let ins = pending[gi][pi].ins;
                            for &h in &sim[ins].dst {
                                let i = touch(&mut ready[gi], &mut present[gi], &mut present_list[gi], h);
                                ready[gi][i] = ready[gi][i].max(newdone);
                            }
                            let s = pending[gi][pi].slot;
                            slots[gi][s] = Some(slots[gi][s].map_or(newdone, |m| m.max(newdone)));
                        }
                    }
                }
                Kind::Load => {
                    let tgop = r.tgop;
                    let mut delay = (last_alu[gi] + if tgop { quiet_tg } else { quiet_load } - t).max(0);
                    let lat;
                    let mut iiv;
                    if tgop {
                        lat = delay + tg_lat0 + mem_extra;
                        iiv = tg_issue0;
                    } else {
                        if alu_since_mem[gi] && !pending[gi].is_empty() {
                            // waits until the pending loads have passed the L1, not until they complete
                            // (ALU-interleaved loads over L2/DRAM windows: ~50 cycles apart, not a full latency)
                            let m = pending[gi].iter().map(|p| p.done - p.extra).max().unwrap();
                            delay = delay.max(m - t);
                        }
                        lat = (delay + load_fast + mem_extra).max(if inorder_loads { last_load_done[gi] + load_ii - t } else { 0 });
                        last_load_done[gi] = t + lat;
                        iiv = load_ii;
                    }
                    let cls: u8 = if tgop { 1 } else { 2 };
                    if cls == 1 && last_space[gi] == 3 {
                        iiv += space_switch;
                    }
                    last_space[gi] = cls;
                    alu_since_mem[gi] = false;
                    for &h in &r.dst {
                        let i = touch(&mut ready[gi], &mut present[gi], &mut present_list[gi], h);
                        ready[gi][i] = t + lat;
                    }
                    slots[gi][r.slot] = Some(slots[gi][r.slot].map_or(t + lat, |m| m.max(t + lat)));
                    let dfr = if tgop { tg_defer } else { defer_fast + if delay != 0 { defer_extra } else { 0 } };
                    pending[gi].push(Pending { slot: r.slot, done: t + lat, lat: delay + dfr, ins: ri, extra: if tgop { 0 } else { mem_extra } });
                    ii = iiv;
                }
                Kind::Store => {
                    let q = if r.tg_name { quiet_tg } else { quiet_dstore };
                    let mut delay = (last_alu[gi] + q - t).max(0);
                    if !r.tg_name {
                        delay = delay.max(last_idle_wait[gi] + store_after_wait - t);
                    }
                    let mut iiv = delay + r.mem_issue0;
                    let st_done = (t + iiv + r.mem_done.1).max(t + r.mem_issue0 + r.mem_done.0) + pipe_extra;
                    let cls: u8 = if r.tg_name { 1 } else { 3 };
                    if (cls == 1 && last_space[gi] == 3) || (cls == 3 && last_space[gi] == 1) {
                        iiv += space_switch;
                    }
                    last_space[gi] = cls;
                    slots[gi][r.slot] = Some(slots[gi][r.slot].map_or(st_done, |m| m.max(st_done)));
                    alu_since_mem[gi] = false;
                    ii = iiv;
                }
                Kind::TgAtomic => {
                    ii = t_;
                    for &h in &r.dst {
                        let i = touch(&mut ready[gi], &mut present[gi], &mut present_list[gi], h);
                        ready[gi][i] = t + load_latency + mem_extra;
                    }
                }
                Kind::Wait => {
                    let done_ = (t + t_).max(alu_done[gi] + wait_drain);
                    if slots[gi][r.slot].is_none() {
                        last_idle_wait[gi] = t;
                    }
                    ii = done_ - t;
                    last_wait[gi] = t;
                    slots[gi][r.slot] = None;
                    let s = r.slot;
                    pending[gi].retain(|p| p.slot != s);
                }
                Kind::Control => {
                    ii = ci;
                    if r.ctl_bump {
                        for &h in &present_list[gi] {
                            let i = h as usize;
                            let rr = ready[gi][i];
                            if rr + cl > t && crossed[gi][i] != rr {
                                ready[gi][i] = rr + cl;
                                crossed[gi][i] = rr + cl;
                            }
                        }
                    }
                    for &h in &r.dst {
                        let i = touch(&mut ready[gi], &mut present[gi], &mut present_list[gi], h);
                        ready[gi][i] = t + cl;
                    }
                    last_alu[gi] = t;
                    alu_done[gi] = alu_done[gi].max(t + cl);
                    if r.is_jmp_any && nxt[gi] + 1 < npcs[gi] && seqs_pcs[gi][nxt[gi] + 1] < r.pc {
                        branch = br;
                    }
                }
                Kind::Barrier => {
                    ii = t_;
                    if r.barrier != 0 {
                        bar_count[gi] += 1;
                        bar_arrive[gi] = None;
                    }
                }
            }
            if matches!(r.kind, Kind::Alu | Kind::Load) && !r.dst.is_empty() {
                for &reg in &r.dst_regs {
                    if let Some(pos) = rcache[gi].iter().position(|&x| x == reg) {
                        rcache[gi].remove(pos);
                    }
                    rcache[gi].push(reg);
                    wtime[gi][reg as usize] = Some((t, r.ndst == 1));
                    last_w[gi][reg as usize] = if r.ndst <= 2 { r.ndst as u8 } else { 4 };
                    if rcache[gi].len() > nc {
                        rcache[gi].remove(0);
                    }
                }
            }
            let occ = t_.max(ii) + branch;
            if r.barrier == 0 {
                sched_free[sch] = t + if r.kind == Kind::Alu && !r.reduce {
                    t_.max(ii)
                } else if r.exec_stack && g > 1 {
                    ctl_sched // if/else/push/pop/jmp_exec_none hold their scheduler (divergence dispatches)
                } else {
                    t_
                };
            } else {
                issue_tok += t_; // released groups pass the barrier together, without an issue slot
            }
            let e = &mut stats[ri];
            e.0 += 1;
            if nxt[gi] != 0 {
                e.1 += t - free[gi];
            }
            e.2 += occ;
            e.3 += branch;
            free[gi] = t + occ;
            last_t = last_t.max(free[gi]);
            nxt[gi] += 1;
            sr_cache[gi] = None;
            blk_cache[gi] = None;
            issued += 1;
            tick_issued[gi] = true;
            if nxt[gi] >= npcs[gi] {
                done += 1;
            }
        }
        }
        if done >= g {
            break;
        }
        let nt = if issued > 0 || wake.is_empty() { t + 1 } else { (t + 1).max(*wake.iter().min().unwrap()) };
        // refill(nt - t). Python min(a, b) is `b if b < a else a`; a bucket already at its cap stays
        // there (cap + non-negative refill, clamped), so only buckets below their cap are visited.
        let dt = (nt - t) as f64;
        if let Some(m) = ms.as_mut() {
            for i in 0..5 {
                m.tok[i] = (m.tok[i] + m.rate[i] * dt).min(m.cap[i]);
            }
        }
        for pt in port.iter_mut() {
            *pt = pymin(*pt + HALF_READ_PORTS * dt, (HALF_READ_PORTS + 2.0) * t_ as f64);
        }
        issue_tok = (issue_tok + ISSUE_PER_CORE * (nt - t)).min(ISSUE_PER_CORE * t_);
        let mut i = 0;
        while i < refilling.len() {
            let b = refilling[i];
            let bk = &buckets[b];
            let cur = tok[b].unwrap();
            let v = if bk.shared {
                pymin(cur + dt * TOKSCALE as f64, bk.capmax)
            } else {
                pymin(cur + bk.cap * dt * TOKSCALE as f64, bk.capmax)
            };
            tok[b] = Some(v);
            if v == bk.capmax {
                in_refill[b] = false;
                refilling.swap_remove(i);
            } else {
                i += 1;
            }
        }
        t = nt;
        if t > 50_000_000 * t_ {
            return Err("simulation limit".into());
        }
    }
    let total = last_t as f64 / t_ as f64;
    let st = if detail {
        Some(
            stats
                .into_iter()
                .enumerate()
                .filter(|(_, s)| s.0 > 0)
                .map(|(ri, (n, s, i, b))| {
                    let pc = sim[ri].pc;
                    (pc, PcStats { n, stall: s as f64 / t_ as f64, issue: i as f64 / t_ as f64, branch: b as f64 / t_ as f64 })
                })
                .collect(),
        )
    } else {
        None
    };
    Ok((total, st))
}

fn m_l1_store_min(mem: Option<(&[&[(u32, Vec<u64>)]], &MemModel)>) -> f64 {
    mem.map(|m| m.1.l1_store_min).unwrap_or(1.0)
}

fn m_l1_load_min(mem: Option<(&[&[(u32, Vec<u64>)]], &MemModel)>) -> f64 {
    mem.map(|m| m.1.l1_load_min).unwrap_or(1.0)
}
