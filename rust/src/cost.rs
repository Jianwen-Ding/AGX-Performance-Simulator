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
            rec.unmeasured = true;
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
    trace_impl(code, active, max_steps, false)
}

/// `trace` driven by the reference interpreter (`isa::exec::exec`) instead of the compiled one.
pub fn trace_reference(code: &[u8], active: usize, max_steps: usize) -> Result<Trace, String> {
    trace_impl(code, active, max_steps, true)
}

fn trace_impl(code: &[u8], active: usize, max_steps: usize, reference: bool) -> Result<Trace, String> {
    let mut st = exec::CoreState::new(128, Some(exec::Memory { base: TABLE_VA, words: vec![0; 65536] }));
    for t in 0..exec::SIMD_WIDTH {
        st.set_reg16(0, t, if t < active { 0 } else { 1 })?;
        st.set_reg32(1, t, t as i128)?;
        st.exec[t] = t < active;
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
        if info.check && (info.ureg || info.uses.iter().any(|&h| tainted[h as usize])) {
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
    mem_issue0: i64,
    mem_done: (i64, i64),
    is_jmp_any: bool,
    ctl_bump: bool,
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

/// agx_cost.simulate: `groups` identical SIMD groups on one core; returns real cycles (and per-pc stats).
pub fn simulate(instrs: &[Instr], pcs: &[usize], p: &Params, groups: usize, detail: bool)
    -> Result<(f64, Option<HashMap<usize, PcStats>>), String> {
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
            tg_name: is_tg,
            mem_issue0,
            mem_done,
            is_jmp_any: r.name == "jmp_exec_any",
            ctl_bump: r.src.iter().any(|&h| h != 0),
        });
    }
    let seq: Vec<usize> = pcs
        .iter()
        .map(|pc| by_pc.get(pc).copied().ok_or_else(|| format!("KeyError: pc {}", pc)))
        .collect::<Result<_, _>>()?;
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
    let mut last_space: Vec<u8> = vec![0; g]; // 0 None, 1 'tg', 2 'device_load', 3 'device_store'
    struct Pending {
        slot: usize,
        done: i64,
        lat: i64,
        ins: usize,
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
    let npcs = seq.len();
    while done < g {
        let mut issued = 0;
        order.clear();
        let k0 = if g == 1 { 0 } else { (((t / t_) as u64).wrapping_mul(2654435761) >> 7) as usize % g };
        order.extend((k0..g).chain(0..k0));
        wake.clear();
        for oi in 0..order.len() {
            let gi = order[oi];
            if nxt[gi] >= npcs {
                continue;
            }
            if free[gi] > t {
                wake.push(free[gi]);
                continue;
            }
            let ri = seq[nxt[gi]];
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
            let sch = gi % SCHEDULERS;
            let half = sch % 2;
            let mut need = 0.0;
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
            // ---- issue at tick t
            issue_tok -= t_;
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
                        lat = delay + tg_lat0;
                        iiv = tg_issue0;
                    } else {
                        if alu_since_mem[gi] && !pending[gi].is_empty() {
                            let m = pending[gi].iter().map(|p| p.done).max().unwrap();
                            delay = delay.max(m - t);
                        }
                        lat = delay + load_fast;
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
                    pending[gi].push(Pending { slot: r.slot, done: t + lat, lat: delay + dfr, ins: ri });
                    ii = iiv;
                }
                Kind::Store => {
                    let q = if r.tg_name { quiet_tg } else { quiet_dstore };
                    let mut delay = (last_alu[gi] + q - t).max(0);
                    if !r.tg_name {
                        delay = delay.max(last_idle_wait[gi] + store_after_wait - t);
                    }
                    let mut iiv = delay + r.mem_issue0;
                    let st_done = (t + iiv + r.mem_done.1).max(t + r.mem_issue0 + r.mem_done.0);
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
                        ready[gi][i] = t + load_latency;
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
                    if r.is_jmp_any && nxt[gi] + 1 < npcs && pcs[nxt[gi] + 1] < r.pc {
                        branch = br;
                    }
                }
                Kind::Barrier => {
                    ii = t_;
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
            sched_free[sch] = t + if r.kind == Kind::Alu && !r.reduce { t_.max(ii) } else { t_ };
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
            if nxt[gi] >= npcs {
                done += 1;
            }
        }
        if done >= g {
            break;
        }
        let nt = if issued > 0 || wake.is_empty() { t + 1 } else { (t + 1).max(*wake.iter().min().unwrap()) };
        // refill(nt - t). Python min(a, b) is `b if b < a else a`; a bucket already at its cap stays
        // there (cap + non-negative refill, clamped), so only buckets below their cap are visited.
        let dt = (nt - t) as f64;
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
