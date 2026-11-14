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
    ("fcmpsel", "fcmpsel"), ("mov_imm", "mov_imm"),
    // undecoded 6-byte no-ops issue exactly like a 6-byte mov_imm; 0x18 costs ~4.3 ns (~6 cycles) per use with no
    // bench row (playground sync-and-end-opcodes), so it is costed as mov_imm and flagged unmeasured
    ("nop6", "mov_imm"), ("fixed18", "mov_imm"), ("fadd32", "fadd32"), ("fmul32", "fmul32"), ("fmadd32", "fmadd32"),
    ("fadd16", "fadd16"), ("fmul16", "fmul16"), ("fmadd16", "fmadd16"), ("convert", "convert_f2i"), ("floor", "floor"),
    ("ceil", "floor"), ("trunc", "floor"), ("rint", "rint"), ("rcp", "rcp"), ("rsqrt", "rsqrt"),
    ("rsqrt_special", "rsqrt_special"), ("log2", "log2"), ("exp2", "exp2"), ("sin_pt_1", "sin_pt_1"),
    ("sin_pt_2", "sin_pt_2"), ("funop", "floor"), ("dfdx", "sin_pt_1"), ("dfdy", "sin_pt_1"),
    ("simd_shuffle", "simd_shuffle"), ("simd_shuffle_down", "simd_shuffle_down"), ("simd_shuffle_up", "simd_shuffle_up1"),
    ("simd_shuffle_xor", "simd_shuffle_xor"), ("simd_shuffle_rotate_up", "simd_shuffle_down"),
    ("simd_shuf_op", "simd_shuffle"), ("quad_shuffle", "quad_shuffle"), ("quad_shuffle_down", "quad_shuffle"),
    ("quad_shuffle_up", "quad_shuffle"), ("quad_shuffle_xor", "quad_shuffle"), ("quad_shuffle_rotate_up", "quad_shuffle"),
    ("icmp_ballot", "icmp_ballot"), ("fcmp_ballot", "fcmp_ballot"), ("icmp_quad_ballot", "quad_shuffle"),
    ("fcmp_quad_ballot", "quad_shuffle"), ("get_sr", "get_sr"), ("TODO.sr_thing", "get_sr"),
    // B18 / B15 (2026-10-21): the lane-offset fmadd (any k costs the same) and the 8x8 SIMD-group matrix fmadd
    ("fmadd32_lane", "fmadd32_lane1"), ("fmadd16_lane", "fmadd16_lane1"),
    ("simd_matrix_fmadd32", "simd_matrix_fmadd32"), ("simd_matrix_fmadd16", "simd_matrix_fmadd16"),
];
// (the rotate and the quad shuffle variants are measured: coverage/timing's simd_shuffle_rotate_up and
// quad_shuffle_{down,up,xor,rotate_up} _r14_r6_1 recordings, latency 4.85 / 3.90 cycles and SIMD issue 1.99 / 1.04
// per step, the same as the simd_shuffle_down and quad_shuffle rows they are costed as)
const UNMEASURED_ALU: &[&str] = &[
    "fixed18",
    "iunop", "ceil", "trunc", "funop", "dfdx", "dfdy", "simd_shuf_op", "TODO.sr_thing",
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
pub const CONTROL: &[&str] = &["call", "ret", "jmp_incomplete", 
    "if_icmp", "if_fcmp", "else_icmp", "else_fcmp", "while_icmp", "while_fcmp", "pop_exec", "push_exec", "jmp_exec_any",
    "jmp_exec_none", "stop", "trap", "TODO.unk51", "no_var", "end_group",
];
/// Measured cost of the seven Mesa-emitted memory barriers (second byte of the 2-byte `f5xx` encoding) in ns at the top GPU clock:
/// [back to back, after a coherent device store, after a coherent device load], playground novel-results/2026-10-18-fence-cost
/// (one SIMD group, unrolled chains with the clock keeper). The context is the last device store / load the group issued before the
/// fence (kept until a fence pays its penalty, see `fence_ticks`). Other `memory_barrier a, b, c` encodings are not measured and cost the median of the measured ones.
const FENCE_COSTS_NS: &[(u8, [f64; 3])] = &[
    (0x96, [85.0, 79.0, 79.0]),    // memory_barrier 1, 2, 9
    (0x33, [27.0, 13.0, 247.0]),   // unknown_barrier_1 (release)
    (0x9d, [86.0, 81.0, 306.0]),   // device_barrier_1 (acquire / invalidate)
    (0x9a, [39.0, 39.0, 40.0]),
    (0x9e, [99.0, 306.0, 94.0]),
    (0x03, [21.0, 29.0, 255.0]),
    (0x99, [91.0, 89.0, 310.0]),
];
const FENCE_UNMEASURED_NS: f64 = 85.0;
/// GPU cycles per ns at the top clock (max_khz 1,398,000)
const CYCLES_PER_NS: f64 = 1.398;
/// Cost in ns of a memory instruction issued with an empty execution mask: a load's (and its wait's) scoreboard
/// latency, a store's issue interval (B31 measurement, see `simulate_groups`).
const MASKED_DEVICE_LOAD_NS: f64 = 11.1;
const MASKED_TG_LOAD_NS: f64 = 12.1;
const MASKED_DEVICE_STORE_NS: f64 = 9.7;

/// Cost in ticks of a fence in a context (0 = none, 1 = after a store, 2 = after a load) and whether it consumed the context: the
/// refetch penalty after a load / store is paid once, by the first fence that has one (the seven-barrier sequence after a load costs
/// 646 ns measured, not 7 penalties), so the context survives fences that pay none.
fn fence_ticks(code: u8, ctx: usize) -> (i64, bool) {
    let row = FENCE_COSTS_NS.iter().find(|(c, _)| *c == code).map(|(_, v)| *v);
    let ns = row.map(|v| v[ctx]).unwrap_or(FENCE_UNMEASURED_NS);
    let consumed = row.map_or(true, |v| ctx != 0 && v[ctx] > v[0] + 100.0);
    (tk(ns * CYCLES_PER_NS), consumed)
}

/// AGX_MEM_LEVELS=1: device / stack accesses by deepest cache level (L1, L2, SLC, DRAM) since the start, by
/// (stack, store, level); `mem_levels_report` prints them
pub static MEM_LEVELS: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];
pub fn mem_levels_report() -> Option<String> {
    if !std::env::var("AGX_MEM_LEVELS").is_ok_and(|v| v == "1") {
        return None;
    }
    let v: Vec<u64> = MEM_LEVELS.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect();
    Some(format!("device loads {:?} stores {:?}; stack loads {:?} stores {:?} (L1, L2, SLC, DRAM)", &v[0..4], &v[4..8], &v[8..12], &v[12..16]))
}

/// Kernels with more code than this are taken to fetch instructions from beyond the instruction cache (see the taken
/// forward jump's scheduler hold).
const FETCH_BOUND_CODE_BYTES: usize = 16 << 10;
/// Instruction fetch lines: 64 bytes taken (the rate is what is measured, not the size).
const FETCH_LINE_SHIFT: usize = 6;

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
/// register writebacks (result-cache evictions) per real cycle per core: 2.0 per reference cycle
const WRITEBACK_PER_CYCLE: f64 = 2.0 / REAL_CYCLES_PER_UNIT;
const WRITEBACK_CAP: f64 = 4.0;
/// A 32-bit source written by the same SIMD group at most this many real cycles earlier is forwarded and
/// takes no register-file read port (one dependent chain per group reads fewer ports than eight).
const FORWARD_WINDOW: f64 = 4.0;
/// Per-half unit occupancy in real cycles (reference units at 32 groups x 1.049).
const HALF_UNIT_SLOW: &[(&str, f64)] = &[("rcp", 2.94 * 1.049), ("rsqrt", 3.92 * 1.049), ("rsqrt_special", 3.92 * 1.049)];
const HALF_UNIT_FAST: f64 = 1.96 * 1.049;
/// 64-bit add and 32x32->64 multiply-add occupy the same half unit (no sfu-pipe rules otherwise).
const HALF_UNIT_WIDE: &[(&str, f64)] = &[("iadd64", 3.92 * 1.049), ("imadd_wide", 3.92 * 1.049)];

/// Classes whose bench row reads three or more 32-bit registers and issues in 2 cycles; a form of them reading at
/// most two 32-bit registers issues in one (coverage/timing: fmadd32 with an immediate 1.06, icmpsel / fcmpsel
/// with immediates or 16-bit operands 1.06-1.17, vs 2.0-2.13).
const PORT_BOUND: &[&str] = &["fmadd32", "icmpsel", "fcmpsel"];
/// Rows derived from the instruction-bench coverage (coverage/timing/<form>.json, reference units):
/// (name, base row, latency, simd_issue or None to take the base row's, issue base row or None). The per-core
/// rate scales the base row's by the issue ratio.
const DERIVED_ROWS: &[(&str, &str, Option<f64>, Option<f64>, Option<&str>)] = &[
    // quad_min.* / quad_max.* / quad_xor (r14l, r6l forms)
    ("quad_minmax", "quad_iadd", Some(9.62), Some(7.71), None),
    // quad_prefix_fadd / fmul / iadd
    ("quad_prefix", "quad_iadd", Some(5.80), Some(3.90), None),
    // cmpsel reading at most two 32-bit registers: the select latency, a single-cycle issue (iadd's)
    ("icmpsel_2r", "icmpsel", None, None, Some("iadd")),
    ("fcmpsel_2r", "fcmpsel", None, None, Some("iadd")),
];

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
        for (name, base, lat, ii, ii_from) in DERIVED_ROWS {
            let Some(b0) = rows.get(*base).cloned() else { continue };
            let src = match ii_from {
                Some(n) => match rows.get(*n) { Some(r) => r.clone(), None => continue },
                None => b0.clone(),
            };
            let ii_real = ii.map(|x| (x * k).max(1.0)).unwrap_or(src.ii);
            let ipc = if ii_from.is_some() { src.ipc } else { b0.ipc * b0.ii / ii_real };
            rows.insert(name.to_string(), Row { lat: lat.map(|x| x * k).unwrap_or(b0.lat), ii: ii_real, ipc, tmpl: b0.tmpl.clone() });
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
    /// `memory_barrier`: its encoding byte (0x96 = f596, ...), costed from the measured fence table
    pub fence: Option<u8>,
    /// latency beyond the class row, in real cycles (operand-dependent forms: saturating fp32)
    pub lat_extra: f64,
    /// 32-bit registers read with the `.cache` hint (the `_cache` classes' extra issue cycle applies only when
    /// one of them is not in the group's result cache)
    pub cache_regs: Vec<u32>,
    /// register halves read one cycle after issue (the select operands of icmpsel / fcmpsel)
    pub late_src: Vec<u32>,
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

/// The 2-byte 0x78 barrier over aligned blocks of 4 SIMD groups (playground barrier finding), which applegpu
/// does not decode. Both second bytes seen on the hardware, `78 09` (the playground's stress tests) and `78 00`
/// (the coverage threadgroup cases, the 0x68 barrier's `68 00` pattern), synchronise alike (playground
/// sync-and-end-opcodes).
pub fn is_barrier78(code: &[u8], pc: usize) -> bool {
    pc + 1 < code.len() && code[pc] == 0x78 && matches!(code[pc + 1], 0x09 | 0x00)
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
            crate::func::mesa_check(code, pc, 2, "barrier_4groups")?;
            out.push(Instr {
                pc, size: 2, name: "barrier_4groups".into(), text: "0x78 barrier (aligned 4 SIMD groups)".into(), dst: vec![], src: vec![],
                reads32: 0, src32: vec![], unit: None, kind: Kind::Barrier, unmeasured: false, slot: 0, cache: false, uniform_src: false, fence: None,
                loop_bytes: None, wait_run: 0, lat_extra: 0.0, cache_regs: vec![], late_src: vec![],
            });
            pc += 2;
            continue;
        }
        let n0 = isa::opcode_to_number(&code[pc..code.len().min(pc + 16)]);
        // Padding between functions, in a kernel with real calls: executing it is still an error, in the trace.
        let Some(d) = isa::get_instruction_descriptor(n0) else {
            pc += 2;
            continue;
        };
        let size = d.decode_size(n0);
        if pc + size <= code.len() {
            crate::func::mesa_check(code, pc, size, d.name)?;
        }
        let n = if size >= 16 { n0 } else { n0 & ((1u128 << (8 * size)) - 1) };
        let ops = d.decode_operands(n)?;
        let mut name = d.name.to_string();
        let mut text = isa::disassemble_normalized(d, n)?;
        let f = d.decode_fields(n);
        let regops: Vec<(&Operand, Vec<u32>)> = ops.iter().map(|o| (o, regs_of(o))).filter(|(_, r)| !r.is_empty()).collect();
        let union = |it: &[(&Operand, Vec<u32>)]| set_of(it.iter().flat_map(|(_, r)| r.iter().copied()));
        let mut rec = Instr {
            pc, size, name: String::new(), text: String::new(), dst: vec![], src: vec![], reads32: 0, src32: vec![], unit: None,
            kind: Kind::Alu, unmeasured: false, slot: 0, cache: false, uniform_src: false, loop_bytes: None, wait_run: 0, fence: None,
            lat_extra: 0.0, cache_regs: vec![], late_src: vec![],
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
            if !["jmp_exec_any", "jmp_exec_none", "stop", "trap", "end_group"].contains(&nm) {
                src.push(0);
            }
            rec.src = set_of(src);
            rec.dst = if !matches!(nm, "stop" | "trap" | "end_group") { vec![0] } else { vec![] };
            rec.uniform_src = ops.iter().any(Operand::is_ureg);
        } else if BARRIERS.contains(&nm) {
            rec.kind = Kind::Barrier;
            rec.unmeasured = nm != "threadgroup_barrier";
            if nm == "memory_barrier" {
                let code = ((n >> 8) & 0xff) as u8;
                rec.fence = Some(code);
                rec.unmeasured = !FENCE_COSTS_NS.iter().any(|(c, _)| *c == code);
            }
        } else if lookup(ALU, nm).is_some() || is_reduce(nm) {
            rec.kind = Kind::Alu;
            if !regops.is_empty() {
                rec.dst = regops[0].1.clone();
                rec.src = union(&regops[1..]);
            }
            if nm == "get_sr" {
                rec.src = vec![];
            }
            // bitop with a constant truth table (0000 / 1111) reads neither source: latency-mode chains run at
            // the issue rate (coverage/timing bitop_0000 / bitop_1111: 1.04 vs 1.99 for every other table)
            let tt = (0..4).map(|i| fget(&f, &format!("tt{}", i)).unwrap_or(0)).collect::<Vec<_>>();
            let const_bitop = nm == "bitop" && (tt.iter().all(|&b| b == 0) || tt.iter().all(|&b| b == 1));
            if const_bitop {
                rec.src = vec![];
            }
            // Cross-half read: an instruction with two or more register sources that writes one 16-bit half and
            // reads the other half of the same register waits for the in-flight write of its destination half
            // (and/or/xor/nand/nor/xnor/fadd16.sat/fmul16.sat `rXl, rXh, r7l`: 1.99 per step with no data
            // dependency). One-source forms (`mov rXl, rXh`, transcendentals) and same-half forms
            // (half.json 'f16 low/high independent') do not wait.
            if !const_bitop && regops.len() >= 3 && regops[0].1.len() == 1 && !is_reduce(nm) {
                let hd = regops[0].1[0];
                if regops[1..].iter().any(|(o, r)| matches!(o, Operand::Reg16(..)) && r.len() == 1 && r[0] / 2 == hd / 2 && r[0] != hd) {
                    rec.src = set_of(rec.src.iter().copied().chain([hd]));
                }
            }
            let wide = set_of(
                regops
                    .iter()
                    .skip(1)
                    .filter(|(o, _)| matches!(o, Operand::Reg32(..) | Operand::Reg64(..)))
                    .flat_map(|(_, r)| r.iter().map(|h| h / 2)),
            );
            let wide = if const_bitop { vec![] } else { wide };
            // icmpsel / fcmpsel read 16-bit select operands a cycle after the compared pair: a select fed by the
            // previous instruction does not wait (16-bit forms whose select operands are other chains' results:
            // 1.06-1.17 per step with 8 chains, as if only the compared operands were dependencies). 32-bit select
            // operands are not covered by the bench (constants there) and keep the ordinary dependency.
            if matches!(nm, "icmpsel" | "fcmpsel") {
                let vals: Vec<&Operand> = ops.iter().filter(|o| matches!(o, Operand::Reg16(..) | Operand::Reg32(..) | Operand::Reg64(..)
                    | Operand::UReg16(..) | Operand::UReg32(..) | Operand::UReg64(..) | Operand::Imm(..))).collect();
                if vals.len() == 5 {
                    let early: Vec<u32> = vals[1..3].iter().flat_map(|o| regs_of(o)).collect();
                    rec.late_src = set_of(vals[3..].iter().filter(|o| matches!(o, Operand::Reg16(..))).flat_map(|o| regs_of(o))
                        .filter(|h| !early.contains(h)));
                }
            }
            rec.reads32 = wide.len();
            rec.src32 = wide.clone();
            rec.cache = regops.iter().skip(1).any(|(o, _)| o.flags() & isa::operand::F_CACHE != 0);
            rec.cache_regs = set_of(
                regops
                    .iter()
                    .skip(1)
                    .filter(|(o, _)| o.flags() & isa::operand::F_CACHE != 0)
                    .flat_map(|(_, r)| r.iter().map(|h| h / 2)),
            );
            // Saturating fp32 forms take one more cycle (fadd32/fmul32/fmadd32 .sat: 2.94 vs 1.99), and one more
            // again when the result is narrowed to a 16-bit register (fadd32/fmul32.sat rXl: 3.90). 16-bit and
            // integer .sat forms cost nothing extra.
            if matches!(nm, "fadd32" | "fmul32" | "fmadd32") && fget(&f, "S") == Some(1) {
                rec.lat_extra = 1.0 + if regops.first().map(|(_, r)| r.len() == 1).unwrap_or(false) { 1.0 } else { 0.0 };
            }
            let imm = ops.iter().filter(|o| matches!(o, Operand::Imm(..))).count();
            let key = if is_reduce(nm) {
                // Reduction classes (coverage/timing, any source width): min / max / xor reduce faster than
                // and / or / iadd / fadd / fmul (quad 9.6 vs 11.5, simd 22.0 vs 22.9), and quad prefix scans take
                // half a quad reduction (5.8); simd prefix scans all match simd_prefix_iadd.
                let op = nm.trim_start_matches("simd_").trim_start_matches("quad_");
                let fast = op.starts_with("min") || op.starts_with("max") || op.starts_with("xor");
                let k = if p.row(nm).is_some() {
                    nm.to_string()
                } else if nm.starts_with("quad_prefix") {
                    "quad_prefix".into()
                } else if nm.starts_with("quad") {
                    if fast { "quad_minmax".into() } else { "quad_iadd".into() }
                } else if nm.contains("prefix") {
                    "simd_prefix_iadd".into()
                } else if fast {
                    "simd_min.u".into()
                } else {
                    "simd_iadd".into()
                };
                rec.unmeasured = p.row(&k).is_none();
                if rec.unmeasured {
                    // derived rows missing (custom bench file): fall back to the measured base classes
                    if nm.starts_with("quad") { "quad_iadd".into() } else if nm.contains("prefix") { "simd_prefix_iadd".into() } else { "simd_iadd".into() }
                } else {
                    k
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
                let short_fp = matches!(nm, "fadd32" | "fmul32" | "fadd16" | "fmul16");
                if short_fp {
                    // fp add/mul: the 4-byte short-immediate encoding issues in one cycle and completes one
                    // tenth of a unit sooner (fadd32_imm row); longer encodings, immediate or not, are the base class
                    if size == 4 && p.row("fadd32_imm").is_some() {
                        key = "fadd32_imm".into();
                    } else if rec.cache && p.row(&format!("{}_cache", key)).is_some() {
                        key += "_cache";
                    }
                } else if PORT_BOUND.contains(&key.as_str()) && rec.reads32 <= 2 && !rec.cache
                    && p.row(&format!("{}_2r", key)).or_else(|| p.row(&format!("{}_imm", key))).is_some() {
                    // the class row was measured reading three or more 32-bit registers (2 issue cycles); with at
                    // most two (immediates, 16-bit operands) the form issues in one
                    key = if p.row(&format!("{}_2r", key)).is_some() { format!("{}_2r", key) } else { format!("{}_imm", key) };
                } else if imm > 0 && rec.reads32 <= 1 && p.row(&format!("{}_imm", key)).is_some() {
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
    /// Legacy cost inputs: load/store 64-byte segments, atomic/TG word addresses.
    pub mem: Vec<(u32, Vec<u64>)>,
    /// Functional device footprint: distinct 64-byte segments including the
    /// complete byte range of each 32/64-bit atomic. No threadgroup addresses.
    /// Cost-emulated traces may leave this empty.
    pub device_segments: Vec<(u32, Vec<u64>)>,
    /// Steps (indices into `pcs`, ascending) issued while the group's execution mask was empty: no lane active, as
    /// for an if body the code does not jump over. The scheduler charges the measured masked costs to them
    /// (`simulate_groups`); empty for most traces.
    pub masked: Vec<u32>,
}

/// Where one SIMD group sits in a dispatch: enough to answer `get_sr` for the geometry registers
/// (playground experiments/registers/sr-geometry-core-alt.md) and to give r1 the global X index (the
/// direct-DRM harness prefix convention). Threads are linearised x fastest; SIMD group `group` holds
/// linear local indices 32*group .. 32*group+31.
/// Launch resources and state that special registers expose (playground sr-resources-width, sr-direct-drm):
/// SR18 threadgroup bytes, SR19 register allocation, SR21 context, SR24 program kind, SR32 pipeline-word offset.
/// `regs` 0, `pipeline` / `context` None: not known to the caller (reading the register is then an error).
#[derive(Clone, Copy, Debug, Default)]
pub struct SrLaunch {
    /// threadgroup memory allocated per workgroup (bytes)
    pub tg_bytes: u32,
    /// register allocation per thread (16-bit registers)
    pub regs: u32,
    /// USC-relative offset of the dispatch's pipeline words (CDM launch word 1)
    pub pipeline: Option<u32>,
    pub context: Option<u32>,
    /// the group runs a preamble (preshader): one lane, slot 96, SR24 = 5
    pub preamble: bool,
}

#[derive(Clone, Debug)]
pub struct GroupCtx {
    pub wg_size: [u32; 3],
    pub wg_id: [u32; 3],
    pub group: u32,
    pub core: u32,
    pub slot: u32,
    /// 32-bit uniform registers preset (index, value), e.g. buffer addresses in u8_u9
    pub uniforms: Vec<(usize, u32)>,
    pub launch: SrLaunch,
}

impl GroupCtx {
    pub fn threads(&self) -> u32 {
        self.wg_size[0] * self.wg_size[1] * self.wg_size[2]
    }
    /// active lanes of this group (a partial last group has fewer)
    pub fn lanes(&self) -> usize {
        (self.threads().saturating_sub(32 * self.group)).min(32) as usize
    }
    fn coordinate(&self, axis: usize, i: u32) -> u32 {
        let [x, y, _] = self.wg_size;
        match axis { 0 => i % x, 1 => (i / x) % y, 2 => i / (x * y), _ => unreachable!() }
    }
    /// Special register `sr` for lane `t`, or None when not modelled.
    pub fn sr(&self, sr: u32, t: usize) -> Option<u32> {
        let l = self.threads();
        let i = 32 * self.group + t as u32;
        Some(match sr {
            0..=2 => self.wg_id[sr as usize],
            4..=6 => self.wg_size[sr as usize - 4],
            8..=10 => self.wg_size[sr as usize - 8],
            7 => (l + 31) / 32,
            15 => 16,
            17 => self.slot,
            20 => self.core,
            18 => self.launch.tg_bytes.div_ceil(256) * 256,
            19 => if self.launch.regs > 0 { self.launch.regs.div_ceil(8).max(1) * 8 } else { return None },
            21 => self.launch.context?,
            24 => if self.launch.preamble { 5 } else { 2 },
            25 => 2,
            32 => self.launch.pipeline?,
            // constant in compute under every launch parameter varied so far (meaning unknown)
            97 => 2048,
            114 => 136,
            48..=50 => self.coordinate(sr as usize - 48, i),
            51 => i,
            52 | 58 => i % 32,
            53 => i / 32,
            56 => i % 4,
            57 => 4.min(l - 4 * (i / 4)),
            59 => 32.min(l - 32 * (i / 32)),
            80..=82 => self.wg_id[sr as usize - 80] * self.wg_size[sr as usize - 80] + self.coordinate(sr as usize - 80, i),
            // constant in compute under all 10 recorded geometries (coverage/expected/geometry) and under divergence:
            // 12 / 13 imageblock width / height (no imageblock), 60 = 15 (Mesa: coverage_mask), 63 = 1 (Mesa:
            // is_active_thread)
            12 | 13 | 54 | 55 | 61 | 62 | 124 | 144 | 146 | 147 => 0,
            60 => 15,
            63 => 1,
            // physical position in the quad: unlike SR56 it ignores the execution mask (playground sr-divergence)
            16 => i % 4,
            // 0 / 8: a global firmware power-management state bit, set on a 56 ms cycle while the GPU is busy and
            // cleared after ~2 ms of idle; it can change mid-dispatch (playground novel-results/2026-10-16-special-registers)
            113 => return None,
            // every other selector read 0 in all 121 executing main-shader scans (playground sr-direct-drm) and under
            // divergence
            0..=255 => 0,
            _ => return None,
        })
    }
}

/// Threadgroup memory base address (bytes) of lane t, as applegpu decodes the base operand: At = 0 a 16-bit
/// register, 2 zero, 1 / 3 a 16-bit uniform (A | (At >> 1) << 8; Mesa keeps constant offsets such as part[1] there).
pub fn tg_base(st: &exec::CoreState, at: u64, a: u64, t: usize) -> Result<usize, String> {
    Ok(match at {
        0 => st.get_reg16(a as usize, t)? as usize,
        2 => 0,
        _ => {
            let u = (a | (at >> 1) << 8) as usize;
            *st.uniforms.get(u).ok_or_else(|| format!("threadgroup base: uniform half {} out of range", u))? as usize
        }
    })
}

/// Word addresses (byte / 4) of a threadgroup load / store / atomic for every active lane and component:
/// byte = base + 4 * size * index + size * component (playground threadgroup-addressing finding).
pub fn tg_words(f: &isa::Fields, st: &exec::CoreState) -> Option<Vec<u64>> {
    let g = |k: &str| fget(f, k);
    let item: i128 = match g("F") {
        Some(1 | 3 | 6 | 7) => 2,
        Some(2) | None => 4,
        _ => 1,
    };
    let mask = g("mask").unwrap_or(1);
    let mut w = Vec::with_capacity(4 * exec::SIMD_WIDTH);
    // field lookups once per instruction (applied with `?` per lane as before: a missing field matters only
    // when a lane is active)
    let (f_at, f_a, f_ot, f_o) = (g("At"), g("A"), g("Ot"), g("O"));
    for t in 0..exec::SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let base = tg_base(st, f_at?, f_a?, t).ok()? as i128;
        let idx = if f_ot? != 0 { f_o? as i128 } else { st.get_reg16(f_o? as usize, t).ok()? };
        for c in 0..4 {
            if mask & (1 << c) != 0 {
                w.push(((base + 4 * item * idx + item * c) / 4) as u64);
            }
        }
    }
    Some(w)
}

/// Word addresses of a device `atomic` (32-bit) for every active lane.
pub fn atomic_words(f: &isa::Fields, st: &exec::CoreState) -> Option<Vec<u64>> {
    let g = |k: &str| fget(f, k);
    let mut w = Vec::with_capacity(exec::SIMD_WIDTH);
    let (f_at, f_a, f_ot, f_o, f_ou) = (g("At"), g("A"), g("Ot"), g("O"), g("Ou"));
    for t in 0..exec::SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let base = if f_at? != 0 { st.ureg64((f_a? >> 1) as usize).ok()? } else { st.get_reg64((f_a? >> 1) as usize, t).ok()? };
        let off: i128 = if f_ot? != 0 {
            isa::desc::sign_extend(f_o? as i128, 16)
        } else {
            let o = st.get_reg32((f_o? >> 1) as usize, t).ok()?;
            if f_ou? == 0 { isa::desc::sign_extend(o, 32) } else { o }
        };
        w.push(((base + 4 * off) >> 2) as u64);
    }
    Some(w)
}

/// Per-lane byte addresses of a device_load / device_store (same formula as the emulator's
/// device_load), for the active lanes; None if an operand cannot be read.
pub fn mem_segments(f: &isa::Fields, st: &exec::CoreState) -> Option<Vec<u64>> {
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
    // field lookups once per instruction (applied with `?` per lane as before: a missing field matters only
    // when a lane is active)
    let (f_at, f_a, f_ot, f_o, f_ou) = (g("At"), g("A"), g("Ot"), g("O"), g("Ou"));
    for t in 0..exec::SIMD_WIDTH {
        if !st.exec[t] {
            continue;
        }
        let base = if f_at? != 0 { st.ureg64((f_a? >> 1) as usize).ok()? } else { st.get_reg64((f_a? >> 1) as usize, t).ok()? };
        let mut off: i128 = if f_ot? != 0 {
            isa::desc::sign_extend(f_o? as i128, 16)
        } else {
            let o = st.get_reg32((f_o? >> 1) as usize, t).ok()?;
            if f_ou? == 0 { isa::desc::sign_extend(o, 32) } else { o }
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
    access: Option<Box<crate::memtrace::Access>>,
    sr: Option<SrRead>,
}

/// Immutable get_sr operands only; values and execution masks are never cached.
struct SrRead { selector: u32, destination: usize, half: bool }

impl SrRead {
    fn new(name: &str, fields: &isa::Fields, ops: &[Operand]) -> Option<Self> {
        if name != "get_sr" { return None; }
        let (destination, half) = match ops.first()? {
            Operand::Reg32(r, _) => (*r as usize, false),
            Operand::Reg16(r, _) => (*r as usize, true),
            _ => return None,
        };
        Some(Self { selector: fget(fields, "SR").unwrap_or(0) as u32, destination, half })
    }

    /// false preserves the approximate interpreter's skip/taint fallback for
    /// unknown state (notably SR113 and absent launch metadata).
    fn execute(&self, ctx: &GroupCtx, st: &mut exec::CoreState) -> Result<bool, String> {
        let sr = self.selector;
        let uniform = !matches!(sr, 16 | 48..=52 | 56..=59 | 80..=82);
        let value = if uniform {
            match ctx.sr(sr, 0) { Some(v) => v, None => return Ok(false) }
        } else { 0 };
        if uniform || sr == 59 {
            let value = if sr == 59 { st.exec.iter().filter(|&&v| v).count() as u32 } else { value };
            for t in 0..exec::SIMD_WIDTH {
                if st.exec[t] { self.write(st, t, value)?; }
            }
            return Ok(true);
        }
        // Axis and global origin are uniform too: don't redispatch the full SR
        // table (or recompute the origin) in each coordinate lane.
        let coordinate = match sr {
            48..=50 => Some((sr as usize - 48, false)),
            80..=82 => Some((sr as usize - 80, true)),
            _ => None,
        };
        if let Some((axis, global)) = coordinate {
            let origin = if global { ctx.wg_id[axis] * ctx.wg_size[axis] } else { 0 };
            let first = 32 * ctx.group;
            for t in 0..exec::SIMD_WIDTH {
                if st.exec[t] { self.write(st, t, origin + ctx.coordinate(axis, first + t as u32))?; }
            }
            return Ok(true);
        }
        let counts: [u32; 8] = if sr == 57 {
            std::array::from_fn(|q| st.exec[4*q..4*q+4].iter().filter(|&&v| v).count() as u32)
        } else { [0; 8] };
        let mut ranks = [0u32; 8];
        for t in 0..exec::SIMD_WIDTH {
            if !st.exec[t] { continue; }
            let value = match sr {
                56 | 58 => {
                    let q = if sr == 56 { t / 4 } else { 0 };
                    let rank = ranks[q]; ranks[q] += 1; rank
                }
                57 => counts[t / 4],
                _ => ctx.sr(sr, t).expect("known lane selector"),
            };
            self.write(st, t, value)?;
        }
        Ok(true)
    }

    fn write(&self, st: &mut exec::CoreState, t: usize, value: u32) -> Result<(), String> {
        if self.half { st.set_reg16(self.destination, t, value as i128) }
        else { st.set_reg32(self.destination, t, value as i128) }
    }
}

#[cfg(test)]
mod sr_tests {
    use super::*;

    fn context() -> GroupCtx {
        GroupCtx { wg_size: [8, 4, 2], wg_id: [3, 2, 1], group: 1, core: 19, slot: 7, uniforms: vec![],
            launch: SrLaunch { tg_bytes: 513, regs: 65, context: Some(47), pipeline: Some(0x1080), preamble: false } }
    }

    #[test]
    fn cost_sr_reads_match_independent_mask_oracles_and_preserve_halves() {
        for sr in (0..256).filter(|&sr| sr != 113) {
            for half in [false, true] {
                let read = SrRead { selector: sr, destination: if half { 8 } else { 4 }, half };
                for mask in [0u32, u32::MAX, 0x55555555, 0x80808080, 0x00002001, 0xffff0000, 0x00001fff, 0x80000000] {
                    let ctx = context();
                    let mut st = exec::CoreState::new(8, None);
                    for t in 0..32 {
                        st.exec[t] = mask & (1 << t) != 0;
                        st.set_reg32(4, t, 0xaabbccdd).unwrap(); st.set_reg32(5, t, 0x12345678).unwrap();
                    }
                    assert!(read.execute(&ctx, &mut st).unwrap());
                    for t in 0..32 {
                        let mut want = 0xaabbccdd;
                        if mask & (1 << t) != 0 {
                            let q = t / 4 * 4;
                            let v = match sr {
                                56 => (q..t).filter(|&i| mask & (1 << i) != 0).count() as u32,
                                57 => (q..q+4).filter(|&i| mask & (1 << i) != 0).count() as u32,
                                58 => (0..t).filter(|&i| mask & (1 << i) != 0).count() as u32,
                                59 => mask.count_ones(),
                                _ => ctx.sr(sr, t).unwrap(),
                            };
                            want = if half { 0xaabb0000 | (v & 0xffff) } else { v };
                        }
                        assert_eq!(st.get_reg32(4, t).unwrap() as u32, want, "sr={sr} half={half} mask={mask:x} lane={t}");
                        assert_eq!(st.get_reg32(5, t).unwrap(), 0x12345678);
                    }
                }
            }
        }
    }

    #[test]
    fn coordinate_fast_path_matches_cartesian_geometry_on_every_axis() {
        for size in [[1, 1, 1], [6, 7, 1], [4, 3, 5], [8, 4, 2], [1, 16, 4]] {
            let mut linear = 0;
            for z in 0..size[2] { for y in 0..size[1] { for x in 0..size[0] {
                let mut ctx = context(); ctx.wg_size = size; ctx.group = linear / 32;
                let t = (linear % 32) as usize;
                for axis in 0..3 {
                    for global in [false, true] {
                        let mut st = exec::CoreState::new(8, None); st.exec = [false; 32]; st.exec[t] = true;
                        let read = SrRead { selector: if global { 80 } else { 48 } + axis as u32, destination: 4, half: false };
                        assert!(read.execute(&ctx, &mut st).unwrap());
                        let expected = [x, y, z][axis] + if global { ctx.wg_id[axis] * size[axis] } else { 0 };
                        assert_eq!(st.get_reg32(4, t).unwrap(), expected as i128);
                    }
                }
                linear += 1;
            } } }
        }
    }

    #[test]
    fn cached_sr_operands_keep_unknown_fallback_and_resolve_launch_each_time() {
        let mut ctx = context();
        let mut st = exec::CoreState::new(8, None);
        let read = SrRead { selector: 21, destination: 4, half: false };
        for context in [Some(47), Some(91), None] {
            ctx.launch.context = context;
            assert_eq!(read.execute(&ctx, &mut st).unwrap(), context.is_some());
            assert_eq!(st.get_reg32(4, 0).unwrap(), context.unwrap_or(91) as i128);
        }
        assert!(!SrRead { selector: 113, destination: 4, half: false }.execute(&ctx, &mut st).unwrap());
        let mut code = crate::kernels::get_sr(7, 4);
        code[1] |= 2; // odd encoded R aliases r4; it is not a 64-bit write
        let n = isa::opcode_to_number(&code);
        let d = isa::get_instruction_descriptor(n).unwrap(); let f = d.decode_fields(n);
        let args = SrRead::new(d.name, &f, &d.fields_to_operands(&f).unwrap()).unwrap();
        assert!(!args.half); assert_eq!(args.destination, 4);
        st.set_reg32(5, 0, 0x12345678).unwrap(); args.execute(&ctx, &mut st).unwrap();
        assert_eq!(st.get_reg32(5, 0).unwrap(), 0x12345678);
        assert!(SrRead { selector: 7, destination: 100, half: false }.execute(&ctx, &mut st).is_err());
    }
}

/// agx_cost.trace: dynamic pc sequence for one SIMD group (applegpu emulator; uniforms and memory
/// read as zero; instructions it cannot execute are skipped and taint their destination).
pub fn trace(code: &[u8], active: usize, max_steps: usize) -> Result<Trace, String> {
    PreparedTrace::new(code).trace_impl(active, max_steps, false, None)
}

/// `trace` for one SIMD group of a dispatch: `get_sr` returns the group's geometry registers, r1 holds
/// the global X index, uniforms are preset, and exec starts with the group's real lanes active.
pub fn trace_group(code: &[u8], ctx: &GroupCtx, max_steps: usize) -> Result<Trace, String> {
    PreparedTrace::new(code).trace_group(ctx, max_steps)
}

/// `trace` driven by the reference interpreter (`isa::exec::exec`) instead of the compiled one.
pub fn trace_reference(code: &[u8], active: usize, max_steps: usize) -> Result<Trace, String> {
    PreparedTrace::new(code).trace_impl(active, max_steps, true, None)
}

/// Shared lazy instruction metadata for the cost model's approximate tracer.
/// Every invocation still starts with independent registers, taint and zero memory.
/// This is not the real-memory functional executor (`func::PreparedDispatch`).
pub struct PreparedTrace<'a> {
    code: &'a [u8],
    cache: crate::decode_cache::DecodeCache<PcInfo, 2>,
}

impl<'a> PreparedTrace<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        Self { code, cache: crate::decode_cache::DecodeCache::new(code.len()) }
    }

    pub fn trace_group(&self, ctx: &GroupCtx, max_steps: usize) -> Result<Trace, String> {
        self.trace_impl(ctx.lanes(), max_steps, false, Some(ctx))
    }

    fn decode_at(&self, pc: usize, nregs: usize) -> Result<&PcInfo, String> {
        let cell = self.cache.at(pc);
        if cell.get().is_none() {
            let code = self.code;
            let n0 = isa::opcode_to_number(&code[pc..code.len().min(pc + 16)]);
            let desc = isa::get_instruction_descriptor(n0).ok_or_else(|| format!("trace: undecodable instruction at {}", pc))?;
            let size = desc.decode_size(n0);
            if size > code.len() - pc {
                return Err(format!("trace: truncated {} at {}: needs {} bytes, has {}", desc.name, pc, size, code.len() - pc));
            }
            let n = desc.mask_instr(n0);
            let fields = desc.decode_fields(n);
            let stop = matches!(desc.name, "stop" | "trap" | "end_group");
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
            let op = isa::compiled::Compiled::new(desc, &fields, &ops, nregs);
            // Preserve the approximate tracer's existing classification (no .TODO accesses).
            let access = crate::memtrace::Access::decode(desc.name, &fields, false);
            // Concurrent misses can decode independently; only the winning immutable value is retained.
            let sr = SrRead::new(desc.name, &fields, &ops);
            let _ = cell.set(Box::new(PcInfo { desc, size, fields, ops, uses, dst, op, stop, jump, off, check, ureg, access, sr }));
        }
        Ok(cell.get().unwrap())
    }

    fn trace_impl(&self, active: usize, max_steps: usize, reference: bool, ctx: Option<&GroupCtx>) -> Result<Trace, String> {
        let code = self.code;
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
                if u >= st.uniforms.len() / 2 {
                    return Err(format!("trace: uniform register {} out of range", u));
                }
                st.uniforms[2 * u] = (v & 0xffff) as u16;
                st.uniforms[2 * u + 1] = (v >> 16) as u16;
            }
        }
        let mut out = Trace::default();
        let mut tainted = vec![false; 1024];
        let mut pc: i64 = 0;
        while pc < code.len() as i64 {
            if pc < 0 {
                return Err(format!("trace: negative pc {}", pc));
            }
            let upc = pc as usize;
            if upc % 2 != 0 {
                return Err(format!("trace: unaligned instruction address {}", upc));
            }
            // Every executed instruction consumes the budget, including the raw 0x78 barrier.
            if out.pcs.len() >= max_steps {
                return Err("trace step limit (non-terminating loop?)".into());
            }
            if !st.exec.iter().any(|&e| e) {
                out.masked.push(out.pcs.len() as u32);
            }
            if is_barrier78(code, upc) {
                out.pcs.push(upc);
                pc += 2;
                continue;
            }
            let info = self.decode_at(upc, st.reg16s.len())?;
            out.pcs.push(upc);
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
                if take {
                    let target = pc as i128 + info.off as i128;
                    if target < 0 || target >= code.len() as i128 || target % 2 != 0 {
                        return Err(format!("trace: jump at {} targets {} outside the program or unaligned", upc, target));
                    }
                    pc = target as i64;
                } else {
                    pc += info.size as i64;
                }
                continue;
            }
            if let Some(w) = info.access.as_ref().and_then(|a| a.capture(&st)) {
                out.mem.push(((out.pcs.len() - 1) as u32, w));
            }
            if let (Some(c), Some(sr)) = (ctx, &info.sr) {
                if sr.execute(c, &mut st)? {
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
            pc += info.size as i64;
        }
        Ok(out)
    }
}

/// agx_cost.fit_branch: taken-branch penalty so an empty counted-loop trip costs the measured 35.25 units.
pub fn fit_branch(p: &mut Params) -> Result<f64, String> {
    p.branch = Some(0.0);
    let mut c = [0.0; 2];
    for (i, t) in [200u32, 1000].into_iter().enumerate() {
        let code = crate::kernels::counted(&[], t, true, crate::kernels::DEFAULT_CONSTS);
        let ins = decode(&code, p)?;
        let tr = trace(&code, 32, 500_000)?;
        c[i] = simulate_trace(&ins, &tr, p, 1, false)?.0;
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
    shuffle: bool,
    fp32: bool,
    lat: i64,
    ii: i64,
    /// issue interval when every `.cache` source is in the result cache (`_cache` classes only)
    ii_cache_hit: Option<i64>,
    cache_regs: Vec<u32>,
    late_src: Vec<u32>,
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
    /// `memory_barrier` encoding byte
    fence: Option<u8>,
    /// device_load / device_store (the contexts that change a fence's cost)
    dev_ctx: u8,
    mem_issue0: i64,
    mem_done: (i64, i64),
    is_jmp_any: bool,
    is_jmp: bool,
    /// call / ret: a taken jump, to another function
    is_call: bool,
    stack_store: bool,
    /// stack_load / stack_store (AGX_MEM_LEVELS)
    stack: bool,
    size: usize,
    ctl_bump: bool,
    /// exec-mask instructions other than the loop's while/jmp_exec_any
    exec_stack: bool,
    /// cost class when issued with an empty execution mask: 0 = as active (control, wait, barriers, atomics and the
    /// unmeasured memory instructions), 1 = ALU, 2 = device load, 3 = device store, 4 = threadgroup load
    masked_cls: u8,
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
    /// Instruction fetch: ns per line for the first group of a core to reach it (0: not modelled, steady-state loops).
    pub fetch_line_ns: f64,
    /// Touched code of at most this many bytes per launch is taken as already in the instruction cache (0: none).
    pub icache_bytes: usize,
    /// Instruction cache with refetch (0: off, each line fetched once per dispatch at `fetch_line_ns`): the core's
    /// lines (12 KiB of 64-byte lines, fewer when instructions cross 8-byte boundaries; playground
    /// novel-results/2026-10-20-icache), least recently used evicted. A miss reached by falling through from the previous
    /// line costs `fetch_line_ns`, one reached by a taken jump, call or return `fetch_jump_ns` (playground
    /// novel-results/2026-11-07-uncached-jump-fetch); the group waits for it, other groups do not
    pub icache_lines: usize,
    /// replacement among the `icache_victims` least recently used lines (pseudo-random; 1: strict LRU)
    pub icache_victims: usize,
    /// the instruction cache starts with the lines the simulated groups' own code paths leave in it (a later wave:
    /// the core ran the same code just before); false: empty
    pub icache_warm: bool,
    pub fetch_jump_ns: f64,
    /// cores sharing the SLC and DRAM (all cores run the same kernel on disjoint data)
    pub active_cores: f64,
    pub l1_bytes: usize,
    pub l2_bytes: usize,
    pub slc_bytes: usize,
    /// extra load latency over an L1 hit: L2 in real cycles; SLC and DRAM total latency in ns
    pub l2_extra: f64,
    pub slc_ns: f64,
    /// a dependent load's first touch in a dispatch of a line the previous command left in the SLC: ~300 ns a wait
    /// with sequential lines (playground novel-results/2026-11-01-first-touch-chains: chases and column sums, one or
    /// 128 groups), below `slc_ns` (349, random lines over 1-32 MiB within a command, 2026-10-10-agx-load-latency).
    /// AGX_FIRST_TOUCH_NS overrides it
    pub first_touch_ns: f64,
    pub dram_ns: f64,
    pub l1_load_rate: f64,
    pub l1_load_min: f64,
    pub l1_store_rate: f64,
    pub l1_store_min: f64,
    pub l2_rate: f64,
    pub slc_rate: f64,
    pub dram_gbps: f64,
    /// fraction of its lines' DRAM traffic a core pays: a line that neighbouring workgroups (other cores) also read
    /// is fetched once for all of them (agx_dispatch estimates it; 1 = no sharing)
    pub dram_share: f64,
    /// replay every group's accesses once before timing (steady state of a long-running loop)
    pub warm: bool,
    /// mean fraction (0..1) of this core's device lines that another core also touches (shared data)
    pub sharing: f64,
    /// the dispatch's device data already sits in the system-level cache (written by the CPU or a previous
    /// dispatch just before, and small enough): first touches hit the SLC instead of DRAM. agx_dispatch sets it
    /// when the estimated GPU-wide footprint fits `slc_bytes`.
    pub resident_slc: bool,
    /// with `resident_slc`: the fraction of the dispatch's lines found in the SLC (1: all)
    pub slc_resident_frac: f64,
    /// with `first_touch_dram`: the fraction of a wave's first touches that are new to the whole dispatch (the rest
    /// were read by earlier waves or other cores and are in the SLC); 1: all
    pub first_touch_frac: f64,
    /// with `warm`: replay only these segments (a later wave finds in the caches what the first wave also touched,
    /// not its own new data; `Launch::partial_warm`); None: all of the wave's
    pub warm_only: Option<std::sync::Arc<std::collections::HashSet<u64>>>,
    /// with `resident_slc`: a segment's first touch in the dispatch is charged at the DRAM rate (L2 latency)
    pub first_touch_dram: bool,
    /// device atomics (no return, distinct words): per-core occupancy max(atomic_min, atomic_per_seg x 32-byte
    /// sectors) reference cycles; lanes on one word serialise GPU-wide at atomic_same_word_rate per cycle (every lane separately: the 32 lanes of an
    /// instruction are not combined; playground 2026-10-16-atomic-contention)
    pub atomic_min: f64,
    pub atomic_per_seg: f64,
    pub atomic_same_word_rate: f64,
    /// threadgroup memory, one SIMD group back to back: reference cycles per load / store plus one per extra
    /// bank-conflict way (core: max(1, conflict degree) per instruction)
    pub tg_group_load: f64,
    pub tg_group_store: f64,
    /// threadgroup loads: reference cycles of the core's threadgroup pipeline per access (times the conflict degree);
    /// 1.0 by default, AGX_TG_CORE_LOAD for experiments (with AGX_TG_GATE)
    pub tg_core_load: f64,
    /// threadgroup atomics, one group back to back (no return / return)
    pub tg_group_atomic: f64,
    pub tg_group_atomic_ret: f64,
    /// device atomic with return: latency beyond a device load (executed past L1)
    pub atomic_ret_extra: f64,
}

impl Default for MemModel {
    fn default() -> Self {
        let k = REAL_CYCLES_PER_UNIT;
        MemModel {
            fetch_line_ns: 0.0,
            icache_bytes: 0,
            icache_lines: 0,
            icache_victims: 1,
            icache_warm: false,
            fetch_jump_ns: 0.0,
            clock_mhz: 1398.0,
            active_cores: 38.0,
            l1_bytes: 8 << 10,
            l2_bytes: 256 << 10,
            // shared-cache capacity for streaming reuse. The model's LRU steps at about 1.15 x slc_bytes (19 MiB: 22 MiB), the measured
            // half-way point of the smooth 14-30 MiB transition is about 21 MiB (playground novel-results/2026-10-18-slc)
            slc_bytes: 19 << 20,
            l2_extra: 90.0,
            slc_ns: 349.0,
            first_touch_ns: std::env::var("AGX_FIRST_TOUCH_NS").ok().and_then(|v| v.parse().ok()).unwrap_or(300.0),
            dram_ns: 388.0,
            l1_load_rate: 2.0 / k,
            l1_load_min: 2.0,
            l1_store_rate: 1.02 / k,
            l1_store_min: 2.0,
            l2_rate: 0.73 / k,
            slc_rate: 0.5 / k,
            dram_gbps: 410.0,
            dram_share: 1.0,
            warm: true,
            sharing: 0.0,
            resident_slc: false,
            slc_resident_frac: 1.0,
            first_touch_frac: 1.0,
            warm_only: None,
            first_touch_dram: false,
            atomic_min: 6.9,
            atomic_per_seg: 1.7,
            atomic_same_word_rate: 0.89,   // playground novel-results/2026-10-16-atomic-contention: 0.86 ns per lane-op on one word at 1,398 MHz
            tg_group_load: std::env::var("AGX_TG_GROUP_LOAD").ok().and_then(|v| v.parse().ok()).unwrap_or(31.0),
            tg_core_load: std::env::var("AGX_TG_CORE_LOAD").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            tg_group_store: 24.0,
            tg_group_atomic: 4.8,
            tg_group_atomic_ret: 5.75,
            atomic_ret_extra: 85.0,
        }
    }
}

/// Hasher for segment numbers: the splitmix64 finalizer (SipHash dominated memory-bound runs). Only lookups use
/// it; nothing iterates the maps it serves, so results do not depend on it.
#[derive(Default, Clone, Copy)]
struct SegHash(u64);

impl std::hash::Hasher for SegHash {
    fn finish(&self) -> u64 {
        let mut z = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8 | self.0 >> 56) ^ b as u64;
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 ^= x;
    }
}

type SegBuild = std::hash::BuildHasherDefault<SegHash>;

/// Least-recently-used set of 64-byte segments: a hash map to slots of a doubly linked list in recency order
/// (head = least recent), so a touch is O(1). Evicts exactly as an order-by-last-touch map would.
struct Lru {
    cap: usize,
    /// evict one of the `victims` least recently used lines, chosen pseudo-randomly (1: strict LRU)
    victims: usize,
    rng: u64,
    at: HashMap<u64, u32, SegBuild>,
    /// slot: (segment, previous (less recent), next (more recent)); NIL ends the list
    slot: Vec<(u64, u32, u32)>,
    free: Vec<u32>,
    head: u32,
    tail: u32,
}

const NIL: u32 = u32::MAX;

impl Lru {
    fn new(bytes: usize) -> Self {
        Lru { cap: (bytes / 64).max(1), victims: 1, rng: 0x9E37_79B9_7F4A_7C15, at: HashMap::default(), slot: Vec::new(), free: Vec::new(), head: NIL, tail: NIL }
    }
    fn contains(&self, s: u64) -> bool {
        self.at.contains_key(&s)
    }
    fn unlink(&mut self, i: u32) {
        let (_, p, n) = self.slot[i as usize];
        if p == NIL { self.head = n } else { self.slot[p as usize].2 = n }
        if n == NIL { self.tail = p } else { self.slot[n as usize].1 = p }
    }
    fn push_tail(&mut self, i: u32) {
        self.slot[i as usize].1 = self.tail;
        self.slot[i as usize].2 = NIL;
        if self.tail == NIL { self.head = i } else { self.slot[self.tail as usize].2 = i }
        self.tail = i;
    }
    fn touch(&mut self, s: u64) {
        if let Some(&i) = self.at.get(&s) {
            if i != self.tail {
                self.unlink(i);
                self.push_tail(i);
            }
            return;
        }
        let i = match self.free.pop() {
            Some(i) => {
                self.slot[i as usize].0 = s;
                i
            }
            None => {
                self.slot.push((s, NIL, NIL));
                (self.slot.len() - 1) as u32
            }
        };
        self.at.insert(s, i);
        self.push_tail(i);
        if self.at.len() > self.cap {
            let mut h = self.head;
            if self.victims > 1 {
                // xorshift64: a fixed sequence, so a launch's prediction is reproducible
                self.rng ^= self.rng << 13;
                self.rng ^= self.rng >> 7;
                self.rng ^= self.rng << 17;
                for _ in 0..self.rng % self.victims.min(self.cap) as u64 {
                    h = self.slot[h as usize].2;
                }
            }
            self.unlink(h);
            self.at.remove(&self.slot[h as usize].0);
            self.free.push(h);
        }
    }
}

/// agx_cost.simulate: `groups` identical SIMD groups on one core; returns real cycles (and per-pc stats).
pub fn simulate(instrs: &[Instr], pcs: &[usize], p: &Params, groups: usize, detail: bool)
    -> Result<(f64, Option<HashMap<usize, PcStats>>), String> {
    let seqs: Vec<&[usize]> = (0..groups).map(|_| pcs).collect();
    simulate_groups(instrs, &seqs, None, None, p, detail, None)
}

/// `simulate` of a trace: its pc sequence and its empty-mask steps (`Trace::masked`).
pub fn simulate_trace(instrs: &[Instr], tr: &Trace, p: &Params, groups: usize, detail: bool)
    -> Result<(f64, Option<HashMap<usize, PcStats>>), String> {
    let seqs: Vec<&[usize]> = (0..groups).map(|_| tr.pcs.as_slice()).collect();
    let masked: Vec<&[u32]> = (0..groups).map(|_| tr.masked.as_slice()).collect();
    simulate_groups(instrs, &seqs, Some(&masked), None, p, detail, None)
}

/// One core running one SIMD group per entry of `seqs` (dynamic pc sequences, e.g. from `trace_group`).
/// With `mem` (each group's `Trace::mem` and a `MemModel`), device loads and stores go through the
/// memory hierarchy; without it they use the L1-hit rules only (as `simulate`).
/// `wg_of`: workgroup of each group (barrier scope; None = all groups in one workgroup). Groups of a
/// workgroup are listed in SIMD-group order.
/// `masked`: each group's `Trace::masked` (steps issued with an empty execution mask; None = no such step).
pub fn simulate_groups(instrs: &[Instr], seqs_pcs: &[&[usize]], masked: Option<&[&[u32]]>,
    mem: Option<(&[&[(u32, Vec<u64>)]], &MemModel)>, p: &Params, detail: bool, wg_of: Option<&[u32]>)
    -> Result<(f64, Option<HashMap<usize, PcStats>>), String> {
    simulate_groups_finish(instrs, seqs_pcs, masked, mem, p, detail, wg_of).map(|(c, st, _)| (c, st))
}

/// `simulate_groups`, also giving the cycle each SIMD group finished: what continuous refill needs to know when a
/// workgroup's registers come free (`dispatch::refill_makespan`).
pub fn simulate_groups_finish(instrs: &[Instr], seqs_pcs: &[&[usize]], masked: Option<&[&[u32]]>,
    mem: Option<(&[&[(u32, Vec<u64>)]], &MemModel)>, p: &Params, detail: bool, wg_of: Option<&[u32]>)
    -> Result<(f64, Option<HashMap<usize, PcStats>>, Vec<f64>), String> {
    let groups = seqs_pcs.len();
    let wg: Vec<u32> = wg_of.map(|w| w.to_vec()).unwrap_or_else(|| vec![0; groups]);
    let local: Vec<usize> = (0..groups).map(|i| (0..i).filter(|&j| wg[j] == wg[i]).count()).collect();
    let mut bar_count = vec![0u64; groups];
    let mut bar_arrive: Vec<Option<i64>> = vec![None; groups];
    let mut bar_release: HashMap<(u32, usize, u64), i64, SegBuild> = HashMap::default(); // lookups only: the hasher cannot change results
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
    // taken forward jumps (if-skip jmp_exec_none, out-of-line jmp_exec_any entries): the same latency as a taken loop
    // branch, plus a scheduler hold that other groups cannot hide. Fitted 2026-10-20 to constant-condition probes at
    // 131,072 lanes (apple-documentation-playground novel-results/2026-10-20-fp32-ieee): +7.7 cycles per group per jump.
    // The hold was fitted to 40 KB straight-line probes, which are instruction-fetch-bound: a taken jump breaks
    // sequential fetch through code the instruction cache cannot keep. In a loop whose code stays cached (16
    // KB or less here, a guess between the two measured cases) a taken jump over 1 to 32 instructions costs what an
    // untaken one does at 40 groups per core (2026-10-23 k_ifk probes: 580-595 cycles per 8 if-skips per group
    // either way, novel-results/2026-10-23-geekbench-captures)
    let code_bytes = instrs.iter().map(|i| i.pc + i.size).max().unwrap_or(0);
    // The fetch-bound case is about the code the groups run, not the program's size: hybridsort's mergeSortFirst
    // (28 KB of code, tex1Dfetch emulation, 4.6 KB executed) was +39% with the hold on every if-skip; 72 lines
    // stay cached. AGX_BR_FWD_EXEC=0: the program's size decides (round 3)
    let run_bytes = if std::env::var("AGX_BR_FWD_EXEC").map_or(true, |v| v != "0") && code_bytes > FETCH_BOUND_CODE_BYTES {
        let lines: std::collections::HashSet<usize> = seqs_pcs.iter().flat_map(|q| q.iter().map(|&pc| pc >> 6)).collect();
        lines.len() * 64
    } else {
        code_bytes
    };
    let fetch_bound = std::env::var("AGX_FETCH_BOUND_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(FETCH_BOUND_CODE_BYTES);
    let br_fwd_sched = if run_bytes > fetch_bound {
        tk(p.cal_f("branch_forward_sched_real", 33.0))
    } else {
        tk(std::env::var("AGX_BR_FWD_CACHED").ok().and_then(|v| v.parse::<f64>().ok())
            .unwrap_or_else(|| p.cal_f("branch_forward_sched_cached_real", 0.0)))
    };
    let br_fwd = tk(p.cal_f("branch_forward_taken_real", p_branch));
    let cl = tk(p.cal_f("control_latency", 2.0));
    let ci = tk(p.cal_f("control_issue_cycles", 1.0));
    let wait_idle = tk(p.wait_idle);
    let wait_drain = tk(p.wait_drain);
    let quiet_tg = tk(p.quiet_tg);
    let quiet_load = tk(p.quiet_load);
    let quiet_dstore = tk(p.quiet_device_store);
    let tg_lat0 = tk(p.tg_load_lat.0);
    let tg_issue0 = tk(p.tg_load_issue);
    // threadgroup loads hold their SIMD group's issue (scripts/probe_tg_alu.py, M2 Max 2026-11-07; playground
    // novel-results/2026-11-07-tg-load-issue-hold): 24 real cycles per load with an immediate base, 28 with a register
    // base, whatever follows; a load after ALU work issues only after the quiet time (the group waits, it is not only
    // latency), so ALU instructions between threadgroup loads cost ~21 cycles per run. AGX_TG_ISSUE_REAL=0 restores the
    // earlier model (21-unit issue interval, quiet time charged to the result only); AGX_TG_ISSUE_REG_REAL sets the
    // register-base extra.
    let tg_block: Option<i64> = match std::env::var("AGX_TG_ISSUE_REAL").ok().and_then(|v| v.parse::<f64>().ok()) {
        Some(c) if c <= 0.0 => None,
        Some(c) => Some(tk(c)),
        None => Some(tk(p.cal_f("tg_load_hold_real", 24.0))),
    };
    let tg_block_reg: i64 = tk(std::env::var("AGX_TG_ISSUE_REG_REAL").ok().and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_else(|| p.cal_f("tg_load_hold_reg_extra_real", 4.0)));
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
    // Instructions issued with an empty execution mask (playground results/2026-10-20-b31-empty-mask-cost, rerun.jsonl:
    // one SIMD group, an if body of 32 copies that no lane enters, issued without a jump; per instruction, the median
    // over three interleaved rounds of (no jump - jmp_exec_none over the body) + the jump's own 11.48 cycles / 32).
    // A masked ALU instruction costs about nothing more than jumping over it (iadd 0.2 / 2.3 / 0.4 ns, fmadd32 -2.3 /
    // 1.1 / -0.5 ns, simd_shuffle 0.2 ns in the first pass): one issue cycle, no unit, no result latency and no
    // dependency stall. Masked memory instructions keep a large part of their cost: device_load + wait 0
    // 10.0 / 17.2 / 11.1 ns (active ~37-41 ns) and threadgroup_load + wait 0 12.1 / 11.5 / 14.8 ns (active ~23-28),
    // modelled as a load completing on its scoreboard slot that much later (the wait still waits; no data moves);
    // device_store 9.9 / 9.7 / 6.6 ns (more than an active one; it writes nothing), modelled as its issue interval.
    // ns x CYCLES_PER_NS, as the fence table. Unmeasured memory classes (atomics, threadgroup stores) cost as active.
    let masked_alu = tk(p.cal_f("masked_alu_issue_real", 1.0));
    let masked_load = tk(p.cal_f("masked_device_load_real", MASKED_DEVICE_LOAD_NS * CYCLES_PER_NS));
    let masked_tg_load = tk(p.cal_f("masked_tg_load_real", MASKED_TG_LOAD_NS * CYCLES_PER_NS));
    let masked_store = tk(p.cal_f("masked_device_store_real", MASKED_DEVICE_STORE_NS * CYCLES_PER_NS));
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
            Some(row) => (tk(row.lat + r.lat_extra), tk(row.ii)),
            None => (0, 0),
        };
        // `_cache` classes: the extra issue cycle only when a `.cache` source misses the result cache
        let ii_cache_hit = match r.unit.as_deref().and_then(|u| u.strip_suffix("_cache")).and_then(|b| p.rows.get(b)) {
            Some(b) if !r.cache_regs.is_empty() => Some(tk(b.ii)),
            _ => None,
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
            shuffle: r.name.contains("shuffle"),
            fp32: matches!(r.name.as_str(), "fadd32" | "fmul32" | "fmadd32"),
            lat,
            ii,
            ii_cache_hit,
            cache_regs: r.cache_regs.clone(),
            late_src: r.late_src.clone(),
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
            fence: r.fence,
            dev_ctx: if r.name.starts_with("device_load") { 2 } else if r.name.starts_with("device_store") { 1 } else { 0 },
            mem_issue0,
            mem_done,
            is_jmp_any: matches!(r.name.as_str(), "jmp_exec_any" | "jmp_incomplete"),
            is_jmp: matches!(r.name.as_str(), "jmp_exec_any" | "jmp_exec_none" | "jmp_incomplete"),
            is_call: matches!(r.name.as_str(), "call" | "ret"),
            stack_store: r.name == "stack_store",
            stack: r.name.starts_with("stack_"),
            size: r.size,
            exec_stack: matches!(r.name.as_str(), "if_icmp" | "if_fcmp" | "else_icmp" | "else_fcmp" | "pop_exec" | "push_exec" | "jmp_exec_none"),
            ctl_bump: r.src.iter().any(|&h| h != 0),
            masked_cls: match (r.kind, r.name.as_str()) {
                (Kind::Alu, _) => 1,
                (Kind::Load, "device_load" | "device_load.TODO") => 2,
                (Kind::Store, "device_store" | "device_store.TODO") => 3,
                (Kind::Load, "threadgroup_load") => 4,
                _ => 0,
            },
        });
    }
    let mut seqs: Vec<Vec<usize>> = vec![];
    // pc -> index into `sim`, as `by_pc` (later entries win), without hashing every dynamic pc
    let mut pc_index: Vec<Option<usize>> = vec![None; by_pc.keys().max().map_or(0, |&m| m + 1)];
    for (&pc, &i) in &by_pc {
        pc_index[pc] = Some(i);
    }
    for pcs in seqs_pcs {
        seqs.push(pcs.iter().map(|&pc| pc_index.get(pc).copied().flatten().ok_or_else(|| format!("KeyError: pc {}", pc))).collect::<Result<_, _>>()?);
    }
    // per group: whether each step was issued with an empty execution mask (empty when none was)
    let mut mflag: Vec<Vec<bool>> = vec![vec![]; groups];
    if let Some(ms) = masked {
        if ms.len() != groups {
            return Err(format!("simulate_groups: {} masked-step lists for {} groups", ms.len(), groups));
        }
        for (gi, list) in ms.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let mut f = vec![false; seqs[gi].len()];
            for &k in list.iter() {
                *f.get_mut(k as usize).ok_or_else(|| format!("simulate_groups: masked step {} beyond the trace", k))? = true;
            }
            mflag[gi] = f;
        }
    }
    let is_masked = |gi: usize, step: usize, r: &SimInstr| r.masked_cls != 0 && mflag[gi].get(step).copied().unwrap_or(false);
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
        /// latency beyond L1 of a load whose slowest line is a first touch in the dispatch found in the SLC
        /// (`first_touch_dram`)
        first_extra: i64,
        /// lines a load has missed for, with the tick its latency beyond L1 ends: a load by another group that finds
        /// them before then waits for that fetch, not an L1 hit (`AGX_INFLIGHT=0`: no such wait)
        inflight: HashMap<u64, i64>,
        cur: Vec<usize>,
        /// L1 load / store pipelines: busy until (ticks); a load's or store's segments pass in order
        l1_free: [i64; 2],
        /// threadgroup-memory and device-atomic pipelines: busy until (ticks)
        tg_free: i64,
        at_free: i64,
        tg_group_free: Vec<i64>,
        /// (group, index into its memory steps) of the core's previous load: its lines are compared by value
        last_load: Option<(usize, usize)>,
        /// bumped whenever the caches change (an issued access touches its lines)
        epoch: u64,
        /// per group: (memory step, epoch, segments missing L1 per level, deepest level) for the step being retried
        lv_cache: Vec<Option<(usize, u64, [f64; 3], usize)>>,
        /// `first_touch_dram`: segments this dispatch has touched, and per access (group, index) how many of its
        /// lines were first touches (an access is re-evaluated while it waits for its buckets)
        seen: std::collections::HashSet<u64>,
        first_frac: f64,
        first: HashMap<(usize, usize), (f64, f64)>,
    }
    // AGX_CHARGE_WRITEBACK=1: a store also takes DRAM bandwidth for its lines when the working set does not fit the
    // SLC, even if they were just loaded (the dirty lines must be written back). Off by default: no hardware data
    // yet (applications notes 66-67: adamw, stream).
    let tg_gate = std::env::var("AGX_TG_GATE").map_or(false, |v| v == "1");
    // on by default since the shuffle-tree probes (results/2026-11-01-tg-xy-and-settle): without it, 8 shuffle-reduction
    // chains on values read from threadgroup memory were +161% (38 workgroups); with it +1%. AGX_TG_SETTLE_ON_READ=0:
    // the old deferral (which also stood in for an unmodelled one-shot reduction-tail cost: mul_mat_vec_q, soft_max)
    let settle_on_read = std::env::var("AGX_TG_SETTLE_ON_READ").map_or(true, |v| v != "0");
    // AGX_ISSUE_PROFILE=1: for SIMD group 0 of the first call whose group 0 runs AGX_ISSUE_PROFILE_MIN (1000) instructions or more (the
    // dispatch's own calls, not the cost model's smaller ones before them), the time before each instruction's issue (since the
    // group's previous issue), summed by PC, printed to stderr: where the model's time goes
    static PROFILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let mut issue_profile: Option<std::collections::BTreeMap<usize, (i64, u64)>> =
        (std::env::var("AGX_ISSUE_PROFILE").is_ok_and(|v| v == "1")
            && seqs_pcs.first().is_some_and(|q| {
                q.len() >= std::env::var("AGX_ISSUE_PROFILE_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1000)
            })
            && {
                // AGX_ISSUE_PROFILE_SKIP=N: the first N qualifying calls are not profiled (the cost model's own
                // calibration loops run before a dispatch's waves)
                static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let skip = std::env::var("AGX_ISSUE_PROFILE_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                SEEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= skip
            }
            && !PROFILED.swap(true, std::sync::atomic::Ordering::SeqCst))
            .then(Default::default);
    let mut profile_prev: i64 = 0;
    let first_touch_dram = mem.is_some_and(|(_, m)| m.resident_slc && m.first_touch_dram);
    let mem_levels = std::env::var("AGX_MEM_LEVELS").is_ok_and(|v| v == "1");
    let inflight_on = std::env::var("AGX_INFLIGHT").map_or(true, |v| v != "0");
    let charge_writeback = std::env::var("AGX_CHARGE_WRITEBACK").map_or(false, |v| v == "1")
        && mem.is_some_and(|(_, m)| !m.resident_slc);
    let mut ms: Option<MemState> = mem.map(|(_, m)| {
        let per_tick = |segs_per_cycle: f64| segs_per_cycle / TICKS as f64;
        let dram = m.dram_gbps * 1e9 / m.active_cores / (m.clock_mhz * 1e6) / 64.0 / m.dram_share.clamp(0.05, 1.0);
        let rate = [per_tick(m.l1_load_rate), per_tick(m.l1_store_rate), per_tick(m.l2_rate), per_tick(m.slc_rate), per_tick(dram)];
        let cap = [64.0, 64.0, 64.0, 64.0, 64.0];
        let ns = |x: f64| tk((x * m.clock_mhz / 1000.0 - 50.0).max(0.0));
        MemState {
            l1: Lru::new(m.l1_bytes),
            // L2 (38 x l2_bytes in all) and SLC are shared GPU-wide: a core's share is total / (cores that use
            // distinct data); with `sharing` = fraction of this core's lines that other cores also use, cores
            // reading the same lines all see the whole cache
            l2: Lru::new((m.l2_bytes as f64 * m.active_cores / eff_sharers(m)) as usize),
            slc: Lru::new((m.slc_bytes as f64 / eff_sharers(m)) as usize),
            tok: cap,
            rate,
            cap,
            extra: [0, tk(m.l2_extra), ns(m.slc_ns), ns(m.dram_ns)],
            first_extra: ns(m.first_touch_ns),
            first_frac: m.first_touch_frac,
            inflight: Default::default(),
            cur: vec![0; groups],
            l1_free: [0, 0],
            tg_free: 0,
            at_free: 0,
            tg_group_free: vec![0; groups],
            last_load: None,
            epoch: 0,
            lv_cache: vec![None; groups],
            seen: Default::default(),
            first: Default::default(),
        }
    });
    let memv: Vec<&[(u32, Vec<u64>)]> = mem.map(|(v, _)| v.to_vec()).unwrap_or_default();
    // a fingerprint of each access's segment list: lists with different fingerprints differ, so only equal
    // fingerprints need the full comparison
    let memfp: Vec<Vec<u64>> = memv.iter().map(|v| v.iter().map(|(_, sg)| {
        sg.iter().fold(sg.len() as u64, |h, &x| (h ^ x).wrapping_mul(0x100_0000_01b3).rotate_left(29))
    }).collect()).collect();
    if let (Some(m), Some((_, mm))) = (ms.as_mut(), mem) {
        if mm.resident_slc && (!mm.warm || mm.warm_only.is_some()) {
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
            // the previous command's data is in the SLC, not L2: command boundaries invalidate L2 (playground
            // novel-results/2026-10-16-l2-command-boundary) and a dependent wait on a new line costs ~300 ns
            // (2026-11-01-first-touch-chains; results/2026-11-01-latency-chain-captures: all 56 captures 12.7 -> 10.3%).
            // AGX_FIRST_TOUCH_L2=1: first touches at L2 latency (the earlier fit to llama.cpp's matrix-vector kernels,
            // -50% on dependent chains)
            let to_l2 = mm.first_touch_dram && std::env::var("AGX_FIRST_TOUCH_L2").is_ok_and(|v| v == "1");
            for &sg in &distinct {
                // (a footprint in the SLC's smooth transition: this fraction of the lines is still there)
                if mm.slc_resident_frac < 1.0
                    && (sg.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64 / (1u64 << 53) as f64 >= mm.slc_resident_frac {
                    continue;
                }
                m.slc.touch(sg);
                if to_l2 {
                    m.l2.touch(sg);
                }
            }
        }
        if mm.warm {
            let longest = memv.iter().map(|v| v.len()).max().unwrap_or(0);
            for i in 0..longest {
                for v in &memv {
                    if let Some((_, segs)) = v.get(i) {
                        for &sg in segs {
                            if mm.warm_only.as_ref().is_some_and(|w| !w.contains(&sg)) {
                                continue;
                            }
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
    let wb_rate = WRITEBACK_PER_CYCLE / t_ as f64; // per tick
    let mut wb_tok = WRITEBACK_CAP;
    let mut wbc: Vec<Vec<u32>> = vec![Vec::with_capacity(nc + 1); groups];
    let mut wb_debt = vec![0.0f64; groups];
    let mut lcg: u64 = 0x2545F4914F6CDD1D;
    let mut last_alu = vec![NEG_INF; g];
    let mut alu_done = vec![NEG_INF; g];
    // SIMD-group reductions / shuffles in flight on the core (their completion times) and the latency each adds to
    // a new one: the core's shuffle unit saturates gradually (1 / 4 / 16 / 48 groups of dependent chains: 3.0x /
    // 7.4x / 9.8x one group's rate, results/2026-11-01-geekbench-air-captures). AGX_SHFL_QUEUE: cycles per one in
    // flight (0: off). 0.6 fits those chains within 6% (GPU 18.96 / 24.25 / 38.29 / 83.17 us, model 20.1 / 23.2 / 38.0 /
    // 83.8) but worsens the instruction-bench fit (p90 0.169 -> 0.278) and the core-scaling grid (p90 0.277 -> 0.332),
    // whose many independent chains per group the unit cap already limits, and leaves the 48 captures unchanged: off
    // until the extra latency follows the unit's load instead of a count
    let shfl_queue = tk(std::env::var("AGX_SHFL_QUEUE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0));
    let mut shfl_inflight: Vec<(usize, i64)> = Vec::new();
    // AGX_SHFL_PIPE=c (experiment): shuffles and, while a shuffle issued in the last 64 cycles, FP32 adds / multiplies
    // share one pipe that takes an instruction every c cycles (mixed shuffle / fadd chains, results/2026-11-01-
    // geekbench-air-captures). 0.6 fits those chains (sf.ptx 16 / 48 groups 28.9 / 74.6 us, GPU 30.6 / 81.5; red_shfl
    // 34.0 / 61.5, GPU 32.0 / 57.4) but leaves the 48 captures unchanged and two held-out sets of verify-cost-model a
    // little worse (heldout24 p90 0.118 -> 0.150, heldout32c median 0.039 -> 0.047): off by default
    let shfl_pipe = tk(std::env::var("AGX_SHFL_PIPE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0));
    let shfl_recent = tk(64.0);
    let mut pipe_free = NEG_INF;
    let mut last_shfl = NEG_INF;
    let mut last_wait = vec![NEG_INF; g];
    // per group: the last device store (1) / load (2) issued before the next fence; 0 after a fence or at the start
    let mut last_mem = vec![0u8; g];
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
    /// pending loads already moved by an ALU push: their done is last_push + lat, merged per (slot, instruction)
    struct Settled {
        slot: usize,
        ins: usize,
        lat: i64,
        lat_less_extra: i64,
    }
    let mut settled: Vec<Vec<Settled>> = (0..g).map(|_| vec![]).collect();
    let mut last_push = vec![NEG_INF; g];
    let mut last_t: i64 = 0;
    let mut stats: Vec<(u64, i64, i64, i64)> = vec![(0, 0, 0, 0); sim.len()];
    // wake times offered this tick: only their minimum, and the minimum of those after t + 1, are ever read
    let mut wk_min: i64;
    let mut wk_gt: i64;
    let mut sr_cache: Vec<Option<i64>> = vec![None; g];
    #[derive(Clone, Copy)]
    enum Retest {
        Issue,
        Bucket(usize, f64),
        Port(usize, f64),
    }
    let mut retest: Vec<Retest> = Vec::new();
    let mut rt_at: Vec<usize> = vec![usize::MAX; buckets.len()];
    let mut bar_ver: u64 = 0;
    let mut bar_none_at: Vec<u64> = vec![u64::MAX; g];
    let mut unblocked = vec![false; g];
    let mut fresh_cache: Vec<Option<(usize, i64)>> = vec![None; g];
    let mut blk_cache: Vec<Option<i64>> = vec![None; g];

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

    // cold instruction fetch, per core, for a whole dispatch (MemModel::fetch_line_ns)
    let mut fetch_line = mem.map_or(0, |(_, m)| tk(m.fetch_line_ns * CYCLES_PER_NS));
    // Code that fits the instruction cache stays there from one launch to the next: nothing to fetch (warm launches).
    if let Some((_, m)) = mem {
        if fetch_line > 0 && m.icache_bytes > 0 {
            let lines: std::collections::HashSet<usize> = seqs_pcs.iter().flat_map(|p| p.iter().map(|pc| pc >> FETCH_LINE_SHIFT)).collect();
            if lines.len() << FETCH_LINE_SHIFT <= m.icache_bytes {
                fetch_line = 0;
            }
        }
    }
    let mut fetched: HashMap<usize, i64> = HashMap::new();
    let mut fetch_free = 0i64;
    // instruction cache with refetch (MemModel::icache_lines): its lines, and the jump-target miss cost
    let (mut icache, fetch_jump) = match mem {
        Some((_, m)) if m.icache_lines > 0 && fetch_line > 0 => {
            let mut c = Lru::new(m.icache_lines * 64);
            c.victims = std::env::var("AGX_ICACHE_VICTIMS").ok().and_then(|v| v.parse().ok()).unwrap_or(m.icache_victims).max(1);
            if m.icache_warm {
                // the previous workgroups on this core ran the same code: the cache holds what their pass left
                for p in seqs_pcs.iter() {
                    for &pc in p.iter() {
                        c.touch((pc >> FETCH_LINE_SHIFT) as u64);
                    }
                }
            }
            (Some(c), tk(m.fetch_jump_ns * CYCLES_PER_NS))
        }
        _ => (None, 0),
    };
    // with it, the core's taken-jump structure: 32 entries (by target), a taken jump to a cached line that has none
    // costs ~127 cycles more (playground novel-results/2026-10-20-icache: 34-96 blocks joined by taken jumps per loop
    // trip, +127 cycles per jump beyond 32). AGX_JUMP_ENTRIES=0 turns it off
    let jump_entries = std::env::var("AGX_JUMP_ENTRIES").ok().and_then(|v| v.parse().ok()).unwrap_or(32usize);
    let mut jumps = (icache.is_some() && jump_entries > 0).then(|| Lru::new(jump_entries * 64));
    let jump_extra = tk(127.0);
    let mut t: i64 = 0;
    macro_rules! wake {
        ($x:expr) => {{
            let x_: i64 = $x;
            wk_min = wk_min.min(x_);
            if x_ > t + 1 {
                wk_gt = wk_gt.min(x_);
            }
        }};
    }
    let mut done = 0usize;
    let npcs: Vec<usize> = seqs.iter().map(Vec::len).collect();
    // Groups busy until free[gi] > t wait in a min-heap; the tick visits only the others (`elig`, a bitset), in the
    // same rotated order. free[gi] changes only when the group issues, so a busy group's only effect on the tick was
    // to offer free[gi] as a wake time: the heap's minimum gives the same min(wake).
    let words = g.div_ceil(64).max(1);
    let mut elig: Vec<u64> = vec![0; words];
    let mut busy: std::collections::BinaryHeap<std::cmp::Reverse<(i64, usize)>> = std::collections::BinaryHeap::new();
    for gi in 0..g {
        if nxt[gi] < npcs[gi] {
            busy.push(std::cmp::Reverse((0, gi)));
        }
    }
    let mut issued_now: Vec<usize> = Vec::with_capacity(g);
    while done < g {
        let mut issued = 0;
        while let Some(&std::cmp::Reverse((f, gi))) = busy.peek() {
            if f > t {
                break;
            }
            busy.pop();
            // A group's blocks (src_ready, a wait's slot, wait_idle) depend only on its own state, which changes only
            // when it issues, so they are tested once, here, when it becomes eligible (the value the first visit this
            // tick would compute): a group still blocked goes back to the heap until the block lifts, and the groups
            // in `elig` are all unblocked.
            let ri = seqs[gi][nxt[gi]];
            let r = &sim[ri];
            // a masked ALU instruction reads no operand: no dependency stall
            let sr = match sr_cache[gi] {
                Some(v) => v,
                None if r.masked_cls == 1 && is_masked(gi, nxt[gi], r) => {
                    sr_cache[gi] = Some(NEG_INF);
                    NEG_INF
                }
                None => {
                    let mut sr = 0i64;
                    let mut any = false;
                    for &h in &r.src {
                        let i = touch(&mut ready[gi], &mut present[gi], &mut present_list[gi], h);
                        let mut v = ready[gi][i];
                        if !r.late_src.is_empty() && r.late_src.contains(&h) {
                            v -= t_;
                        }
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
            // These three blocks depend only on this group's own state, which changes only when it issues:
            // park the group in the heap until the block lifts (same wake time as rechecking every tick).
            let blocked = if sr > t {
                sr
            } else if r.kind == Kind::Wait {
                match slots[gi][r.slot] {
                    Some(m) if m > t => m,
                    _ if last_wait[gi] + wait_idle > t => last_wait[gi] + wait_idle,
                    _ => 0,
                }
            } else {
                0
            };
            if blocked > t {
                busy.push(std::cmp::Reverse((blocked, gi)));
                continue;
            }
            unblocked[gi] = true;
            elig[gi / 64] |= 1 << (gi % 64);
        }
        let k0 = if g == 1 { 0 } else { (((t / t_) as u64).wrapping_mul(2654435761) >> 7) as usize % g };
        wk_min = i64::MAX;
        wk_gt = i64::MAX;
        // Pass 0 tries only the group's own scheduler, and a scheduler busy at the tick start stays busy all tick
        // (sched_free only grows, by issuing through a free one). Its eligible groups would each offer
        // sched_free[home] as a wake time and do nothing else, so they are not visited in pass 0: the scheduler offers
        // the wake time once. Groups gi with gi % SCHEDULERS == s are the bits (0x1111.. << s) of every word.
        let mut free_mask = 0u64;
        for s in 0..SCHEDULERS {
            let m = 0x1111_1111_1111_1111u64 << s;
            if sched_free[s] <= t {
                free_mask |= m;
            } else if elig.iter().any(|&e| e & m != 0) {
                wake!(sched_free[s]);
            }
        }
        for x in &retest {
            if let Retest::Bucket(b, _) = *x {
                rt_at[b] = usize::MAX;
            }
        }
        retest.clear();
        let mut ff_ok = true;
        let mut wmin_time = i64::MAX;
        for &gi in &issued_now {
            tick_issued[gi] = false;
        }
        issued_now.clear();
        for pass in 0..npass {
        // pass 1 issues only through a scheduler that is free this tick: with none free, every group would skip
        if pass == 1 && sched_free.iter().all(|&f| f > t) {
            break;
        }
        if pass == 1 {
            // Pass 1 acts only through a group whose home scheduler is busy (the others skip), and only such a group
            // can make another home busy by stealing: with none eligible, the pass does nothing.
            let mut busy_mask = 0u64;
            for s in 0..SCHEDULERS {
                if sched_free[s] > t {
                    busy_mask |= 0x1111_1111_1111_1111u64 << s;
                }
            }
            if elig.iter().all(|&e| e & busy_mask == 0) {
                break;
            }
        }
        // Visit the eligible groups in the rotated order k0..g, 0..k0 with a cursor over `elig`, which is read as it
        // changes: groups only leave it during a tick (by issuing), and a group that issued is behind the cursor.
        // Eligible groups are unfinished, not busy (free <= t), unblocked, and have not issued this tick.
        // Pass 0 visits the groups of the schedulers free at the tick start (`free_mask`, above). In pass 1 a group
        // whose home scheduler is free skips, so it visits those whose home is busy at the moment the cursor
        // reaches them: the mask is recomputed after each steal, and a home made busy by a steal brings in its
        // groups ahead of the cursor, as the full order would.
        let mut mask = if pass == 0 { free_mask } else { 0 };
        let mut mask_stale = true; // sched_free changed since `mask` was computed
        let segs = [(k0, g), (0, k0)];
        let mut seg = 0;
        let mut cur = k0;
        loop {
            if pass == 1 && mask_stale {
                mask = 0;
                for s in 0..SCHEDULERS {
                    if sched_free[s] > t {
                        mask |= 0x1111_1111_1111_1111u64 << s;
                    }
                }
                mask_stale = false;
            }
            let mut gi = usize::MAX;
            while seg < 2 {
                let hi = segs[seg].1;
                while cur < hi {
                    let wd = (elig[cur / 64] & mask) >> (cur % 64);
                    if wd == 0 {
                        cur = (cur / 64 + 1) * 64;
                        continue;
                    }
                    cur += wd.trailing_zeros() as usize;
                    break;
                }
                if cur < hi {
                    gi = cur;
                    break;
                }
                seg += 1;
                if seg < 2 {
                    cur = segs[1].0;
                }
            }
            if gi == usize::MAX {
                break;
            }
            cur = gi + 1;
            debug_assert!(nxt[gi] < npcs[gi] && !tick_issued[gi] && free[gi] <= t);
            debug_assert!(unblocked[gi]);
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
                wake!(sched_free[sch]);
                continue;
            }
            let ri = seqs[gi][nxt[gi]];
            let r = &sim[ri];
            let msk = is_masked(gi, nxt[gi], r);
            if issue_tok < t_ {
                wake!(t + 1);
                retest.push(Retest::Issue);
                continue;
            }
            if r.unit.is_some() && !msk {
                let b = r.bucket;
                if tok[b].is_none() {
                    tok[b] = Some(buckets[b].init);
                    if buckets[b].init != buckets[b].capmax {
                        in_refill[b] = true;
                        refilling.push(b);
                    }
                }
                // (the token test first: the forwarding count only matters once the unit has a token)
                if tok[b].unwrap() < r.cost {
                    wake!(t + 1);
                    // one entry per bucket with the least cost: any(tok >= c_i) is tok >= min(c_i)
                    match rt_at[b] {
                        usize::MAX => {
                            rt_at[b] = retest.len();
                            retest.push(Retest::Bucket(b, r.cost));
                        }
                        k => {
                            if let Retest::Bucket(_, c) = &mut retest[k] {
                                if r.cost < *c {
                                    *c = r.cost;
                                }
                            }
                        }
                    }
                    continue;
                }
                need = r.need_port;
                if need > 0.0 {
                    // registers written within the forwarding window. Depends on this group's write times (which
                    // change only when it issues) and can only drop as t grows: cached with the last tick it holds.
                    let fresh = match fresh_cache[gi] {
                        Some((n, until)) if t <= until => n,
                        _ => {
                            let wt = &wtime[gi];
                            let (mut n, mut until) = (0usize, i64::MAX);
                            for &reg in &r.src32 {
                                if let Some(Some((t0, _))) = wt.get(reg as usize) {
                                    if t - t0 <= fwd {
                                        n += 1;
                                        until = until.min(t0 + fwd);
                                    }
                                }
                            }
                            fresh_cache[gi] = Some((n, until));
                            n
                        }
                    };
                    need = (need - (fresh as i64 * t_) as f64).max(0.0);
                }
                if port[half] < need {
                    wake!(t + 1);
                    retest.push(Retest::Port(half, need));
                    continue;
                }
            }
            if (r.ndst == 1 || r.ndst == 2) && !msk {
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
                    wake!(h);
                    wmin_time = wmin_time.min(h);
                    continue;
                }
            }
            if r.sfu_op && !msk {
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
                    wake!(gate);
                    wmin_time = wmin_time.min(gate);
                    continue;
                }
            }
            if g > 1 && r.wide_cost.is_some() && !msk {
                let gate = core_fast[half].max(core_slow[half]);
                if gate > t {
                    wake!(gate);
                    wmin_time = wmin_time.min(gate);
                    continue;
                }
            }
            if shfl_pipe > 0 && !msk && r.fp32 && t - last_shfl < shfl_recent && pipe_free > t {
                wake!(pipe_free);
                wmin_time = wmin_time.min(pipe_free);
                continue;
            }
            // barrier: wait until every group of the scope has arrived (after draining its ALU work)
            if r.barrier != 0 {
                if bar_arrive[gi].is_none() {
                    ff_ok = false; // arrival spends writeback tokens without issuing
                    // arriving writes back every register in the group's result cache (barrier-fill probe:
                    // 2 writebacks per cycle per core for F dirty registers per group)
                    let n = wbc[gi].len() as f64;
                    if n > 0.0 {
                        if wb_tok < n.min(WRITEBACK_CAP) {
                            wake!(t + (((n.min(WRITEBACK_CAP) - wb_tok) / wb_rate).ceil() as i64).max(1));
                            continue;
                        }
                        wb_tok -= n.min(WRITEBACK_CAP);
                        wb_debt[gi] = n - n.min(WRITEBACK_CAP);
                        wbc[gi].clear();
                    }
                    if wb_debt[gi] > 0.0 {
                        // more writebacks than the bucket holds: pay the rest as the bucket refills
                        let take = wb_debt[gi].min(wb_tok);
                        wb_tok -= take;
                        wb_debt[gi] -= take;
                        if wb_debt[gi] > 0.0 {
                            wake!(t + ((wb_debt[gi] / wb_rate).ceil() as i64).max(1));
                            continue;
                        }
                    }
                    bar_arrive[gi] = Some(t.max(alu_done[gi] + bar_drain));
                    bar_ver += 1;
                }
                // still waiting with nothing arrived or issued since the last look: still waiting (the scan below
                // depends only on arrivals, barrier counts and finished groups)
                if bar_none_at[gi] == bar_ver {
                    continue;
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
                    None => {
                        bar_none_at[gi] = bar_ver;
                        continue;
                    }
                    Some(x) if x > t => {
                        wake!(x);
                        wmin_time = wmin_time.min(x);
                        continue;
                    }
                    _ => {}
                }
            }
            // threadgroup loads and stores issue one at a time per group and at the core's threadgroup rate: back-to-
            // back loads whose results nothing reads take ~32.6 cycles per load per group at up to 16 groups per core
            // and ~2.1 cycles per load per core at 32 (M2 Max, novel-results/2026-10-23-geekbench-captures); the
            // pipelines' occupancy (charged below) delays the next access, not only this one's result
            // EXPERIMENT (AGX_TG_GATE=1, off by default): it fits whole-dispatch times of threadgroup-load loops
            // better (tgcheck.py: median error 41% -> 31% with AGX_TG_CORE_LOAD=1.5) but not the stored in-shader
            // per-step measurements (tg.json: 29.5%), which keep ~31 cycles per step per group at 32 groups while the
            // dispatch takes twice as long: not all 32 groups seem to progress at once.
            // With the issue hold (tg_block) a load waits only for the core's pipeline: its own group is already held.
            let blocked_load = tg_block.is_some() && r.kind == Kind::Load;
            if let (Some(m), false, true) = (ms.as_mut(), msk, tg_gate || blocked_load) {
                if r.tg_name && r.barrier == 0 && !r.atomic && matches!(r.kind, Kind::Load | Kind::Store) && !memv.is_empty() {
                    let until = if blocked_load { m.tg_free } else { m.tg_group_free[gi].max(m.tg_free) };
                    if until > t {
                        ff_ok = false;
                        wake!(until);
                        continue;
                    }
                }
            }
            // device memory: segments of this dynamic instruction, level of each, bandwidth tokens
            let mut mem_charge: Option<([f64; 5], i64, Vec<u64>)> = None;
            if let (Some(m), false) = (ms.as_mut(), msk) {
                let dev = !r.atomic && ((r.kind == Kind::Load && !r.tgop && !r.tg_name) || (r.kind == Kind::Store && !r.tg_name));
                if dev && !memv.is_empty() {
                    ff_ok = false; // the visit records the core's last load lines
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
                            // (a stack store's lanes share a word's slot, whatever lines that crosses)
                            need[1] = if r.stack_store { 1.0 } else { n.max(1.0) }.max(m_l1_store_min(mem));
                        } else {
                            // the same lines as the core's previous load (every group on one line): 1.22 cycles per
                            // load instead of 1.0 (coalesce-shared vs coalesce-private)
                            let fp = memfp[gi][m.cur[gi]];
                            let same = m.last_load.is_some_and(|(g2, c2)| memfp[g2][c2] == fp && memv[g2][c2].1 == *segs);
                            need[0] = n.max(m_l1_load_min(mem) * if same { 1.22 } else { 1.0 });
                            m.last_load = Some((gi, m.cur[gi]));
                        }
                        // each line's level depends only on the cache contents: reused while they are unchanged
                        let level = match m.lv_cache[gi] {
                            Some((c, e, nd, lv)) if c == m.cur[gi] && e == m.epoch => {
                                need[2] = nd[0];
                                need[3] = nd[1];
                                need[4] = nd[2];
                                lv
                            }
                            _ => {
                                let mut level = 0usize;
                                for &sg in segs {
                                    // a stack store does not read the line first (probe_stack_cost.py: 56-62 ns per store at
                                    // 256 bytes and 4 KB of frame alike, where loads go from 68 to 172)
                                    let lv = if r.stack_store { 0 } else if m.l1.contains(sg) { 0 } else if m.l2.contains(sg) { 1 } else if m.slc.contains(sg) { 2 } else { 3 };
                                    if lv >= 1 {
                                        need[1 + lv] += 1.0;
                                    }
                                    level = level.max(lv);
                                }
                                m.lv_cache[gi] = Some((m.cur[gi], m.epoch, [need[2], need[3], need[4]], level));
                                if mem_levels {
                                    let k = (r.stack as usize) * 8 + (store as usize) * 4 + level;
                                    MEM_LEVELS[k].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                level
                            }
                        };
                        // data a previous command left in the system-level cache: L2 latency, but its first touch
                        // in this dispatch goes at the GPU-wide DRAM rate (reads back to back never exceed ~380 GB/s,
                        // 1-12 MiB; playground novel-results/2026-11-01-read-bandwidth)
                        let mut lat_extra = if store { 0 } else { m.extra[level] };
                        if first_touch_dram && need[2] + need[3] > 0.0 {
                            let key = (gi, m.cur[gi]);
                            let (first_any, first) = match m.first.get(&key) {
                                Some(&f) => f,
                                None => {
                                    // (`first_frac` < 1: only the hashed fraction of the first touches is new to the
                                    // dispatch and goes at the DRAM rate; all of them still wait the first-touch latency)
                                    let ff = m.first_frac;
                                    let mut any = 0.0;
                                    let mut new = 0.0;
                                    for &sg in segs.iter() {
                                        if m.seen.insert(sg) {
                                            any += 1.0;
                                            if ff >= 1.0 || ((sg ^ 0x5bd1_e995).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) >> 11) as f64 / ((1u64 << 53) as f64) < ff {
                                                new += 1.0;
                                            }
                                        }
                                    }
                                    m.first.insert(key, (any, new));
                                    (any, new)
                                }
                            };
                            static FTF_LAT_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                            let lat_all = *FTF_LAT_ALL.get_or_init(|| std::env::var("AGX_FIRST_TOUCH_FRAC").is_ok_and(|v| v == "2"));
                            static FTF_SHARED_L2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                            let shared_l2 = *FTF_SHARED_L2.get_or_init(|| std::env::var("AGX_FIRST_TOUCH_FRAC").map_or(true, |v| v == "3"));
                            if (if lat_all { first_any } else { first }) > 0.0 && level == 2 && !store {
                                lat_extra = m.first_extra;
                            } else if shared_l2 && first_any > 0.0 && level == 2 && !store {
                                // (experiment: lines other cores or earlier waves read in this dispatch: GPU-wide L2)
                                lat_extra = m.extra[1];
                            }
                            let from_slc = first.min(need[3]);
                            let from_l2 = (first - from_slc).min(need[2]);
                            need[3] -= from_slc;
                            need[2] -= from_l2;
                            need[4] += from_slc + from_l2;
                        }
                        if store && charge_writeback {
                            need[4] = need[4].max(n.max(1.0)); // every written line goes back to DRAM
                        }
                        let mut until = 0i64;
                        for i in 2..5 {
                            let want = need[i].min(m.cap[i]);
                            if need[i] > 0.0 && m.tok[i] < want {
                                until = until.max(((want - m.tok[i]) / m.rate[i]).ceil() as i64);
                            }
                        }
                        if until > 0 {
                            wake!(t + until.max(1));
                            continue;
                        }
                        mem_charge = Some((need, lat_extra, segs.clone()));
                    }
                }
            }
            // register-file writeback: a write that evicts a register from the group's result cache needs a
            // writeback slot (core: ~2 per reference cycle; 9-16 rotating destinations at 32 groups)
            let mut evict = 0usize;
            if matches!(r.kind, Kind::Alu | Kind::Load) && !r.dst.is_empty() && !msk {
                // counted on a shadow cache that evicts a random one of its 3 oldest entries (9 / 10 / 12 / 16
                // rotating destinations evict on 60 / 68 / 87 / 100% of writes; LRU would evict on all)
                let fresh = r.dst_regs.iter().filter(|reg| !wbc[gi].contains(reg)).count();
                evict = (wbc[gi].len() + fresh).saturating_sub(nc);
                if evict > 0 && wb_tok < evict as f64 {
                    ff_ok = false;
                    wake!(t + (((evict as f64 - wb_tok) / wb_rate).ceil() as i64).max(1));
                    continue;
                }
            }
            // ---- issue at tick t
            wb_tok -= evict as f64;
            if matches!(r.kind, Kind::Alu | Kind::Load) && !msk {
                for &reg in &r.dst_regs {
                    // recency order (oldest first); a miss evicts a random one of the 3 oldest entries
                    if let Some(pos) = wbc[gi].iter().position(|&x| x == reg) {
                        wbc[gi].remove(pos);
                    } else if wbc[gi].len() >= nc {
                        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let v = ((lcg >> 33) as usize) % 3.min(wbc[gi].len());
                        wbc[gi].remove(v);
                    }
                    wbc[gi].push(reg);
                }
            }
            issue_tok -= t_;
            let mut pipe_extra = 0i64;
            if let (Some(m), Some((_, mm)), false) = (ms.as_mut(), mem, msk) {
                if (r.tg_name && r.barrier == 0) || r.atomic {
                    let mv = memv[gi];
                    while m.cur[gi] < mv.len() && (mv[m.cur[gi]].0 as usize) < nxt[gi] {
                        m.cur[gi] += 1;
                    }
                    if m.cur[gi] < mv.len() && mv[m.cur[gi]].0 as usize == nxt[gi] {
                        let words = &mv[m.cur[gi]].1;
                        let k = REAL_CYCLES_PER_UNIT;
                        let ret_extra = if r.atomic && r.kind == Kind::Load { tk(mm.atomic_ret_extra * k) } else { 0 };
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
                            // bank conflicts: 32 banks of 4 bytes; lanes on the same word share one access, except
                            // for threadgroup atomics, which serialise per lane (one word for all lanes: 32 cycles)
                            let tga = r.kind == Kind::TgAtomic;
                            let mut per_bank = [0usize; 32];
                            if tga {
                                for &w in words {
                                    per_bank[(w % 32) as usize] += 1;
                                }
                            } else {
                                let mut ws = words.clone();
                                ws.sort_unstable();
                                ws.dedup();
                                for &w in &ws {
                                    per_bank[(w % 32) as usize] += 1;
                                }
                            }
                            let degree = per_bank.iter().copied().max().unwrap_or(1).max(1);
                            // the group's own threadgroup accesses are serial
                            let base = if tga {
                                if r.dst.is_empty() { mm.tg_group_atomic } else { mm.tg_group_atomic_ret }
                            } else if r.kind == Kind::Store {
                                mm.tg_group_store
                            } else {
                                mm.tg_group_load
                            };
                            let own = base + (degree - 1) as f64;
                            let gstart = t.max(m.tg_group_free[gi]);
                            m.tg_group_free[gi] = gstart + tk(own * k);
                            let gextra = m.tg_group_free[gi] - t - tk(own * k);
                            let per = if !tga && r.kind == Kind::Load { mm.tg_core_load } else { 1.0 };
                            (tk(degree as f64 * per * k), tk(k) - gextra.max(0), &mut m.tg_free)
                        };
                        let start = t.max(*free);
                        *free = start + occ;
                        pipe_extra = (*free - t - occ_min).max(0) + ret_extra;
                    }
                }
            }
            let mut inflight_extra = 0i64;
            if let (Some(m), Some((need, lx, segs))) = (ms.as_mut(), mem_charge.as_ref()) {
                // hardware: cores and groups asking for a line that is still on its way each wait for it (128 groups
                // summing the same two columns as fast as their own: playground novel-results/2026-11-01-first-touch-chains)
                if r.kind == Kind::Load && inflight_on {
                    if let Some(&ready) = segs.iter().filter_map(|sg| m.inflight.get(sg)).max() {
                        inflight_extra = (ready - t).max(0);
                    }
                }
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
                if r.kind == Kind::Load && inflight_on && *lx > 0 {
                    let done = t + lx + pipe_extra;
                    for &sg in segs {
                        let e = m.inflight.entry(sg).or_insert(0);
                        *e = (*e).max(done);
                    }
                }
                for &sg in segs {
                    m.l1.touch(sg);
                    m.l2.touch(sg);
                    m.slc.touch(sg);
                }
                m.epoch += 1;
            }
            let mem_extra = (mem_charge.as_ref().map(|c| c.1).unwrap_or(0) + pipe_extra).max(inflight_extra);
            if r.unit.is_some() && !msk {
                let b = r.bucket;
                tok[b] = Some(tok[b].unwrap() - r.cost);
                if !in_refill[b] && tok[b] != Some(buckets[b].capmax) {
                    in_refill[b] = true;
                    refilling.push(b);
                }
                port[half] -= need;
            }
            let mut branch = 0i64;
            let mut fwd_taken = false;
            let ii: i64;
            if msk {
                // empty execution mask (constants above): no unit, no result, no memory access
                ii = match r.masked_cls {
                    1 => masked_alu,
                    3 => masked_store,
                    _ => t_,
                };
                if r.masked_cls == 2 || r.masked_cls == 4 {
                    let done = t + if r.masked_cls == 2 { masked_load } else { masked_tg_load };
                    slots[gi][r.slot] = Some(slots[gi][r.slot].map_or(done, |m| m.max(done)));
                }
            } else {
            match r.kind {
                Kind::Alu => {
                    let mut lat = r.lat;
                    if shfl_pipe > 0 && !msk && (r.shuffle || (r.fp32 && t - last_shfl < shfl_recent)) {
                        pipe_free = pipe_free.max(t) + shfl_pipe;
                        if r.shuffle {
                            last_shfl = t;
                        }
                    }
                    if r.shuffle && shfl_queue > 0 && !msk {
                        shfl_inflight.retain(|&(_, d)| d > t);
                        // other SIMD-groups' shuffles (a group's own independent ones pipeline: instruction bench)
                        lat += shfl_queue * shfl_inflight.iter().filter(|&&(g2, _)| g2 != gi).count() as i64;
                        shfl_inflight.push((gi, t + lat));
                    }
                    let mut iiv = r.ii;
                    if let Some(h) = r.ii_cache_hit {
                        if r.cache_regs.iter().all(|x| rcache[gi].contains(x)) {
                            iiv = h;
                        }
                    }
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
                    // the load's issue touched its destinations (`present` is never cleared), so they are indexed
                    // directly here
                    // Settled entries (pushed at least once) always move to t + lat here; they are merged per
                    // (slot, instruction) with the largest lat, which gives the same maxima as one push each.
                    // a threadgroup load whose result this instruction reads has completed (the instruction issues with its
                    // sources ready): ALU work no longer defers it. Without this, loads never followed by a `wait`
                    // (threadgroup loads the compiler reads directly) were deferred again by every later ALU
                    // instruction for the rest of the program: a loop of FMAs on values loaded before it paid ~21 units
                    // per FMA (the GPU: ~4 cycles; playground 2026-11-01-fma-accumulator-chains). AGX_TG_SETTLE_ON_READ=1
                    // only, for now: it fixes those (Geekbench's gemm_block_size_bias +43% -> -2%) but llama.cpp's
                    // shared-memory reductions then come out 13-27% fast, an undercount the deferral had offset
                    // (the 40 captures' mean error 10.5% -> 12.1%)
                    if settle_on_read && !r.src.is_empty() && !(pending[gi].is_empty() && settled[gi].is_empty()) {
                        // (threadgroup loads only: device loads are followed by a `wait`, which settles them)
                        let reads = |ins: usize| sim[ins].tgop && sim[ins].dst.iter().any(|h| r.src.contains(h));
                        pending[gi].retain(|p| !reads(p.ins));
                        settled[gi].retain(|a| !reads(a.ins));
                    }
                    let (rg, sl) = (&mut ready[gi], &mut slots[gi]);
                    last_push[gi] = t;
                    for a in settled[gi].iter() {
                        let newdone = t + a.lat;
                        for &h in &sim[a.ins].dst {
                            let x = &mut rg[h as usize];
                            *x = (*x).max(newdone);
                        }
                        let m = &mut sl[a.slot];
                        *m = Some(m.map_or(newdone, |m| m.max(newdone)));
                    }
                    let mut k = 0;
                    while k < pending[gi].len() {
                        let p = &pending[gi][k];
                        let newdone = t + p.lat;
                        if newdone > p.done {
                            for &h in &sim[p.ins].dst {
                                let x = &mut rg[h as usize];
                                *x = (*x).max(newdone);
                            }
                            let m = &mut sl[p.slot];
                            *m = Some(m.map_or(newdone, |m| m.max(newdone)));
                            let p = pending[gi].swap_remove(k);
                            match settled[gi].iter_mut().find(|a| a.slot == p.slot && a.ins == p.ins) {
                                Some(a) => {
                                    a.lat = a.lat.max(p.lat);
                                    a.lat_less_extra = a.lat_less_extra.max(p.lat - p.extra);
                                }
                                None => settled[gi].push(Settled { slot: p.slot, ins: p.ins, lat: p.lat, lat_less_extra: p.lat - p.extra }),
                            }
                        } else {
                            k += 1;
                        }
                    }
                }
                Kind::Load => {
                    let tgop = r.tgop;
                    let q = if tgop { quiet_tg } else { quiet_load };
                    let mut delay = (last_alu[gi] + q - t).max(0);
                    let lat;
                    let mut iiv;
                    if tgop {
                        lat = delay + tg_lat0 + mem_extra;
                        iiv = match tg_block { Some(b) => delay + b + if r.src.is_empty() { 0 } else { tg_block_reg }, None => tg_issue0 };
                    } else {
                        if alu_since_mem[gi] && !(pending[gi].is_empty() && settled[gi].is_empty()) {
                            // waits until the pending loads have passed the L1, not until they complete
                            // (ALU-interleaved loads over L2/DRAM windows: ~50 cycles apart, not a full latency)
                            let m = pending[gi].iter().map(|p| p.done - p.extra)
                                .chain(settled[gi].iter().map(|a| last_push[gi] + a.lat_less_extra)).max().unwrap();
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
                    // completes through the scoreboard like a load (a `wait` waits for it)
                    let done = t + if ms.is_some() { tg_lat0 } else { load_latency } + mem_extra;
                    for &h in &r.dst {
                        let i = touch(&mut ready[gi], &mut present[gi], &mut present_list[gi], h);
                        ready[gi][i] = done;
                    }
                    if ms.is_some() {
                        slots[gi][r.slot] = Some(slots[gi][r.slot].map_or(done, |m| m.max(done)));
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
                    settled[gi].retain(|a| a.slot != s);
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
                    } else if r.is_jmp && nxt[gi] + 1 < npcs[gi] && seqs_pcs[gi][nxt[gi] + 1] > r.pc + r.size {
                        branch = br_fwd; // taken forward jump (if-skip / out-of-line entry): measured 2026-10-20
                        fwd_taken = true;
                    } else if r.is_call && nxt[gi] + 1 < npcs[gi] && seqs_pcs[gi][nxt[gi] + 1] != r.pc + r.size {
                        branch = br_fwd; // as a taken forward jump until measured otherwise
                        fwd_taken = true;
                    }
                }
                Kind::Barrier => {
                    ii = match r.fence {
                        Some(code) => {
                            let (v, consumed) = fence_ticks(code, last_mem[gi] as usize);
                            if consumed {
                                last_mem[gi] = 0;
                            }
                            v
                        }
                        None => t_,
                    };
                    if r.barrier != 0 {
                        bar_count[gi] += 1;
                        bar_arrive[gi] = None;
                    }
                }
            }
            }
            if matches!(r.kind, Kind::Alu | Kind::Load) && !r.dst.is_empty() && !msk {
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
            if r.dev_ctx != 0 && !msk {
                last_mem[gi] = r.dev_ctx;
            }
            let occ = t_.max(ii) + branch;
            // a taken forward jump also holds its scheduler (throughput, not hidden by other groups; 2026-10-20 probes)
            let fwd_hold = if fwd_taken { br_fwd_sched } else { 0 };
            if r.barrier == 0 {
                mask_stale = true;
                sched_free[sch] = t + fwd_hold + if r.kind == Kind::Alu && !r.reduce {
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
            if let Some(p) = issue_profile.as_mut() {
                if gi == 0 {
                    let pc = sim[seqs[gi][nxt[gi]]].pc;
                    let x = p.entry(pc).or_insert((0, 0));
                    x.0 += t - profile_prev;
                    x.1 += 1;
                    profile_prev = t;
                }
            }
            free[gi] = t + occ;
            // the next instruction's line: fetched by the first group to reach it, which waits, and the rest wait as long
            if fetch_line > 0 && nxt[gi] + 1 < npcs[gi] {
                let (pc0, pc1) = (seqs_pcs[gi][nxt[gi]], seqs_pcs[gi][nxt[gi] + 1]);
                let line = pc1 >> FETCH_LINE_SHIFT;
                if let Some(ic) = icache.as_mut() {
                    if line != pc0 >> FETCH_LINE_SHIFT {
                        // a line still in the cache is ready (or arrives when another group's fetch of it ends)
                        let fall_through = pc1 == pc0 + sim[seqs[gi][nxt[gi]]].size;
                        let ready = if ic.contains(line as u64) {
                            let mut x = fetched.get(&line).copied().unwrap_or(0);
                            if let (Some(j), false) = (jumps.as_mut(), fall_through) {
                                if !j.contains(pc1 as u64) {
                                    x = x.max(free[gi] + jump_extra);
                                }
                            }
                            x
                        } else {
                            // (a jump taken to the next instruction is a jump: hw 324 ns per block when it starts the next line)
                            let x = free[gi] + if fall_through { fetch_line } else { fetch_jump };
                            fetched.insert(line, x);
                            x
                        };
                        ic.touch(line as u64);
                        if let (Some(j), false) = (jumps.as_mut(), fall_through) {
                            j.touch(pc1 as u64);
                        }
                        free[gi] = free[gi].max(ready);
                    }
                } else {
                    let ready = *fetched.entry(line).or_insert_with(|| {
                        fetch_free = fetch_free.max(free[gi]) + fetch_line;
                        fetch_free
                    });
                    free[gi] = free[gi].max(ready);
                }
            }
            last_t = last_t.max(free[gi]);
            nxt[gi] += 1;
            sr_cache[gi] = None;
            unblocked[gi] = false;
            blk_cache[gi] = None;
            fresh_cache[gi] = None;
            issued += 1;
            bar_ver += 1;
            tick_issued[gi] = true;
            issued_now.push(gi);
            elig[gi / 64] &= !(1 << (gi % 64));
            if nxt[gi] >= npcs[gi] {
                done += 1;
            } else {
                busy.push(std::cmp::Reverse((free[gi], gi)));
            }
        }
        }
         if done >= g {
            break;
        }
        let wmin = match (busy.peek().map(|r| r.0 .0), wk_min) {
            (Some(b), a) => Some(a.min(b)),
            (None, i64::MAX) => None,
            (None, a) => Some(a),
        };
        let nt = match wmin {
            Some(m) if issued == 0 => (t + 1).max(m),
            _ => t + 1,
        };
        // refill(nt - t). Python min(a, b) is `b if b < a else a`; a bucket already at its cap stays
        // there (cap + non-negative refill, clamped), so only buckets below their cap are visited.
        // A tick that issued nothing and whose groups all stopped at a time threshold or at a resource that only
        // refills (issue slots, unit tokens, register ports) repeats unchanged until a threshold passes or a
        // resource suffices. Those ticks are stepped here with the same per-tick refill, without visiting groups.
        let ff = issued == 0 && ff_ok && !retest.is_empty();
        if ff {
            for &f in &sched_free {
                if f > t {
                    wmin_time = wmin_time.min(f);
                }
            }
            wmin_time = wmin_time.min(wk_gt);
            if let Some(r) = busy.peek() {
                wmin_time = wmin_time.min(r.0 .0);
            }
        }
        let mut nt = nt;
        loop {
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
            wb_tok = (wb_tok + wb_rate * dt).min(WRITEBACK_CAP);
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
            if !ff || t >= wmin_time {
                break;
            }
            let pass = retest.iter().any(|x| match *x {
                Retest::Issue => issue_tok >= t_,
                Retest::Bucket(b, cost) => tok[b].unwrap() >= cost,
                Retest::Port(h, need) => port[h] >= need,
            });
            if pass {
                break;
            }
            nt = t + 1;
        }
    }
    // a dispatch completes only when its device atomics have drained: a no-return atomic's queueing delay is charged to
    // the next atomic or a `wait`, so without this a shader that ends right after its atomics (Mesa's code, e.g. the
    // atomcontend probe) would finish before its serialised atomics (playground 2026-10-16-atomic-contention)
    if let Some(m) = ms.as_ref() {
        last_t = last_t.max(m.at_free);
    }
    let total = last_t as f64 / t_ as f64;
    let finish: Vec<f64> = free.iter().map(|&f| f as f64 / t_ as f64).collect();
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
    if let Some(p) = issue_profile {
        let names: HashMap<usize, &str> = instrs.iter().map(|x| (x.pc, x.name.as_str())).collect();
        let sum: i64 = p.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = p.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1 .0));
        eprintln!("issue profile, SIMD group 0: {sum} ticks before issues");
        let shown = std::env::var("AGX_ISSUE_PROFILE_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
        for (pc, (time, n)) in rows.iter().take(shown) {
            eprintln!("  pc {pc:6} {:<20} {n:6}x {time:10} ({:.1}%)", names.get(pc).unwrap_or(&"?"), *time as f64 / sum.max(1) as f64 * 100.0);
        }
    }
    Ok((total, st, finish))
}

fn m_l1_store_min(mem: Option<(&[&[(u32, Vec<u64>)]], &MemModel)>) -> f64 {
    mem.map(|m| m.1.l1_store_min).unwrap_or(1.0)
}

fn m_l1_load_min(mem: Option<(&[&[(u32, Vec<u64>)]], &MemModel)>) -> f64 {
    mem.map(|m| m.1.l1_load_min).unwrap_or(1.0)
}

/// cores competing for distinct L2/SLC capacity: active cores when data is private, 1 when all share it
fn eff_sharers(m: &MemModel) -> f64 {
    // GPU-wide footprint = |this core| x active / (1 + (active - 1) x overlap); a core's share of a shared
    // cache scales with |this core| / footprint
    (m.active_cores / (1.0 + (m.active_cores - 1.0) * m.sharing)).max(1.0)
}

/// Fraction of the device segments in `a` (one core's traces) that also appear in `b` (another core's).
pub fn segment_overlap(a: &[Trace], b: &[Trace], device_pcs: &std::collections::HashSet<usize>) -> f64 {
    let set = |ts: &[Trace]| {
        let mut s: std::collections::HashSet<u64> = Default::default();
        for t in ts {
            for (step, segs) in &t.mem {
                if device_pcs.contains(&t.pcs[*step as usize]) {
                    s.extend(segs.iter().copied());
                }
            }
        }
        s
    };
    let (sa, sb) = (set(a), set(b));
    if sa.is_empty() {
        return 0.0;
    }
    sa.iter().filter(|x| sb.contains(x)).count() as f64 / sa.len() as f64
}


#[cfg(test)]
mod fence_tests {
    use super::*;

    #[test]
    fn measured_fence_costs_and_context_consumption() {
        // back to back: 85 / 27 / 86 ns at 1.398 cycles per ns, in ticks
        assert_eq!(fence_ticks(0x96, 0), (tk(85.0 * CYCLES_PER_NS), false));
        assert_eq!(fence_ticks(0x33, 0), (tk(27.0 * CYCLES_PER_NS), false));
        // f533 after a load pays the refetch (247 ns) and consumes the context; f596 after a load pays none and keeps it
        assert_eq!(fence_ticks(0x33, 2), (tk(247.0 * CYCLES_PER_NS), true));
        assert_eq!(fence_ticks(0x96, 2), (tk(79.0 * CYCLES_PER_NS), false));
        // f59e is the one that is expensive after a store (306 ns)
        assert_eq!(fence_ticks(0x9e, 1), (tk(306.0 * CYCLES_PER_NS), true));
        assert_eq!(fence_ticks(0x9e, 2), (tk(94.0 * CYCLES_PER_NS), false));
        // an unmeasured encoding costs the default and always consumes
        assert_eq!(fence_ticks(0x77, 0), (tk(FENCE_UNMEASURED_NS * CYCLES_PER_NS), true));
    }
}

/// Cost of one in-shader clock read (`clock()` / `%globaltimer`: `device_barrier_1` + a coherent load of a CPU-written timer line +
/// `wait 0`), in ns, from the playground's clock-load finding (novel-results/2026-10-18-clock-load): the mean of two sweeps of the median
/// per-read cost, by lanes reading at once and by the number of timer-line copies (lane's copy = core & (copies - 1)).
/// The cost is dominated by the load fetching the CPU-written line (~0.5 us) and by contention on one line, so it is a measured table,
/// not derivable from the fence / load model (which predicts ~0.34 us for the same instructions on GPU-resident data).
pub const CLOCK_READ_LANES: [usize; 4] = [1, 32, 1024, 65536];
pub const CLOCK_READ_COPIES: [usize; 5] = [1, 2, 4, 8, 16];
const CLOCK_READ_NS: [[f64; 5]; 4] = [
    [896.0, 804.0, 833.0, 798.0, 795.0],
    [798.0, 839.0, 821.0, 821.0, 771.0],
    [4985.0, 2283.0, 952.0, 565.0, 777.0],
    [37345.0, 20167.0, 4098.0, 2283.0, 2476.0],
];

/// `copies`: timer-line copies the CPU thread maintains (rounded down to a power of two in 1..=16). Between measured lane counts the cost
/// is interpolated linearly in log2(lanes); beyond 65,536 lanes it stays at the last row.
pub fn clock_read_ns(active_lanes: usize, copies: usize) -> f64 {
    let c = CLOCK_READ_COPIES.iter().rposition(|&x| x <= copies.max(1)).unwrap_or(0);
    let l = active_lanes.max(1) as f64;
    let x = l.log2();
    let pts: Vec<(f64, f64)> = CLOCK_READ_LANES.iter().enumerate().map(|(i, &n)| ((n as f64).log2(), CLOCK_READ_NS[i][c])).collect();
    if x <= pts[0].0 {
        return pts[0].1;
    }
    for w in pts.windows(2) {
        if x <= w[1].0 {
            let f = (x - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + f * (w[1].1 - w[0].1);
        }
    }
    pts[pts.len() - 1].1
}

#[cfg(test)]
mod clock_tests {
    use super::*;

    #[test]
    fn clock_read_table_and_interpolation() {
        // measured cells are returned exactly
        assert_eq!(clock_read_ns(1, 1), 896.0);
        assert_eq!(clock_read_ns(1024, 8), 565.0);
        assert_eq!(clock_read_ns(65536, 1), 37345.0);
        // copies round down to a tabulated power of two (3 -> 2), and a single line is far slower under load than per-core copies
        assert_eq!(clock_read_ns(65536, 3), clock_read_ns(65536, 2));
        assert!(clock_read_ns(65536, 1) > 10.0 * clock_read_ns(65536, 16));
        // between 32 and 1,024 lanes (one line) the cost lies between the cells; beyond the table it stays flat
        let mid = clock_read_ns(181, 1);
        assert!(mid > 798.0 && mid < 4985.0);
        assert_eq!(clock_read_ns(1 << 20, 1), 37345.0);
        // one SIMD group never pays contention
        assert!(clock_read_ns(32, 1) < 900.0);
    }
}
