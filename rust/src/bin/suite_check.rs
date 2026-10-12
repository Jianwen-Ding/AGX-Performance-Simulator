//! Offline check of the functional simulator against the hardware-recorded test suite (testsuite/).
//!
//! suite_check [DIR] [-v] [--only CATEGORY] [--case NAME] [--traces]
//! For every case: map the out buffer (harness initial pattern) and the input table at their recorded VAs, set the
//! uniforms (u4 = 0, u5 = count, u6_u7 = out VA, u8_u9 = table VA), run every SIMD group of the grid and compare the
//! deterministic parts word for word. Prints per-category pass counts and, with -v, each failure's first mismatch
//! or the instruction the simulator could not run.
//! Cases whose program runs an fp32 transcendental the simulator does not reproduce bit for bit (log2, exp2, sin_pt_2:
//! the hardware's approximation is not tabulated, see fsem.rs) pass as CLOSE when every differing word is within
//! ULP_TOL ulp of the hardware (words compared as ordered fp32); the summary lists the ulp error per such op.
//! Everything else is compared exactly.
//! --traces instead compares the first workgroup's paths and memory accesses from the functional simulator with the
//! cost model's emulation (`cost::trace_group`, memory read as zero): "mismatch" there means the path depends on data.

use agx_sim::func::{self, Dispatch, Memory};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::Read;

fn b64(s: &str) -> Vec<u8> {
    let t: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    };
    let mut out = vec![];
    for ch in t.chunks(4) {
        let n = ch.iter().take_while(|&&c| c != b'=').count();
        let v = ch.iter().take(n).enumerate().fold(0u32, |a, (i, &c)| a | val(c) << (18 - 6 * i as u32));
        for i in 0..n.saturating_sub(1) {
            out.push((v >> (16 - 8 * i)) as u8);
        }
    }
    out
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn load(p: &std::path::Path) -> Value {
    let mut s = String::new();
    flate2::read::GzDecoder::new(std::fs::File::open(p).unwrap()).read_to_string(&mut s).unwrap();
    serde_json::from_str(&s).unwrap()
}

static TRACES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// --traces: the first workgroup's SIMD-group paths from the functional simulator (real memory) against the cost
/// model's own emulation (memory read as zero); they agree wherever control flow and addresses do not depend on data.
fn compare_traces(d: &Dispatch, mem: &Memory) -> Result<Option<String>, String> {
    let f = func::trace_workgroups(d, mem, &[[0, 0, 0]])?.remove(0);
    for (g, ft) in f.iter().enumerate() {
        let ctx = agx_sim::cost::GroupCtx { wg_size: d.wg, wg_id: [0, 0, 0], group: g as u32, core: 0, slot: g as u32, uniforms: d.uniforms.clone() };
        let ct = agx_sim::cost::trace_group(&d.code, &ctx, 5_000_000)?;
        if let Some(i) = ft.pcs.iter().zip(&ct.pcs).position(|(a, b)| a != b).or((ft.pcs.len() != ct.pcs.len()).then_some(ft.pcs.len().min(ct.pcs.len()))) {
            return Ok(Some(format!("group {} path differs at step {} ({} vs {} steps)", g, i, ft.pcs.len(), ct.pcs.len())));
        }
        if ft.mem != ct.mem {
            let i = ft.mem.iter().zip(&ct.mem).position(|(a, b)| a != b).unwrap_or(ft.mem.len().min(ct.mem.len()));
            return Ok(Some(format!("group {} accesses differ at access {} (pc {:?})", g, i, ft.mem.get(i).map(|m| ft.pcs[m.0 as usize]))));
        }
    }
    Ok(None)
}

/// Largest accepted distance (fp32 ulp) from the hardware for words of a case using an approximated transcendental.
const ULP_TOL: u64 = 2;

/// fp32 transcendentals modelled as correctly rounded, not as the hardware's (untabulated) approximation.
const APPROX: [&str; 3] = ["log2", "exp2", "sin_pt_2"];

/// The approximated transcendentals a program runs (decoded linearly; stops at the first undecodable word).
fn approx_ops(code: &[u8]) -> Vec<&'static str> {
    let mut v = vec![];
    let mut p = 0;
    while p < code.len() {
        let mut buf = code[p..].to_vec();
        buf.resize(buf.len().max(16), 0);
        let Some((d, _, size)) = agx_sim::isa::decode_one(&buf) else { break };
        if size == 0 {
            break;
        }
        if let Some(n) = APPROX.iter().find(|&&n| n == d.name) {
            if !v.contains(n) {
                v.push(*n);
            }
        }
        p += size;
    }
    v
}

/// Distance in fp32 ulp between two words read as floats (sign-magnitude mapped to a monotonic integer line).
fn ulp(a: u32, b: u32) -> u64 {
    let o = |x: u32| if x & 0x8000_0000 != 0 { -((x & 0x7fff_ffff) as i64) } else { x as i64 };
    (o(a) - o(b)).unsigned_abs()
}

#[derive(Default, Clone)]
struct UlpStats {
    words: u64,
    differ: u64,
    le1: u64,
    le2: u64,
    max: u64,
}

enum Verdict {
    Pass,
    /// within ULP_TOL of the hardware: (ops, stats of the compared words)
    Close(String, UlpStats),
    /// mismatch message, plus the ulp stats when the case uses an approximated op
    Mismatch(String, Option<(String, UlpStats)>),
}

/// Ok(Pass / Close / Mismatch); Err = simulator error
fn check(c: &Value) -> Result<Verdict, String> {
    let u = |v: &Value| v.as_u64().unwrap();
    let launch = &c["launch"];
    let dim = |k: &str| -> [u32; 3] {
        let a = launch[k].as_array().unwrap();
        [u(&a[0]) as u32, u(&a[1]) as u32, u(&a[2]) as u32]
    };
    let lay = &c["layout"];
    let (table_va, out_va, out_words) = (u(&lay["table_va"]), u(&lay["out_va"]), u(&lay["out_words"]) as usize);
    let count = launch.get("count").map(u).unwrap_or(u(&launch["grid"][0])) as u32;
    let table = b64(c["input"]["table"].as_str().unwrap());
    let mut out_init = vec![];
    for j in 0..out_words {
        let w: u32 = if j < out_words - 16 { 0xcdcdcdcd } else { 0xa5a50000 + (j - (out_words - 16)) as u32 };
        out_init.extend_from_slice(&w.to_le_bytes());
    }
    let mut mem = Memory::default();
    let tlen = table.len();
    mem.map(out_va, out_init);
    mem.map(table_va, table);
    let d = Dispatch {
        code: hex(c["code"].as_str().unwrap()),
        grid: dim("grid"),
        wg: dim("wg"),
        shared_bytes: u(&launch["shared_bytes"]) as u32,
        uniforms: vec![(4, 0), (5, count), (6, out_va as u32), (7, (out_va >> 32) as u32), (8, table_va as u32), (9, (table_va >> 32) as u32)],
        num_registers: 128,
        max_steps: 50_000_000,
    };
    if TRACES.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(match compare_traces(&d, &mem)? {
            None => Verdict::Pass,
            Some(m) => Verdict::Mismatch(m, None),
        });
    }
    func::run(&d, &mut mem)?;
    let got_out = mem.words(out_va, out_words)?;
    let got_tab = mem.words(table_va, tlen / 4)?;
    let want_out = words(&b64(c["expect"]["out"].as_str().unwrap()));
    let want_tab = words(&b64(c["expect"]["table"].as_str().unwrap()));
    let det = |k: &str| c.get(k).and_then(Value::as_bool).unwrap_or(true);
    // order-dependent words (contended atomics) are not compared: [start, end, step]
    let mut skip = vec![false; want_tab.len()];
    for r in c.get("ignore_table").and_then(Value::as_array).into_iter().flatten() {
        let v: Vec<usize> = r.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let mut i = v[0];
        while i < v[1].min(skip.len()) {
            skip[i] = true;
            i += v[2].max(1);
        }
    }
    let got_tab: Vec<u32> = got_tab.iter().zip(&skip).map(|(&g, &s)| if s { 0 } else { g }).collect();
    let want_tab: Vec<u32> = want_tab.iter().zip(&skip).map(|(&g, &s)| if s { 0 } else { g }).collect();
    let first = |a: &[u32], b: &[u32]| a.iter().zip(b).position(|(x, y)| x != y);
    let mut pairs: Vec<(&[u32], &[u32], &str)> = vec![];
    if det("deterministic_table") {
        pairs.push((&got_tab, &want_tab, "table"));
    }
    if det("deterministic_out") {
        pairs.push((&got_out, &want_out, "out"));
    }
    let mut msg = None;
    let mut st = UlpStats::default();
    for (g, w, what) in &pairs {
        if let Some(i) = first(g, w) {
            let n = g.iter().zip(w.iter()).filter(|(x, y)| x != y).count();
            msg.get_or_insert(format!("{} word {} : got {:#010x}, hardware {:#010x} ({} words differ)", what, i, g[i], w[i], n));
        }
        for (&x, &y) in g.iter().zip(w.iter()) {
            st.words += 1;
            if x != y {
                let u = ulp(x, y);
                st.differ += 1;
                st.le1 += (u <= 1) as u64;
                st.le2 += (u <= 2) as u64;
                st.max = st.max.max(u);
            }
        }
    }
    let Some(m) = msg else { return Ok(Verdict::Pass) };
    let ops = approx_ops(&d.code);
    if ops.is_empty() {
        return Ok(Verdict::Mismatch(m, None));
    }
    let key = ops.join("+");
    Ok(if st.max <= ULP_TOL {
        Verdict::Close(key, st)
    } else {
        Verdict::Mismatch(format!("{}; max {} ulp", m, st.max), Some((key, st)))
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v");
    TRACES.store(args.iter().any(|a| a == "--traces"), std::sync::atomic::Ordering::Relaxed);
    let opt = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let only = opt("--only");
    let one = opt("--case");
    let dir = args.iter().find(|a| !a.starts_with('-') && Some(*a) != only.as_ref() && Some(*a) != one.as_ref()).cloned()
        .unwrap_or_else(|| agx_sim::cost::repo_root().join("testsuite").display().to_string());
    // per category: (exact, close, mismatch, error)
    let mut cats: BTreeMap<String, (u32, u32, u32, u32)> = BTreeMap::new();
    let mut reasons: BTreeMap<String, u32> = BTreeMap::new();
    // per approximated op set: (cases close, cases mismatched, stats over the compared words of those cases)
    let mut ulps: BTreeMap<String, (u32, u32, UlpStats)> = BTreeMap::new();
    let mut add = |k: String, close: bool, s: &UlpStats| {
        let e = ulps.entry(k).or_default();
        if close { e.0 += 1 } else { e.1 += 1 }
        e.2.words += s.words;
        e.2.differ += s.differ;
        e.2.le1 += s.le1;
        e.2.le2 += s.le2;
        e.2.max = e.2.max.max(s.max);
    };
    let mut entries: Vec<_> = walk(std::path::Path::new(&dir));
    entries.sort();
    for p in entries {
        let c = load(&p);
        let cat = c["category"].as_str().unwrap().to_string();
        let name = c["name"].as_str().unwrap().to_string();
        if only.as_ref().map(|o| o != &cat).unwrap_or(false) || one.as_ref().map(|o| o != &name).unwrap_or(false) {
            continue;
        }
        let e = cats.entry(cat.clone()).or_default();
        match check(&c) {
            Ok(Verdict::Pass) => e.0 += 1,
            Ok(Verdict::Close(k, s)) => {
                e.1 += 1;
                if verbose {
                    println!("CLOSE    {}/{}: {} words differ, all within {} ulp (max {}; {})", cat, name, s.differ, ULP_TOL, s.max, k);
                }
                add(k, true, &s);
            }
            Ok(Verdict::Mismatch(m, u)) => {
                e.2 += 1;
                if verbose {
                    println!("MISMATCH {}/{}: {}", cat, name, m);
                }
                if let Some((k, s)) = u {
                    add(k, false, &s);
                }
            }
            Err(m) => {
                e.3 += 1;
                let key = m.split(':').next().unwrap_or("").to_string();
                *reasons.entry(key).or_default() += 1;
                if verbose {
                    println!("ERROR    {}/{}: {}", cat, name, m);
                }
            }
        }
    }
    let (mut p, mut t, mut cl) = (0, 0, 0);
    for (cat, (ok, close, mis, err)) in &cats {
        let n = ok + close + mis + err;
        println!("{:12} pass {:4} / {:4}   close {:4}   mismatch {:4}   unsupported {:4}", cat, ok + close, n, close, mis, err);
        p += ok + close;
        cl += close;
        t += n;
    }
    println!("total        pass {:4} / {:4}   ({} exact, {} within {} ulp)", p, t, p - cl, cl, ULP_TOL);
    if !ulps.is_empty() {
        println!("approximated transcendentals (cases whose words differ; ulp over those words):");
        for (k, (close, mis, s)) in &ulps {
            println!("  {:22} cases close {:3} mismatch {:3}   words {:7} differ {:6} (<=1 ulp {:6}, <=2 ulp {:6})   max {} ulp",
                k, close, mis, s.words, s.differ, s.le1, s.le2, s.max);
        }
    }
    if !reasons.is_empty() {
        println!("unsupported (first error per case): {:?}", reasons);
    }
}

fn walk(d: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v = vec![];
    for e in std::fs::read_dir(d).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            v.extend(walk(&p));
        } else if p.to_string_lossy().ends_with(".json.gz") {
            v.push(p);
        }
    }
    v
}
