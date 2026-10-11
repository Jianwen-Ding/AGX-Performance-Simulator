//! Re-score device-memory measurements (scripts/mem_probes.py) with the memory model: each SIMD group of
//! the one workgroup on a core is traced with its own geometry (`trace_group`), and the core is simulated
//! with `simulate_groups` and the default `MemModel` at the measured clock. Prints measured vs predicted
//! core cycles per memory instruction (reference units).
//!
//! mem_check [-v] [--limit X] FILE.json ...   (exit 1 if any file's median |err| exceeds X)
//! mem_check --stored [--limit X]                 (the files of results/2026-10-11-memory-hierarchy used as checks)

use agx_sim::cost::{self, read_json, to_ref, GroupCtx, MemModel, Params};
use agx_sim::kernels::{self, get_sr, mov};

pub fn predict(p: &Params, row: &serde_json::Value, steps: [u32; 2], mm: &MemModel) -> f64 {
    let g = |k: &str| row[k].as_u64().unwrap_or(0) as u32;
    let groups = g("groups");
    let steps = if row.get("barrier").is_some() { [8, 32] } else { steps };
    let wb = g("window_bytes");
    let mask = if wb > 0 { wb / 4 - 1 } else { 0 };
    let store = row["kind"].as_str() == Some("store");
    let glob = row["global"].as_bool().unwrap_or(false);
    let count = 38 * 32 * groups;
    let data = (64 + 7 * count).div_ceil(4096) * 4096;
    let trips = row["trips"].as_u64().unwrap_or(8) as u32;
    let mut cyc = [0.0; 2];
    for (i, &s) in steps.iter().enumerate() {
        let mut code = get_sr(53, 30);
        code.extend(mov(29, trips));
        let kind = row["kind"].as_str().unwrap_or("load");
        if let Some(bar) = row["barrier"].as_str() {
            code.extend(kernels::barrier_loop(s, bar == "0x78", row["filler_iadds"].as_u64().unwrap_or(0) as u32));
        } else if matches!(kind, "tgload" | "tgstore" | "atomic") {
            let (st, d) = if kind == "atomic" { (g("stride"), data) } else { (4 * g("stride"), 0) };
            code.extend(kernels::mem_loop_kind(s, kind, st, g("j"), g("w"), d, glob));
        } else {
        let private = row["private"].as_bool().unwrap_or(false);
        let w = if private { u32::MAX } else { g("w") };
        code.extend(kernels::mem_loop_pad(s, store, g("stride"), g("j"), w, data, g("advance"), mask, glob || private, g("pad"),
            row["pad_kind"].as_str().unwrap_or("indep")));
        }
        let ins = cost::decode(&code, p).unwrap();
        let traces: Vec<cost::Trace> = (0..groups)
            .map(|k| {
                let ctx = GroupCtx { wg_size: [32 * groups, 1, 1], wg_id: [0, 0, 0], group: k, core: 0, slot: k, uniforms: vec![] };
                cost::trace_group(&code, &ctx, 10_000_000).unwrap()
            })
            .collect();
        let seqs: Vec<&[usize]> = traces.iter().map(|t| t.pcs.as_slice()).collect();
        let mems: Vec<&[(u32, Vec<u64>)]> = traces.iter().map(|t| t.mem.as_slice()).collect();
        cyc[i] = cost::simulate_groups(&ins, &seqs, Some((&mems, mm)), p, false, None).unwrap().0;
    }
    to_ref(cyc[1] - cyc[0]) / (trips * (steps[1] - steps[0])) as f64
}

fn main() {
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v");
    let limit: Option<f64> = args.iter().position(|a| a == "--limit").map(|i| args[i + 1].parse().unwrap());
    let mut files: Vec<String> = args.iter().enumerate()
        .filter(|(i, a)| !a.starts_with('-') && !(*i > 0 && args[i - 1] == "--limit")).map(|(_, a)| a.clone()).collect();
    if args.iter().any(|a| a == "--stored") {
        let r = cost::repo_root().join("results/2026-10-11-memory-hierarchy");
        for f in ["coalesce-shared.json", "coalesce-private.json", "store.json", "pad.json", "private.json", "stream-dram.json", "tg.json",
            "tgstore.json", "atomic-global.json", "barrier.json"] {
            files.push(r.join(f).display().to_string());
        }
    }
    let mut fails = 0;
    for f in files.iter() {
        let d = read_json(std::path::Path::new(f)).unwrap();
        let st: Vec<u32> = d["steps"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut errs = vec![];
        for r in d["rows"].as_array().unwrap() {
            let Some(meas) = r["core_cycles_per_step"].as_f64() else { continue };
            let mm = MemModel { clock_mhz: r["mhz"].as_f64().unwrap_or(1398.0) * cost::REAL_CYCLES_PER_UNIT, warm: false, ..Default::default() };
            let mm = MemModel { l2_extra: std::env::var("AGX_L2_EXTRA").ok().and_then(|v| v.parse().ok()).unwrap_or(mm.l2_extra), ..mm };
            let pred = predict(&p, r, [st[0], st[1]], &mm);
            let e = (pred - meas) / meas;
            errs.push(e.abs());
            if verbose {
                println!("{} S {:2} win {:9} G {:2}: measured {:8.2} predicted {:8.2} ({:+.0}%)", r["kind"].as_str().or(r["barrier"].as_str()).unwrap_or("?"),
                    r["stride"], r["window_bytes"], r["groups"], meas, pred, 100.0 * e);
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |x: f64| errs[((errs.len() - 1) as f64 * x) as usize];
        let ok = limit.map(|l| q(0.5) <= l).unwrap_or(true);
        fails += !ok as i32;
        println!("{}{}: {} rows, median |err| {:.3}, p90 {:.3}, max {:.3}", if ok { "ok   " } else { "FAIL " }, f, errs.len(), q(0.5), q(0.9), q(1.0));
    }
    println!("{} failures", fails);
    std::process::exit(if fails > 0 { 1 } else { 0 });
}
