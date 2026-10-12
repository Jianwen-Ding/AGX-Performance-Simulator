//! Re-score stored whole-dispatch measurements (scripts/dispatch_probes.py, results/2026-10-11-dispatch)
//! with `agx_sim::dispatch` at 1398 MHz. Exit 1 if the median |error| exceeds the limit (default 0.10).
//!
//! dispatch_check [-v] [--limit X]

use agx_sim::cost::{self, read_json, repo_root, Params};
use agx_sim::dispatch::{self, Launch};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v");
    let limit: f64 = args.iter().position(|a| a == "--limit").map(|i| args[i + 1].parse().unwrap()).unwrap_or(0.10);
    let dir = repo_root().join("results/2026-10-11-dispatch");
    let d = read_json(&dir.join("dispatch.json")).unwrap();
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    let mut errs = vec![];
    let rows = d["rows"].as_array().unwrap();
    let launch = |r: &serde_json::Value| -> (Vec<u8>, Launch) {
        let mut code = std::fs::read(dir.join(r["code"].as_str().unwrap())).unwrap();
        while code.last() == Some(&0) {
            code.pop();
        }
        let g = |k: &str| r[k].as_u64().unwrap() as u32;
        let l = Launch {
            grid: [g("grid"), 1, 1],
            wg: [g("wg"), 1, 1],
            regs: g("regs"),
            tg_bytes: r["tg_bytes"].as_u64().unwrap_or(0) as u32,
            uniforms: vec![(4, 0), (5, g("grid")), (7, 1), (9, 2)],
            ..Default::default()
        };
        (code, l)
    };
    // dispatches are independent: simulate them on every core, biggest grids first
    let preds = agx_sim::par::map_by(rows, |r| r["grid"].as_u64().unwrap_or(0), |r| {
        let (code, l) = launch(r);
        dispatch::simulate(&code, &l, &p).unwrap().time_ns / 1000.0
    });
    for (r, pred) in rows.iter().zip(preds) {
        let g = |k: &str| r[k].as_u64().unwrap() as u32;
        let meas = r["min_us"].as_f64().unwrap();
        let e = (pred - meas) / meas;
        errs.push(e.abs());
        if verbose {
            println!("{:11} grid {:9} wg {:5} regs {:3}: measured {:8.1} us predicted {:8.1} us ({:+.0}%)", r["case"].as_str().unwrap(),
                g("grid"), g("wg"), g("regs"), meas, pred, 100.0 * e);
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |x: f64| errs[((errs.len() - 1) as f64 * x) as usize];
    let ok = q(0.5) <= limit;
    println!("{}dispatch: {} dispatches, median |err| {:.3}, p90 {:.3}, max {:.3} (limit {})", if ok { "ok   " } else { "FAIL " },
        errs.len(), q(0.5), q(0.9), q(1.0), limit);
    std::process::exit(if ok { 0 } else { 1 });
}
