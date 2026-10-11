//! Re-score core-scaling measurements (scripts/scale_probes.py) with the model: for each row
//! (op, chains, groups, idle = none) the model simulates `groups` SIMD groups running the bench loop
//! with s1 and s2 steps; the slope is core cycles per step (reference units), as measured.
//!
//! scale_check [FILE.json ...]   (default: results/2026-10-11-core-scaling/grid.json)

use agx_sim::cost::{self, repo_root, read_json, to_ref, Params};
use agx_sim::kernels;

fn main() {
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v");
    let files: Vec<String> = args.into_iter().filter(|a| a != "-v").collect();
    let files = if files.is_empty() {
        vec![repo_root().join("results/2026-10-11-core-scaling/grid.json").display().to_string()]
    } else {
        files
    };
    let t_ = 10u32;
    for f in files {
        let d = read_json(std::path::Path::new(&f)).unwrap();
        let steps: Vec<u32> = d["steps"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut errs = vec![];
        for r in d["rows"].as_array().unwrap() {
            if r.get("idle").and_then(|x| x.as_array()).map(|a| !a.is_empty()).unwrap_or(false) {
                continue;
            }
            let Some(meas) = r.get("core_cycles_per_step").and_then(|x| x.as_f64()) else { continue };
            let op = r["op"].as_str().unwrap();
            let ch = r["chains"].as_u64().unwrap() as u32;
            let g = r["groups"].as_u64().unwrap() as usize;
            let sim = |s: u32| {
                let code = kernels::bench_loop(op, ch, s, t_);
                let ins = cost::decode(&code, &p).unwrap();
                let tr = cost::trace(&code, 32, 1_000_000).unwrap();
                cost::simulate(&ins, &tr.pcs, &p, g, false).unwrap().0
            };
            let pred = to_ref(sim(steps[1]) - sim(steps[0])) / (t_ * (steps[1] - steps[0])) as f64;
            let e = (pred - meas) / meas;
            errs.push(e.abs());
            if verbose {
                println!("{:18} ch {} G {:2}: measured {:7.2} predicted {:7.2} ({:+.0}%)", op, ch, g, meas, pred, 100.0 * e);
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |x: f64| errs[((errs.len() - 1) as f64 * x) as usize];
        println!("{}: {} rows, median |err| {:.3}, p90 {:.3}, max {:.3}", f, errs.len(), q(0.5), q(0.9), q(1.0));
    }
}
