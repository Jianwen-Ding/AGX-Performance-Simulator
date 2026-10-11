//! Timing of the model's three stages (decode, trace, simulate) on the verifier's workloads:
//! every random kernel of the stored datasets in a 500-trip counted loop, 1 and 4 SIMD groups.

use agx_sim::cost::{self, read_json, repo_root, Params};
use agx_sim::kernels::{counted, kernel, mem_kernel, DEFAULT_CONSTS, MEM_CONSTS};
use std::time::Instant;

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    let r = repo_root().join("results/2026-10-11-cost-model");
    let mut codes = vec![];
    for f in ["random-kernels.json", "heldout24.json", "heldout48.json", "heldout32c.json"] {
        for x in read_json(&r.join(f)).unwrap().as_array().unwrap() {
            let (_, body) = kernel(x["seed"].as_u64().unwrap(), x["ops"].as_array().unwrap().len());
            codes.push(counted(&body, 500, true, DEFAULT_CONSTS));
        }
    }
    for f in ["heldout-mem24.json", "heldout-mem24b.json"] {
        for x in read_json(&r.join(f)).unwrap().as_array().unwrap() {
            let (_, body) = mem_kernel(x["seed"].as_u64().unwrap(), 24);
            codes.push(counted(&body, 500, true, MEM_CONSTS));
        }
    }
    let (mut td, mut tt, mut ts1, mut ts4, mut dynamic) = (0.0, 0.0, 0.0, 0.0, 0usize);
    let mut check = 0.0;
    for _ in 0..reps {
        for code in &codes {
            let t0 = Instant::now();
            let ins = cost::decode(code, &p).unwrap();
            let t1 = Instant::now();
            let tr = cost::trace(code, 32, 500_000).unwrap();
            let t2 = Instant::now();
            check += cost::simulate(&ins, &tr.pcs, &p, 1, false).unwrap().0;
            let t3 = Instant::now();
            check += cost::simulate(&ins, &tr.pcs, &p, 4, false).unwrap().0;
            let t4 = Instant::now();
            td += (t1 - t0).as_secs_f64();
            tt += (t2 - t1).as_secs_f64();
            ts1 += (t3 - t2).as_secs_f64();
            ts4 += (t4 - t3).as_secs_f64();
            dynamic += tr.pcs.len();
        }
    }
    let n = (codes.len() * reps) as f64;
    println!("{} kernels x {} reps, {:.0} dynamic instructions each (checksum {})", codes.len(), reps, dynamic as f64 / n, check);
    for (name, t) in [("decode", td), ("trace", tt), ("simulate 1 group", ts1), ("simulate 4 groups", ts4)] {
        println!("{:<18} {:9.3} ms/kernel", name, 1e3 * t / n);
    }
}
