use agx_sim::cost::{read_json, repo_root};
use agx_sim::kernels;

#[test]
fn random_kernels_match_stored_ops() {
    let r = repo_root().join("results/2026-10-11-cost-model");
    for f in ["random-kernels.json", "heldout24.json", "heldout48.json", "heldout32c.json"] {
        let v = read_json(&r.join(f)).unwrap();
        for x in v.as_array().unwrap() {
            let seed = x["seed"].as_u64().unwrap();
            let ops: Vec<String> = x["ops"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
            let (lines, _) = kernels::kernel(seed, ops.len());
            assert_eq!(lines, ops, "{} seed {}", f, seed);
        }
    }
    for f in ["heldout-mem24.json", "heldout-mem24b.json"] {
        let v = read_json(&r.join(f)).unwrap();
        for x in v.as_array().unwrap() {
            let seed = x["seed"].as_u64().unwrap();
            let ops: Vec<String> = x["ops"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
            let (lines, _) = kernels::mem_kernel(seed, 24);
            assert_eq!(lines, ops, "{} seed {}", f, seed);
        }
    }
}

/// The compiled emulator must give the reference interpreter's trace on every verifier workload kind.
#[test]
fn compiled_trace_matches_reference() {
    use agx_sim::cost::{trace, trace_reference};
    use agx_sim::kernels::{bench_loop, counted, OPS, DEFAULT_CONSTS, MEM_CONSTS};
    let mut codes = vec![];
    for (name, _, _) in OPS {
        codes.push(bench_loop(name, 8, 16, 20));
    }
    for seed in 1000..1040 {
        codes.push(counted(&kernels::kernel(seed, 32).1, 30, true, DEFAULT_CONSTS));
    }
    for seed in 7000..7024 {
        codes.push(counted(&kernels::mem_kernel(seed, 24).1, 30, true, MEM_CONSTS));
    }
    for code in &codes {
        for active in [32, 17, 1] {
            let a = trace(code, active, 500_000).unwrap();
            let b = trace_reference(code, active, 500_000).unwrap();
            assert_eq!(a.pcs, b.pcs);
            assert_eq!(a.skipped, b.skipped);
            assert_eq!(a.warnings, b.warnings);
        }
    }
}
