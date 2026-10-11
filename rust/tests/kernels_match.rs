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
