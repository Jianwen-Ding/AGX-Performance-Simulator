//! The `agx_cost` command line: occupancy is added without moving what was there.

use agx_sim::kernels::e;
use serde_json::Value;
use std::process::Command;

fn run(code: &std::path::Path, args: &[&str]) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_agx_cost")).arg(code).args(args).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_str(String::from_utf8(out.stdout).unwrap().lines().next().unwrap()).unwrap()
}

#[test]
fn occupancy_fields_are_added_and_existing_ones_stay() {
    let dir = std::env::temp_dir().join(format!("agx-cost-occupancy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("code.bin");
    let code = [e("iadd r79, r0, 0"), e("iadd r2, r79, 1"), vec![0x88, 0]].concat();
    std::fs::write(&path, code).unwrap();

    let plain = run(&path, &[]);
    let keys: Vec<&str> = plain.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ["instructions", "dynamic", "groups", "units", "cycles", "emulator_skipped", "warnings", "unmeasured_classes", "occupancy"]);
    let o = &plain["occupancy"];
    assert_eq!((&o["register_source"], &o["register_halves"], &o["registers"]), (&Value::from("inferred"), &Value::from(160), &Value::from(80)));
    assert_eq!(o["resident_simd_groups_per_core"], 20);
    assert_eq!(o["resident_threads_per_core"], 640);
    assert!(plain.get("core_throughput").is_none());

    // --core-throughput changes nothing else
    let with = run(&path, &["--core-throughput"]);
    for k in keys {
        assert_eq!(plain[k], with[k], "{}", k);
    }
    let t = &with["core_throughput"];
    assert_eq!(t["threads_per_wave"], 640);
    assert_eq!(t["validated_on_hardware"], false);
    assert!(t["threads_per_us"].as_f64().unwrap() > 0.0, "{}", t);

    // a declared count wins and adds "paths" as before; 128 registers against 80
    let declared = run(&path, &["--regs", "256", "--core-throughput"]);
    assert_eq!(declared["paths"], "geometry");
    assert_eq!(declared["occupancy"]["register_source"], "declared");
    assert_eq!(declared["occupancy"]["resident_simd_groups_per_core"], 12);
    assert_eq!(declared["occupancy"]["inferred_register_halves"], 160);
    assert_eq!(declared["core_throughput"]["threads_per_wave"], 384);

    // whole workgroups of the launch's size
    let wide = run(&path, &["--regs", "160", "--wg", "64", "--core-throughput"]);
    assert_eq!(wide["occupancy"]["resident_workgroups_per_core"], 10);
    assert_eq!(wide["core_throughput"]["threads_per_wave"], 640);

    // the estimate is in real cycles whatever --units scales
    let refu = run(&path, &["--units", "ref", "--core-throughput"]);
    assert_eq!(refu["core_throughput"], with["core_throughput"]);
    assert_ne!(refu["cycles"], with["cycles"]);

    // no workgroup fits: reported, with the rest intact
    let none = run(&path, &["--regs", "4096", "--core-throughput"]);
    assert_eq!(none["occupancy"]["resident_workgroups_per_core"], 0);
    assert!(none["core_throughput"]["error"].as_str().unwrap().contains("fits"));
    assert!(none["cycles"].is_number());

    // --annotate's table still follows one JSON line
    let out = Command::new(env!("CARGO_BIN_EXE_agx_cost")).arg(&path).arg("--annotate").output().unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.lines().nth(1).unwrap().contains("instruction"));
    std::fs::remove_dir_all(dir).unwrap();
}
