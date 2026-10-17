//! Synthetic input-domain and batch-isolation regression; no GPU is involved.
use serde_json::{json, Value};
use std::process::Command;

#[test]
fn invalid_jobs_do_not_abort_or_corrupt_later_batch_reports() {
    let dir = std::env::temp_dir().join(format!("agx-dispatch-exit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let code = dir.join("stop.bin");
    std::fs::write(&code, [0x88, 0]).unwrap();
    let good = vec![code.to_str().unwrap(), "--grid", "64", "--wg", "32", "--no-mem"];
    let run = |args: &[&str], workers: &str| Command::new(env!("CARGO_BIN_EXE_agx_dispatch"))
        .args(args).env("AGX_THREADS", workers).output().unwrap();
    let before = run(&good, "1");
    assert!(before.status.success(), "{}", String::from_utf8_lossy(&before.stderr));
    let reference: Value = serde_json::from_slice(&before.stdout).unwrap();
    assert_eq!(reference["workgroups"], 2);
    assert_eq!(reference["groups_per_wg"], 1);
    let mut lines = vec![json!(&good).to_string()];
    let mut expected = vec![None];
    for (bad, needle) in [
        (vec!["--wg", "0"], "workgroup"), (vec!["--wg", "65536,65536"], "workgroup"),
        (vec!["--cores", "0"], "core"), (vec!["--regs", "2147483648"], "occupancy"),
        (vec!["--tg-bytes", "65537"], "threadgroup memory"),
        (vec!["--uniform", "u256=1"], "uniform"), (vec!["--uniform", "bad"], "IDX=VALUE"),
        (vec!["--clock", "NaN"], "clock_mhz"), (vec!["--clock", "0"], "clock_mhz"),
        (vec!["--launch-us", "-1"], "launch_ns"), (vec!["--clock"], "requires a value"),
        (vec!["--grid", "1,2,3,4"], "dimensions"), (vec!["--grid", "1,,3"], "dimension"),
        (vec!["--alloc", "0xg=4"], "u64"), (vec!["--alloc", "0x1000=bad"], "u32"),
    ] {
        let mut args = good.clone();
        args.extend(bad);
        let result = run(&args, "1");
        assert_eq!(result.status.code(), Some(1), "{}", String::from_utf8_lossy(&result.stderr));
        assert!(String::from_utf8_lossy(&result.stderr).contains(needle));
        lines.push(json!(args).to_string());
        expected.push(Some(needle));
    }
    lines.push("[]".into()); expected.push(Some("empty job"));
    for text in ["{", "{}", "[1]"] {
        lines.push(text.into()); expected.push(Some("batch line"));
    }
    lines.push(String::new()); // blank lines do not consume a result slot
    lines.push(json!(&good).to_string()); expected.push(None);
    let batch = dir.join("jobs.jsonl");
    std::fs::write(&batch, lines.join("\n")).unwrap();
    for workers in ["1", "4"] {
        let result = run(&["--batch", batch.to_str().unwrap()], workers);
        assert_eq!(result.status.code(), Some(1), "{}", String::from_utf8_lossy(&result.stderr));
        let rows: Vec<Value> = String::from_utf8(result.stdout).unwrap().lines()
            .map(|s| serde_json::from_str(s).unwrap()).collect();
        assert_eq!(rows.len(), expected.len());
        for (row, needle) in rows.iter().zip(&expected) {
            match needle {
                None => assert_eq!(row, &reference),
                Some(text) => assert!(row["error"].as_str().unwrap().contains(text), "{}", row),
            }
        }
    }
    std::fs::write(&batch, " \n\n").unwrap();
    assert_eq!(run(&["--batch", batch.to_str().unwrap()], "1").status.code(), Some(1));
    assert_eq!(run(&["--batch"], "1").status.code(), Some(2));
    // Invalid launch rejection precedes shader/preamble/buffer I/O.
    let missing = dir.join("missing");
    let result = run(&[missing.to_str().unwrap(), "--wg", "0"], "1");
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&result.stderr).contains("workgroup"));
    assert_eq!(run(&good, "1").stdout, before.stdout);
    std::fs::remove_dir_all(dir).unwrap();
}
