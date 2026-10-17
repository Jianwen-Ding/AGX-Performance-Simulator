//! CLI regression with a synthetic stop-only fixture, not a hardware recording.
use flate2::{write::GzEncoder, Compression};
use serde_json::{json, Value};
use std::{io::Write, process::Command};

#[test]
fn replay_status_rejects_empty_failed_and_incomplete_suites() {
    let dir = std::env::temp_dir().join(format!("agx-suite-exit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let run = |args: &[&str]| Command::new(env!("CARGO_BIN_EXE_suite_check"))
        .arg(&dir).args(args).output().unwrap();
    let write = |c: &Value| {
        let mut gz = GzEncoder::new(std::fs::File::create(dir.join("stop.json.gz")).unwrap(), Compression::default());
        gz.write_all(&serde_json::to_vec(c).unwrap()).unwrap();
        gz.finish().unwrap();
    };
    assert_eq!(run(&[]).status.code(), Some(1));
    let mut c = json!({
        "name": "synthetic-stop", "category": "synthetic", "device": "offline synthetic fixture",
        "code": "8800", "launch": {"grid": [32,1,1], "wg": [32,1,1], "regs": 8, "shared_bytes": 0},
        "layout": {"table_va": 1048576, "out_va": 65536, "out_words": 32},
        "input": {"table": "AAAAAA=="},
        "expect": {"table": "AAAAAA==", "out": "zc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3NzQAApaUBAKWlAgClpQMApaUEAKWlBQClpQYApaUHAKWlCAClpQkApaUKAKWlCwClpQwApaUNAKWlDgClpQ8ApaU="}
    });
    write(&c);
    let ok = run(&[]);
    assert!(ok.status.success(), "{}", String::from_utf8_lossy(&ok.stdout));
    assert!(run(&["--case-prefix", "synthetic-"]).status.success());
    assert_eq!(run(&["--case-prefix", "does-not-exist"]).status.code(), Some(1));
    assert_eq!(run(&["--case", "does-not-exist"]).status.code(), Some(1));
    c["expect"]["table"] = json!("AQAAAA=="); // one wrong word
    write(&c);
    assert_eq!(run(&[]).status.code(), Some(1));
    for (field, value) in [("table", ""), ("out", "AAAAAA==")] {
        let mut short = c.clone();
        short["expect"]["table"] = json!("AAAAAA==");
        short["expect"][field] = json!(value);
        write(&short);
        let r = run(&["-v"]);
        assert_eq!(r.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&r.stdout).contains("length:"));
    }
    c["code"] = json!("ff"); // unsupported instruction
    write(&c);
    assert_eq!(run(&[]).status.code(), Some(1));
    std::fs::remove_dir_all(dir).unwrap();
}
