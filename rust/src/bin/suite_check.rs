//! Offline check of the functional simulator against the hardware-recorded test suite (testsuite/).
//!
//! suite_check [DIR] [-v] [--only CATEGORY] [--case NAME]
//! For every case: map the out buffer (harness initial pattern) and the input table at their recorded VAs, set the
//! uniforms (u4 = 0, u5 = count, u6_u7 = out VA, u8_u9 = table VA), run every SIMD group of the grid and compare the
//! deterministic parts word for word. Prints per-category pass counts and, with -v, each failure's first mismatch
//! or the instruction the simulator could not run.

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

/// Ok(None) = pass; Ok(Some(msg)) = mismatch; Err = simulator error
pub fn check(c: &Value) -> Result<Option<String>, String> {
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
    func::run(&d, &mut mem)?;
    let got_out = mem.words(out_va, out_words)?;
    let got_tab = mem.words(table_va, tlen / 4)?;
    let want_out = words(&b64(c["expect"]["out"].as_str().unwrap()));
    let want_tab = words(&b64(c["expect"]["table"].as_str().unwrap()));
    let det = |k: &str| c.get(k).and_then(Value::as_bool).unwrap_or(true);
    let first = |a: &[u32], b: &[u32]| a.iter().zip(b).position(|(x, y)| x != y);
    if det("deterministic_table") {
        if let Some(i) = first(&got_tab, &want_tab) {
            let n = got_tab.iter().zip(&want_tab).filter(|(x, y)| x != y).count();
            return Ok(Some(format!("table word {} : got {:#010x}, hardware {:#010x} ({} words differ)", i, got_tab[i], want_tab[i], n)));
        }
    }
    if det("deterministic_out") {
        if let Some(i) = first(&got_out, &want_out) {
            let n = got_out.iter().zip(&want_out).filter(|(x, y)| x != y).count();
            return Ok(Some(format!("out word {} : got {:#010x}, hardware {:#010x} ({} words differ)", i, got_out[i], want_out[i], n)));
        }
    }
    Ok(None)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v");
    let opt = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let only = opt("--only");
    let one = opt("--case");
    let dir = args.iter().find(|a| !a.starts_with('-') && Some(*a) != only.as_ref() && Some(*a) != one.as_ref()).cloned()
        .unwrap_or_else(|| agx_sim::cost::repo_root().join("testsuite").display().to_string());
    let mut cats: BTreeMap<String, (u32, u32, u32)> = BTreeMap::new();
    let mut reasons: BTreeMap<String, u32> = BTreeMap::new();
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
            Ok(None) => e.0 += 1,
            Ok(Some(m)) => {
                e.1 += 1;
                if verbose {
                    println!("MISMATCH {}/{}: {}", cat, name, m);
                }
            }
            Err(m) => {
                e.2 += 1;
                let key = m.split(':').next().unwrap_or("").to_string();
                *reasons.entry(key).or_default() += 1;
                if verbose {
                    println!("ERROR    {}/{}: {}", cat, name, m);
                }
            }
        }
    }
    let (mut p, mut t) = (0, 0);
    for (cat, (ok, mis, err)) in &cats {
        println!("{:12} pass {:4} / {:4}   mismatch {:4}   unsupported {:4}", cat, ok, ok + mis + err, mis, err);
        p += ok;
        t += ok + mis + err;
    }
    println!("total        pass {:4} / {:4}", p, t);
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
