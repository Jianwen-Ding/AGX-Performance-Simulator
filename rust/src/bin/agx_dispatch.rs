//! Predict the time of one compute dispatch on the whole GPU (see `agx_sim::dispatch`).
//!
//! agx_dispatch code.bin --grid X[,Y,Z] [--wg x[,y,z]] [--regs HALVES] [--tg-bytes N] [--clock MHZ]
//!              [--cores N] [--launch-us US] [--uniform IDX=VALUE ...] [--no-mem] [--max-waves N]
//!
//! code.bin is the main shader as submitted (e.g. the direct-DRM harness's control main with a body).
//! Uniforms are 32-bit uniform registers (u4 = 4); the harness uses u4 = 0, u5 = count, u6_u7 = output,
//! u8_u9 = table. Prints one JSON object.

use agx_sim::cost::{self, Params};
use agx_sim::dispatch::{self, Launch};

fn dims(s: &str) -> [u32; 3] {
    let v: Vec<u32> = s.split(',').map(|x| x.parse().expect("dimension")).collect();
    [v[0], *v.get(1).unwrap_or(&1), *v.get(2).unwrap_or(&1)]
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() {
        eprintln!("usage: agx_dispatch code.bin --grid X[,Y,Z] [--wg x,y,z] [--regs H] [--tg-bytes N] [--clock MHZ] [--uniform I=V] [--no-mem]");
        std::process::exit(2);
    }
    let mut code = std::fs::read(&a[0]).expect("code.bin");
    while code.last() == Some(&0) {
        code.pop(); // trailing zero padding (the harness appends 16 bytes)
    }
    let mut l = Launch::default();
    let mut i = 1;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--grid" => l.grid = dims(&v),
            "--wg" => l.wg = dims(&v),
            "--regs" => l.regs = v.parse().unwrap(),
            "--tg-bytes" => l.tg_bytes = v.parse().unwrap(),
            "--clock" => l.clock_mhz = v.parse().unwrap(),
            "--cores" => l.cores = v.parse().unwrap(),
            "--launch-us" => l.launch_ns = v.parse::<f64>().unwrap() * 1000.0,
            "--max-waves" => l.max_waves = v.parse().unwrap(),
            "--uniform" => {
                let (k, x) = v.split_once('=').expect("IDX=VALUE");
                l.uniforms.push((k.trim_start_matches('u').parse().unwrap(), parse_u32(x)));
            }
            "--no-mem" => {
                l.mem = false;
                i += 1;
                continue;
            }
            x => panic!("unknown argument {}", x),
        }
        i += 2;
    }
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    match dispatch::simulate(&code, &l, &p) {
        Ok(r) => println!(
            "{}",
            serde_json::json!({
                "time_us": r.time_ns / 1000.0, "core_cycles": r.core_cycles, "clock_mhz": l.clock_mhz,
                "workgroups": r.workgroups, "groups_per_wg": r.groups_per_wg, "group_cap_per_core": r.group_cap_per_core,
                "wgs_per_core": r.wgs_per_core, "launch_core_us": r.launch_core_ns / 1000.0, "busiest_core_wgs": r.busiest_core_wgs, "waves": r.waves,
                "simulated_waves": r.simulated_waves, "launch_us": l.launch_ns / 1000.0, "warnings": r.warnings,
            })
        ),
        Err(e) => {
            eprintln!("agx_dispatch: {}", e);
            std::process::exit(1);
        }
    }
}

fn parse_u32(s: &str) -> u32 {
    if let Some(h) = s.strip_prefix("0x") { u32::from_str_radix(h, 16).unwrap() } else { s.parse().unwrap() }
}
