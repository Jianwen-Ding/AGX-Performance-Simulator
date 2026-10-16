//! Predict the time of one compute dispatch on the whole GPU (see `agx_sim::dispatch`).
//!
//! agx_dispatch code.bin --grid X[,Y,Z] [--wg x[,y,z]] [--regs HALVES] [--tg-bytes N] [--clock MHZ]
//!              [--cores N] [--launch-us US] [--uniform IDX=VALUE ...] [--no-mem] [--max-waves N]
//!              [--preamble PRE.bin] [--map VA=FILE ...] [--alloc VA=BYTES ...]
//!
//! code.bin is the main shader as submitted (e.g. the direct-DRM harness's control main with a body).
//! Uniforms are 32-bit uniform registers (u4 = 4); the harness uses u4 = 0, u5 = count, u6_u7 = output,
//! u8_u9 = table. Prints one JSON object.
//!
//! With --preamble, --map or --alloc (as for agx_run) the SIMD groups' paths come from the functional simulator
//! over that memory (`func::trace_workgroups`) instead of the cost model's emulation, which reads memory as zero:
//! needed for driver-compiled code (the preamble fills the uniforms from the root table) and data-dependent loops.
//!
//! agx_dispatch --batch JOBS: JOBS has one dispatch per line, as a JSON array of the arguments above. The model's
//! parameters are loaded once, the dispatches run in parallel, and one JSON object per line is printed in order
//! (`{"error": ...}` for a dispatch that failed; the exit status is then 1). Each result is the one the single form
//! prints.

use agx_sim::cost::{self, Params};
use agx_sim::dispatch::{self, Launch};
use agx_sim::func;

fn dims(s: &str) -> [u32; 3] {
    let v: Vec<u32> = s.split(',').map(|x| x.parse().expect("dimension")).collect();
    [v[0], *v.get(1).unwrap_or(&1), *v.get(2).unwrap_or(&1)]
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() {
        eprintln!("usage: agx_dispatch code.bin --grid X[,Y,Z] [--wg x,y,z] [--regs H] [--tg-bytes N] [--clock MHZ] [--uniform I=V] [--no-mem]");
        eprintln!("       agx_dispatch --batch JOBS   (one JSON array of those arguments per line)");
        std::process::exit(2);
    }
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    if a[0] == "--batch" {
        let text = std::fs::read_to_string(a.get(1).expect("--batch JOBS")).expect("jobs file");
        let jobs: Vec<Vec<String>> = text.lines().filter(|x| !x.trim().is_empty())
            .map(|x| serde_json::from_str(x).expect("a job is a JSON array of strings")).collect();
        // one dispatch per core when there are enough of them, each simulating on its own thread; else one at a time,
        // each with the cores to itself
        let outer = jobs.len() >= agx_sim::par::threads();
        let run = |job: &Vec<String>| {
            if outer {
                agx_sim::par::set_local_threads(1);
            }
            run_job(job, &p)
        };
        let res: Vec<Result<serde_json::Value, String>> =
            if outer { agx_sim::par::map(&jobs, run) } else { jobs.iter().map(run).collect() };
        let mut bad = false;
        for r in res {
            match r {
                Ok(v) => println!("{}", v),
                Err(e) => {
                    bad = true;
                    println!("{}", serde_json::json!({ "error": e }));
                }
            }
        }
        std::process::exit(if bad { 1 } else { 0 });
    }
    match run_job(&a, &p) {
        Ok(v) => println!("{}", v),
        Err(e) => {
            eprintln!("agx_dispatch: {}", e);
            std::process::exit(1);
        }
    }
}

/// One dispatch: its arguments as on the command line, the JSON result.
fn run_job(a: &[String], p: &Params) -> Result<serde_json::Value, String> {
    let mut code = std::fs::read(&a[0]).map_err(|e| format!("{}: {}", a[0], e))?;
    while code.last() == Some(&0) {
        code.pop(); // trailing zero padding (the harness appends 16 bytes)
    }
    let mut l = Launch::default();
    let (mut preamble, mut mem, mut functional) = (None, func::Memory::default(), false);
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
            "--preamble" => {
                preamble = Some(std::fs::read(&v).map_err(|e| format!("{}: {}", v, e))?);
                functional = true;
            }
            "--map" | "--alloc" => {
                let (va, x) = v.split_once('=').expect("VA=FILE or VA=BYTES");
                let bytes = if a[i] == "--map" { std::fs::read(x).map_err(|e| format!("{}: {}", x, e))? } else { vec![0u8; parse_u32(x) as usize] };
                mem.map(parse_u64(va), bytes);
                functional = true;
            }
            "--no-mem" => {
                l.mem = false;
                i += 1;
                continue;
            }
            x => return Err(format!("unknown argument {}", x)),
        }
        i += 2;
    }
    let d = if functional {
        if let Some(pre) = &preamble {
            l.uniforms = func::run_preamble(pre, &l.uniforms, &mut mem, 128).map_err(|e| format!("preamble: {}", e))?;
        }
        Some(func::Dispatch { code: code.clone(), grid: l.grid, wg: l.wg, shared_bytes: l.tg_bytes, uniforms: l.uniforms.clone(),
                              num_registers: 128, max_steps: l.max_steps as u64, placement: 0,
                              launch: agx_sim::cost::SrLaunch { tg_bytes: l.tg_bytes, regs: l.regs, ..Default::default() } })
    } else {
        None
    };
    let tracer = |wg: [u32; 3]| -> Result<Vec<cost::Trace>, String> {
        func::trace_workgroups(d.as_ref().unwrap(), &mem, &[wg]).map(|mut v| v.remove(0))
    };
    let r = dispatch::simulate_with(&code, &l, p, if functional { Some(&tracer) } else { None })?;
    Ok(serde_json::json!({
        "time_us": r.time_ns / 1000.0, "core_cycles": r.core_cycles, "clock_mhz": l.clock_mhz,
        "workgroups": r.workgroups, "groups_per_wg": r.groups_per_wg, "group_cap_per_core": r.group_cap_per_core,
        "wgs_per_core": r.wgs_per_core, "launch_core_us": r.launch_core_ns / 1000.0, "busiest_core_wgs": r.busiest_core_wgs, "waves": r.waves,
        "simulated_waves": r.simulated_waves, "launch_us": l.launch_ns / 1000.0, "warnings": r.warnings,
        "paths": if functional { "functional" } else { "cost-emulation" },
    }))
}

fn parse_u64(s: &str) -> u64 {
    if let Some(h) = s.strip_prefix("0x") { u64::from_str_radix(h, 16).unwrap() } else { s.parse().unwrap() }
}

fn parse_u32(s: &str) -> u32 {
    if let Some(h) = s.strip_prefix("0x") { u32::from_str_radix(h, 16).unwrap() } else { s.parse().unwrap() }
}
