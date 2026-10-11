//! `agx-cost code.bin [--groups G] [--annotate] [--units real|ref] [--allow-unknown] [--load-latency N]`
//!
//! Drop-in replacement for `python3 scripts/agx_cost.py`: same arguments, same JSON line and the
//! same `--annotate` table.

use agx_sim::cost::{self, Params, REAL_CYCLES_PER_UNIT};
use agx_sim::py_round_repr;
use std::process::exit;

fn usage() -> ! {
    eprintln!("usage: agx-cost code.bin [--groups G] [--load-latency CYCLES] [--units real|ref] [--allow-unknown] [--annotate]");
    exit(2)
}

fn json_list(items: impl IntoIterator<Item = String>) -> String {
    let v: Vec<String> = items.into_iter().map(|s| serde_json::to_string(&s).unwrap()).collect();
    format!("[{}]", v.join(", "))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut code_path = None;
    let (mut groups, mut units, mut allow_unknown, mut annotate, mut load_latency) = (1usize, "real".to_string(), false, false, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--groups" => groups = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--load-latency" => load_latency = Some(args.next().and_then(|v| v.parse::<i64>().ok()).unwrap_or_else(|| usage()) as f64),
            "--units" => {
                units = args.next().unwrap_or_else(|| usage());
                if units != "real" && units != "ref" {
                    usage()
                }
            }
            "--allow-unknown" => allow_unknown = true,
            "--annotate" => annotate = true,
            "-h" | "--help" => usage(),
            _ if code_path.is_none() && !a.starts_with("--") => code_path = Some(a),
            _ => usage(),
        }
    }
    let code_path = code_path.unwrap_or_else(|| usage());
    let run = || -> Result<(), String> {
        let mut p = Params::load(&cost::default_bench(), &cost::default_calib(), load_latency)?;
        p.allow_unknown = allow_unknown;
        let code = std::fs::read(&code_path).map_err(|e| format!("{}: {}", code_path, e))?;
        let ins = cost::decode(&code, &p)?;
        let tr = cost::trace(&code, 32, 500_000)?;
        if p.branch.is_none() {
            cost::fit_branch(&mut p)?;
        }
        let (cyc, stats) = cost::simulate(&ins, &tr.pcs, &p, groups, true)?;
        let stats = stats.unwrap();
        let scale = if units == "real" { 1.0 } else { 1.0 / REAL_CYCLES_PER_UNIT };
        let mut unmeasured: Vec<String> = ins.iter().filter(|r| r.unmeasured).map(|r| r.name.clone()).collect();
        unmeasured.sort();
        unmeasured.dedup();
        println!(
            "{{\"instructions\": {}, \"dynamic\": {}, \"groups\": {}, \"units\": {}, \"cycles\": {}, \"emulator_skipped\": {}, \"warnings\": {}, \"unmeasured_classes\": {}}}",
            ins.len(),
            tr.pcs.len(),
            groups,
            serde_json::to_string(&units).unwrap(),
            py_round_repr(cyc * scale, 1),
            json_list(tr.skipped.iter().cloned()),
            json_list(tr.warnings.iter().cloned()),
            json_list(unmeasured)
        );
        if annotate {
            println!("{:>6}  {:>7}  {:>8}  {:>8}  {:<16} instruction", "pc", "n", "issue", "stall", "class");
            for r in &ins {
                let s = stats.get(&r.pc).cloned().unwrap_or_default();
                let flag = if r.unmeasured { " *unmeasured" } else { "" };
                let class = r.unit.clone().unwrap_or_else(|| r.kind.as_str().to_string());
                println!("{:6}  {:7}  {:8.0}  {:8.0}  {:<16} {}{}", r.pc, s.n, s.issue * scale, s.stall * scale, class, r.text, flag);
            }
        }
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("agx-cost: {}", e);
        exit(1);
    }
}
