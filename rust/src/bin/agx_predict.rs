//! `agx-predict PHASE_A.bin PHASE_B.bin UNITS GROUPS`
//!
//! The model side of `scripts/cost_hw.py`'s `predict`: reference units per unit of
//! (PHASE_B - PHASE_A), i.e. `to_ref(simulate(B) - simulate(A)) / UNITS`. Prints the value as a
//! shortest round-trip float, so `float()` in Python recovers it exactly.

use agx_sim::cost::{self, to_ref, Params};
use agx_sim::isa::operand::py_float;

fn sim(path: &str, p: &Params, groups: usize) -> Result<f64, String> {
    let code = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let ins = cost::decode(&code, p)?;
    let tr = cost::trace(&code, 32, 500_000)?;
    Ok(cost::simulate(&ins, &tr.pcs, p, groups, false)?.0)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 4 {
        eprintln!("usage: agx-predict PHASE_A.bin PHASE_B.bin UNITS GROUPS");
        std::process::exit(2);
    }
    let run = || -> Result<f64, String> {
        let units: f64 = a[2].parse().map_err(|_| "UNITS must be a number")?;
        let groups: usize = a[3].parse().map_err(|_| "GROUPS must be an integer")?;
        let mut p = Params::new()?;
        if p.branch.is_none() {
            cost::fit_branch(&mut p)?;
        }
        Ok(to_ref(sim(&a[1], &p, groups)? - sim(&a[0], &p, groups)?) / units)
    };
    match run() {
        Ok(v) => println!("{}", py_float(v)),
        Err(e) => {
            eprintln!("agx-predict: {}", e);
            std::process::exit(1);
        }
    }
}
