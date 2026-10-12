//! `agx_run`: functional simulation of one compute dispatch over arbitrary memory (e.g. a driver-compiled shader).
//!
//! agx_run MAIN.bin --grid X[,Y,Z] --wg x[,y,z] [--preamble PRE.bin] [--shared BYTES] [--regs N]
//!         [--uniform I=V ...] [--map VA=FILE ...] [--alloc VA=BYTES ...] [--dump VA=BYTES:FILE ...] [--max-steps N]
//!
//! Memory regions are mapped at the given virtual addresses (hex with 0x, or decimal): --map from a file's bytes,
//! --alloc zero-filled. --uniform sets 32-bit uniform registers (u0_u1 = 0,1 for the root-table pointer of a Mesa
//! shader). With --preamble, that program runs first as one lane and the uniforms it stores (uniform_store) are the
//! main shader's uniforms. --dump writes a region back after the run. Prints one JSON line (instructions, groups,
//! uniforms after the preamble).

use agx_sim::func::{self, Dispatch, Memory};

fn num(s: &str) -> u64 {
    let s = s.trim();
    if let Some(h) = s.strip_prefix("0x") { u64::from_str_radix(h, 16) } else { s.parse() }.unwrap_or_else(|_| die(&format!("bad number {:?}", s)))
}
fn dims(s: &str) -> [u32; 3] {
    let v: Vec<u32> = s.split(',').map(|x| num(x) as u32).collect();
    [v[0], *v.get(1).unwrap_or(&1), *v.get(2).unwrap_or(&1)]
}
fn die(m: &str) -> ! {
    println!("{{\"error\":{}}}", serde_json::to_string(m).unwrap());
    std::process::exit(1)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() || a[0].starts_with('-') {
        eprintln!("usage: agx_run MAIN.bin --grid X,Y,Z --wg x,y,z [--preamble PRE.bin] [--uniform I=V] [--map VA=FILE] [--alloc VA=BYTES] [--dump VA=BYTES:FILE]");
        std::process::exit(2);
    }
    let code = std::fs::read(&a[0]).unwrap_or_else(|e| die(&format!("{}: {}", a[0], e)));
    let (mut grid, mut wg, mut shared, mut regs, mut max_steps) = ([1u32, 1, 1], [1u32, 1, 1], 0u32, 128usize, 100_000_000u64);
    let (mut preamble, mut uniforms, mut dumps) = (None, vec![], vec![]);
    let mut mem = Memory::default();
    let mut i = 1;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_else(|| die(&format!("{} needs a value", a[i])));
        let kv = || v.split_once('=').unwrap_or_else(|| die(&format!("{} expects K=V", a[i])));
        match a[i].as_str() {
            "--grid" => grid = dims(&v),
            "--wg" => wg = dims(&v),
            "--shared" => shared = num(&v) as u32,
            "--regs" => regs = num(&v) as usize,
            "--max-steps" => max_steps = num(&v),
            "--preamble" => preamble = Some(std::fs::read(&v).unwrap_or_else(|e| die(&format!("{}: {}", v, e)))),
            "--uniform" => {
                let (k, x) = kv();
                uniforms.push((num(k) as usize, num(x) as u32));
            }
            "--map" => {
                let (va, f) = kv();
                mem.map(num(va), std::fs::read(f).unwrap_or_else(|e| die(&format!("{}: {}", f, e))));
            }
            "--alloc" => {
                let (va, n) = kv();
                mem.map(num(va), vec![0u8; num(n) as usize]);
            }
            "--dump" => {
                let (va, rest) = kv();
                let (n, f) = rest.split_once(':').unwrap_or_else(|| die("--dump expects VA=BYTES:FILE"));
                dumps.push((num(va), num(n) as usize, f.to_string()));
            }
            o => die(&format!("unknown option {}", o)),
        }
        i += 2;
    }
    let mut code = code;
    while code.len() > 2 && code[code.len() - 2..] == [0, 0] {
        code.truncate(code.len() - 2); // trailing padding
    }
    let after = match &preamble {
        Some(p) => Some(func::run_preamble(p, &uniforms, &mut mem, regs).unwrap_or_else(|e| die(&format!("preamble: {}", e)))),
        None => None,
    };
    let d = Dispatch { code, grid, wg, shared_bytes: shared, uniforms: after.clone().unwrap_or(uniforms), num_registers: regs, max_steps, placement: 0 };
    let st = func::run(&d, &mut mem).unwrap_or_else(|e| die(&e));
    for (va, n, f) in &dumps {
        let mut b = Vec::with_capacity(*n);
        for k in 0..*n {
            b.push(mem.read(va + k as u64, 1).unwrap_or_else(|e| die(&e)) as u8);
        }
        std::fs::write(f, b).unwrap_or_else(|e| die(&format!("{}: {}", f, e)));
    }
    let u: Vec<String> = after.unwrap_or_default().iter().filter(|(_, v)| *v != 0).map(|(k, v)| format!("\"u{}\":{}", k, v)).collect();
    println!("{{\"instructions\":{},\"groups\":{},\"preamble_uniforms\":{{{}}}}}", st.instructions, st.groups, u.join(","));
}
