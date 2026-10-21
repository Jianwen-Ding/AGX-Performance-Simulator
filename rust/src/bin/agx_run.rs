//! `agx_run`: functional simulation of one compute dispatch over arbitrary memory (e.g. a driver-compiled shader).
//!
//! agx_run MAIN.bin --grid X[,Y,Z] --wg x[,y,z] [--preamble PRE.bin] [--shared BYTES] [--regs N]
//!         [--uniform I=V ...] [--map VA=FILE ...] [--alloc VA=BYTES ...] [--dump VA=BYTES:FILE ...] [--max-steps N]
//!         [--alloc-regs N] [--vm-slot N] [--pipeline OFFSET]
//!
//! Memory regions are mapped at the given virtual addresses (hex with 0x, or decimal): --map from a file's bytes,
//! --alloc zero-filled. --uniform sets 32-bit uniform registers (u0_u1 = 0,1 for the root-table pointer of a Mesa
//! shader). With --preamble, that program runs first as one lane and the uniforms it stores (uniform_store) are the
//! main shader's uniforms. --dump writes a region back after the run. Prints one JSON line (instructions, groups,
//! uniforms after the preamble). --alloc-regs / --vm-slot / --pipeline give what get_sr SR19 (register allocation, 16-bit
//! registers), SR21 (VM slot) and SR32 (USC offset of the pipeline words) read; without them those reads are errors.
//! --m2-core-map FILE supplies a JSON array of packed M2 Max SR20 IDs, one per linear workgroup.
//! It replays supplied placement, leaving SR17 slots and default placement unchanged when omitted.
//! --lines OBJ.o reads the DWARF line table of the AGX object the code is the .text of (llc output with debug
//! locations); an error at a PC then also prints that PC's source location (`"source":"file:line:column"`).
//! --lines-offset N is where MAIN.bin starts in that object's .text (a kernel cut out of a module).
//! --agx-strict GPU (g13, g13x, g14, g14d; or AGX_STRICT=GPU) fails on the first executed instruction that Mesa's
//! compiler would not emit for that GPU (`agx_sim::isa::mesa`).

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
    let mut core_map: Option<Vec<u32>> = None;
    let mut lines: Option<agx_sim::lines::LineTable> = None;
    let mut lines_offset = 0u64;
    let mut mem = Memory::default();
    let mut launch = agx_sim::cost::SrLaunch::default();
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
            "--alloc-regs" => launch.regs = num(&v) as u32,
            "--vm-slot" => launch.context = Some(num(&v) as u32),
            "--pipeline" => launch.pipeline = Some(num(&v) as u32),
            "--agx-strict" => agx_sim::isa::mesa::set_mode(Some(agx_sim::isa::mesa::MesaGen::parse(&v).unwrap_or_else(|e| die(&e)))),
            "--m2-core-map" => {
                let bytes = std::fs::read(&v).unwrap_or_else(|e| die(&format!("{}: {}", v, e)));
                core_map = Some(serde_json::from_slice(&bytes).unwrap_or_else(|e| die(&format!("M2 core map: {}", e))));
            }
            "--lines" => {
                let elf = std::fs::read(&v).unwrap_or_else(|e| die(&format!("{}: {}", v, e)));
                lines = Some(agx_sim::lines::LineTable::from_elf(&elf).unwrap_or_else(|e| die(&format!("{}: {}", v, e))));
            }
            "--lines-offset" => lines_offset = num(&v),
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
    let mut d = Dispatch { code, grid, wg, shared_bytes: shared, uniforms: uniforms.clone(), num_registers: regs, max_steps, placement: 0,
                  launch: agx_sim::cost::SrLaunch { tg_bytes: shared, ..launch } };
    if let Some(cores) = &core_map { func::validate_m2_core_map(&d, cores).unwrap_or_else(|e| die(&e)); }
    let after = match &preamble {
        Some(p) => Some(func::run_preamble(p, &uniforms, &mut mem, regs).unwrap_or_else(|e| die(&format!("preamble: {}", e)))),
        None => None,
    };
    d.uniforms = after.clone().unwrap_or(uniforms);
    let st = match &core_map {
        Some(cores) => func::run_with_m2_core_map(&d, &mut mem, cores),
        None => func::run(&d, &mut mem),
    }.unwrap_or_else(|e| {
        let at = lines.as_ref().zip(agx_sim::lines::error_pc(&e)).and_then(|(t, pc)| t.describe(pc + lines_offset));
        match at {
            Some(src) => {
                println!("{{\"error\":{},\"source\":{}}}", serde_json::to_string(&e).unwrap(), serde_json::to_string(&src).unwrap());
                std::process::exit(1)
            }
            None => die(&e),
        }
    });
    for (va, n, f) in &dumps {
        let b = mem.read_bytes(*va, *n).unwrap_or_else(|e| die(&e));
        std::fs::write(f, b).unwrap_or_else(|e| die(&format!("{}: {}", f, e)));
    }
    let u: Vec<String> = after.unwrap_or_default().iter().filter(|(_, v)| *v != 0).map(|(k, v)| format!("\"u{}\":{}", k, v)).collect();
    let provenance = core_map.as_ref().map(|c| format!(",\"core_map\":\"m2\",\"mapped_workgroups\":{}", c.len())).unwrap_or_default();
    println!("{{\"instructions\":{},\"groups\":{},\"preamble_uniforms\":{{{}}}{}}}", st.instructions, st.groups, u.join(","), provenance);
}

