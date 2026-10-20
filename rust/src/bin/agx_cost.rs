//! `agx-cost code.bin [--groups G] [--annotate] [--units real|ref] [--allow-unknown] [--load-latency N]`
//!         `[--uniform IDX=VALUE ...] [--wg x[,y,z]] [--wg-id x[,y,z]] [--simd-group G] [--grid X[,Y,Z]]`
//!         `[--regs HALVES] [--tg-bytes N] [--map VA=FILE ...] [--alloc VA=BYTES ...] [--preamble PRE.bin]`
//!         `[--mesa-isa GPU]`
//!
//! Drop-in replacement for `python3 scripts/agx_cost.py`: same arguments, same JSON line and the
//! same `--annotate` table.
//!
//! The costed path is one SIMD group's. By default it comes from the cost model's emulation, which reads
//! uniforms and memory as zero, so a loop or branch on a kernel argument (`tid < n`, `k < K`) takes the
//! zero path and a loop on loaded data follows zeros (`warnings` names such conditions). Launch inputs
//! make those paths real:
//! - `--uniform` (32-bit uniform registers, as `agx_dispatch`: u4 = 4), `--wg`, `--wg-id`, `--simd-group`,
//!   `--regs` and `--tg-bytes` trace that SIMD group of that workgroup, with its special registers
//!   (`cost::trace_group`); memory still reads as zero.
//! - `--map`, `--alloc` (and `--preamble`, as for `agx_run`) take the path from the functional simulator over
//!   that memory instead (`func::PreparedDispatch::trace_workgroup`): exact control flow for data-dependent
//!   loops. `--grid` (default: just large enough for `--wg-id`) sets the dispatch the workgroup belongs to.
//!
//! With launch inputs the JSON line also has `"paths"`: `"geometry"` or `"functional"`.
//!
//! `--mesa-isa GPU` (g13, g13x, g14, g14d; or `AGX_MESA_ISA=GPU`) fails on any instruction Mesa's compiler would
//! not emit for that GPU (`agx_sim::isa::mesa`).

use agx_sim::cost::{self, GroupCtx, Params, SrLaunch, REAL_CYCLES_PER_UNIT};
use agx_sim::{func, py_round_repr};
use std::process::exit;

fn usage() -> ! {
    eprintln!("usage: agx-cost code.bin [--groups G] [--load-latency CYCLES] [--units real|ref] [--allow-unknown] [--annotate]");
    eprintln!("                [--uniform IDX=VALUE] [--wg x,y,z] [--wg-id x,y,z] [--simd-group G] [--grid X,Y,Z]");
    eprintln!("                [--regs HALVES] [--tg-bytes N] [--map VA=FILE] [--alloc VA=BYTES] [--preamble PRE.bin]");
    eprintln!("                [--mesa-isa g13|g13x|g14|g14d]");
    exit(2)
}

fn json_list(items: impl IntoIterator<Item = String>) -> String {
    let v: Vec<String> = items.into_iter().map(|s| serde_json::to_string(&s).unwrap()).collect();
    format!("[{}]", v.join(", "))
}

fn parse_u64(s: &str) -> Option<u64> {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

fn dims(s: &str) -> Option<[u32; 3]> {
    let v: Vec<&str> = s.split(',').collect();
    if !(1..=3).contains(&v.len()) {
        return None;
    }
    let mut out = [1; 3];
    for (i, x) in v.into_iter().enumerate() {
        out[i] = x.parse().ok()?;
    }
    Some(out)
}

/// Inputs that select which path of which SIMD group is costed.
#[derive(Default)]
struct Launch {
    uniforms: Vec<(usize, u32)>,
    wg: Option<[u32; 3]>,
    wg_id: Option<[u32; 3]>,
    grid: Option<[u32; 3]>,
    group: Option<u32>,
    regs: u32,
    tg_bytes: u32,
    buffers: Vec<(u64, Result<String, usize>)>,
    preamble: Option<String>,
}

impl Launch {
    fn functional(&self) -> bool {
        !self.buffers.is_empty() || self.preamble.is_some()
    }
    fn any(&self) -> bool {
        self.functional() || !self.uniforms.is_empty() || self.wg.is_some() || self.wg_id.is_some() || self.group.is_some()
            || self.grid.is_some() || self.regs != 0 || self.tg_bytes != 0
    }
    fn sr_launch(&self) -> SrLaunch {
        SrLaunch { tg_bytes: self.tg_bytes, regs: self.regs, ..Default::default() }
    }
}

/// The SIMD group's path under the launch inputs, and which kind of path it is.
fn launch_trace(code: &[u8], l: &Launch, max_steps: usize) -> Result<(cost::Trace, &'static str), String> {
    let wg = l.wg.unwrap_or([32, 1, 1]);
    let wg_id = l.wg_id.unwrap_or([0; 3]);
    let group = l.group.unwrap_or(0);
    let threads = wg[0] * wg[1] * wg[2];
    if threads == 0 || group >= threads.div_ceil(32) {
        return Err(format!("--simd-group {} is outside a {}-thread workgroup", group, threads));
    }
    if !l.functional() {
        let ctx = GroupCtx { wg_size: wg, wg_id, group, core: 0, slot: group, uniforms: l.uniforms.clone(), launch: l.sr_launch() };
        return Ok((cost::trace_group(code, &ctx, max_steps)?, "geometry"));
    }
    let grid = l.grid.unwrap_or([(wg_id[0] + 1) * wg[0], (wg_id[1] + 1) * wg[1], (wg_id[2] + 1) * wg[2]]);
    let mut mem = func::Memory::default();
    for (va, b) in &l.buffers {
        let bytes = match b {
            Ok(path) => std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?,
            Err(n) => vec![0; *n],
        };
        mem.map(*va, bytes);
    }
    let mut uniforms = l.uniforms.clone();
    if let Some(pre) = &l.preamble {
        let pcode = std::fs::read(pre).map_err(|e| format!("{}: {}", pre, e))?;
        uniforms = func::run_preamble(&pcode, &uniforms, &mut mem, 128).map_err(|e| format!("preamble: {}", e))?;
    }
    let d = func::Dispatch { code: code.to_vec(), grid, wg, shared_bytes: l.tg_bytes, uniforms, num_registers: 128,
                             max_steps: max_steps as u64, placement: 0, launch: l.sr_launch() };
    let mut traces = func::PreparedDispatch::new(&d)?.trace_workgroup(&mem, wg_id)?;
    if group as usize >= traces.len() {
        return Err(format!("workgroup {:?} has {} SIMD groups", wg_id, traces.len()));
    }
    Ok((traces.swap_remove(group as usize), "functional"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut code_path = None;
    let (mut groups, mut units, mut allow_unknown, mut annotate, mut load_latency) = (1usize, "real".to_string(), false, false, None);
    let mut l = Launch::default();
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--groups" => groups = value().parse().unwrap_or_else(|_| usage()),
            "--load-latency" => load_latency = Some(value().parse::<i64>().unwrap_or_else(|_| usage()) as f64),
            "--units" => {
                units = value();
                if units != "real" && units != "ref" {
                    usage()
                }
            }
            "--allow-unknown" => allow_unknown = true,
            "--annotate" => annotate = true,
            "--uniform" => {
                let v = value();
                let (k, x) = v.split_once('=').unwrap_or_else(|| usage());
                let k = k.trim_start_matches('u').parse().unwrap_or_else(|_| usage());
                let x = parse_u64(x).filter(|&x| x <= u32::MAX as u64).unwrap_or_else(|| usage()) as u32;
                l.uniforms.push((k, x));
            }
            "--wg" => l.wg = Some(dims(&value()).unwrap_or_else(|| usage())),
            "--wg-id" => l.wg_id = Some(dims(&value()).unwrap_or_else(|| usage())),
            "--grid" => l.grid = Some(dims(&value()).unwrap_or_else(|| usage())),
            "--simd-group" => l.group = Some(value().parse().unwrap_or_else(|_| usage())),
            "--regs" => l.regs = value().parse().unwrap_or_else(|_| usage()),
            "--tg-bytes" => l.tg_bytes = value().parse().unwrap_or_else(|_| usage()),
            "--map" | "--alloc" => {
                let v = value();
                let (va, x) = v.split_once('=').unwrap_or_else(|| usage());
                let va = parse_u64(va).unwrap_or_else(|| usage());
                let b = if a == "--map" { Ok(x.to_string()) } else { Err(parse_u64(x).unwrap_or_else(|| usage()) as usize) };
                l.buffers.push((va, b));
            }
            "--preamble" => l.preamble = Some(value()),
            "--mesa-isa" => {
                let g = agx_sim::isa::mesa::MesaGen::parse(&value()).unwrap_or_else(|e| { eprintln!("agx-cost: {}", e); exit(2) });
                agx_sim::isa::mesa::set_mode(Some(g));
            }
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
        let (tr, paths) = if l.any() { let (t, k) = launch_trace(&code, &l, 500_000)?; (t, Some(k)) } else { (cost::trace(&code, 32, 500_000)?, None) };
        if p.branch.is_none() {
            cost::fit_branch(&mut p)?;
        }
        let (cyc, stats) = cost::simulate(&ins, &tr.pcs, &p, groups, true)?;
        let stats = stats.unwrap();
        let scale = if units == "real" { 1.0 } else { 1.0 / REAL_CYCLES_PER_UNIT };
        let mut unmeasured: Vec<String> = ins.iter().filter(|r| r.unmeasured).map(|r| r.name.clone()).collect();
        unmeasured.sort();
        unmeasured.dedup();
        let paths = paths.map(|k| format!(", \"paths\": \"{}\"", k)).unwrap_or_default();
        println!(
            "{{\"instructions\": {}, \"dynamic\": {}, \"groups\": {}, \"units\": {}, \"cycles\": {}, \"emulator_skipped\": {}, \"warnings\": {}, \"unmeasured_classes\": {}{}}}",
            ins.len(),
            tr.pcs.len(),
            groups,
            serde_json::to_string(&units).unwrap(),
            py_round_repr(cyc * scale, 1),
            json_list(tr.skipped.iter().cloned()),
            json_list(tr.warnings.iter().cloned()),
            json_list(unmeasured),
            paths
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
