//! Predict the time of one compute dispatch on the whole GPU (see `agx_sim::dispatch`).
//!
//! agx_dispatch code.bin --grid X[,Y,Z] [--wg x[,y,z]] [--regs HALVES] [--tg-bytes N] [--clock MHZ]
//!              [--cores N] [--launch-us US] [--uniform IDX=VALUE ...] [--no-mem] [--max-waves N]
//!              [--m2-core-map FILE] (supplied M2 core IDs; exhaustive isolated waves)
//!              [--busiest-core-wgs N] (measured peak count; representative-core sensitivity only)
//!              [--all-workgroups] (with --map: trace every workgroup, on the cores in turn; slower, and needed for
//!              data-dependent kernels, whose time is their slowest workgroup's)
//!              [--preamble PRE.bin] [--map VA=FILE ...] [--alloc VA=BYTES ...] [--sampler-heap VA=COUNT] [--agx-strict GPU]
//!              [--launch-profile drm|runtime] [--group-launch-ns NS] [--stack BYTES] [--store-gbps GBPS]
//!
//! --launch-profile: the launch costs of a command as the direct-DRM harness submits it (`drm`, the default: 5.6 us
//! fixed, max(100 ns per workgroup, 13.6 ns per SIMD group) per core) or as the CUDA driver / agx_runtime submit it back to
//! back, as an application keeps the GPU busy (`runtime`: 2.3 us fixed, max(100 ns per workgroup, 3.2 ns per SIMD
//! group; empty kernels over 1 to 15,200 workgroups of 32 and 1024 threads, playground
//! novel-results/2026-11-01-runtime-launch-cost; and 104 ns per 64-byte line of code executed, the cold instruction
//! fetch, 2026-11-01-code-fetch). Options after it override its values.
//!
//! code.bin is the main shader as submitted (e.g. the direct-DRM harness's control main with a body).
//! Uniforms are 32-bit uniform registers (u4 = 4); the harness uses u4 = 0, u5 = count, u6_u7 = output,
//! u8_u9 = table. Prints one JSON object.
//!
//! With --preamble, --map or --alloc (as for agx_run) the SIMD groups' paths come from the functional simulator
//! over that memory (`func::trace_workgroups`) instead of the cost model's emulation, which reads memory as zero:
//! needed for driver-compiled code (the preamble fills the uniforms from the root table) and data-dependent loops.
//!
//! --agx-strict GPU (g13, g13x, g14, g14d; or AGX_STRICT=GPU) fails on instructions Mesa's compiler would not
//! emit for that GPU (`agx_sim::isa::mesa`).
//!
//! agx_dispatch --batch JOBS: JOBS has one dispatch per line, as a JSON array of the arguments above. The model's
//! parameters are loaded once, the dispatches run in parallel, and one JSON object per line is printed in order
//! (`{"error": ...}` for a dispatch that failed; the exit status is then 1). Each result is the one the single form
//! prints.

use agx_sim::cost::{self, Params};
use agx_sim::dispatch::{self, Launch};
use agx_sim::{func, mapped_dispatch};

fn number<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("{}: invalid value {:?}", what, s))
}

fn dims(s: &str) -> Result<[u32; 3], String> {
    let v: Vec<_> = s.split(',').collect();
    if !(1..=3).contains(&v.len()) { return Err("expected one to three dimensions".into()); }
    let mut out = [1; 3];
    for (i, x) in v.into_iter().enumerate() { out[i] = number(x, "dimension")?; }
    Ok(out)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() {
        eprintln!("usage: agx_dispatch code.bin --grid X[,Y,Z] [--wg x,y,z] [--regs H] [--tg-bytes N] [--clock MHZ] [--uniform I=V] [--no-mem] [--m2-core-map FILE]");
        eprintln!("       agx_dispatch --batch JOBS   (one JSON array of those arguments per line)");
        std::process::exit(2);
    }
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    if a[0] == "--batch" {
        if a.len() != 2 { eprintln!("agx_dispatch: expected --batch JOBS"); std::process::exit(2); }
        let text = match std::fs::read_to_string(&a[1]) {
            Ok(s) => s,
            Err(e) => { eprintln!("agx_dispatch: {}: {}", a[1], e); std::process::exit(1); }
        };
        let jobs: Vec<Result<Vec<String>, String>> = text.lines().enumerate().filter(|(_, x)| !x.trim().is_empty())
            .map(|(i, x)| serde_json::from_str(x).map_err(|e| format!("batch line {}: expected an argument array: {}", i + 1, e))).collect();
        if jobs.is_empty() { eprintln!("agx_dispatch: batch contains no jobs"); std::process::exit(1); }
        // one dispatch per core when there are enough of them, each simulating on its own thread; else one at a time,
        // each with the cores to itself
        let outer = jobs.len() >= agx_sim::par::threads();
        let run = |job: &Result<Vec<String>, String>| {
            if outer {
                agx_sim::par::set_local_threads(1);
            }
            match job { Ok(args) => run_job(args, &p), Err(e) => Err(e.clone()) }
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
        Ok(v) => {
            println!("{}", v);
            if let Some(r) = agx_sim::cost::mem_levels_report() { eprintln!("{}", r); }
        }
        Err(e) => {
            eprintln!("agx_dispatch: {}", e);
            std::process::exit(1);
        }
    }
}

/// One dispatch: its arguments as on the command line, the JSON result.
fn run_job(a: &[String], p: &Params) -> Result<serde_json::Value, String> {
    let path = a.first().ok_or("empty job: expected a shader path")?;
    let mut l = Launch::default();
    let mut preamble = None;
    let mut core_map_path = None;
    let mut max_waves_explicit = false;
    let mut buffers = vec![];
    let mut sampler_heap = None;
    let mut all_workgroups = false;
    enum Buffer { File(String), Allocate(usize) }
    let mut i = 1;
    while i < a.len() {
        let option = a[i].as_str();
        if option == "--no-mem" { l.mem = false; i += 1; continue; }
        if option == "--all-workgroups" { all_workgroups = true; i += 1; continue; }
        if !["--grid", "--wg", "--regs", "--tg-bytes", "--clock", "--cores", "--busiest-core-wgs", "--m2-core-map", "--sampler-heap", "--launch-us", "--max-waves", "--uniform", "--preamble", "--map", "--alloc", "--agx-strict", "--launch-profile", "--group-launch-ns", "--stack", "--store-gbps"].contains(&option) {
            return Err(format!("unknown argument {}", option));
        }
        let v = a.get(i + 1).ok_or_else(|| format!("{} requires a value", option))?;
        match option {
            "--grid" => l.grid = dims(v).map_err(|e| format!("--grid: {}", e))?,
            "--wg" => l.wg = dims(v).map_err(|e| format!("--wg: {}", e))?,
            "--regs" => l.regs = number(v, option)?,
            "--tg-bytes" => l.tg_bytes = number(v, option)?,
            "--clock" => l.clock_mhz = number(v, option)?,
            "--cores" => l.cores = number(v, option)?,
            "--busiest-core-wgs" => l.busiest_core_wgs_override = Some(number(v, option)?),
            "--launch-us" => l.launch_ns = number::<f64>(v, option)? * 1000.0,
            "--group-launch-ns" => l.group_launch_ns = number(v, option)?,
            "--stack" => l.stack_bytes = number(v, option)?,
            "--store-gbps" => l.store_gbps = number(v, option)?,
            "--launch-profile" => match v.as_str() {
                "drm" => {
                    let d = Launch::default();
                    (l.launch_ns, l.wg_launch_ns, l.group_launch_ns, l.fetch_line_ns, l.stack_launch_ns, l.store_gbps, l.store_launch_ns) = (d.launch_ns, d.wg_launch_ns, d.group_launch_ns, d.fetch_line_ns, d.stack_launch_ns, d.store_gbps, d.store_launch_ns);
                    l.later_waves_warm = d.later_waves_warm;
                    l.partial_warm = d.partial_warm;
                    l.chain_waves = d.chain_waves;
                    l.icache_lines = d.icache_lines;
                    l.icache_victims = d.icache_victims;
                }
                "runtime" => {
                    (l.launch_ns, l.wg_launch_ns, l.group_launch_ns, l.fetch_line_ns, l.stack_launch_ns, l.store_gbps, l.store_launch_ns, l.later_waves_warm, l.partial_warm, l.chain_waves) = (2300.0, 100.0, 3.2, 104.0, 5000.0, 365.0, 700.0, true, true, true);
                    l.stack_slot_ns = 1150.0;
                    // code beyond the instruction cache is fetched through it in the simulation (Launch::icache_lines)
                    // 12 KiB, 192 lines (playground novel-results/2026-10-20-icache), replacement not strict LRU: past
                    // the capacity the time rises gradually to streaming by ~20 KB (1.33x the lines: 48% of lines miss,
                    // random replacement 46%, LRU 100%). Random over the whole cache, a later wave starting with the
                    // lines the earlier one left: Rodinia hotspot +66% -> +1%, srad_cuda_1 +43 -> +16%, particlefilter
                    // likelihood +31 -> +12% (results/2026-11-07-held-out)
                    l.icache_lines = 192;
                    l.icache_victims = 192;
                }
                other => return Err(format!("--launch-profile: {other:?} is not drm or runtime")),
            },
            "--max-waves" => { l.max_waves = number(v, option)?; max_waves_explicit = true; },
            "--m2-core-map" => core_map_path = Some(v.clone()),
            "--agx-strict" => agx_sim::isa::mesa::set_mode(Some(agx_sim::isa::mesa::MesaGen::parse(v)?)),
            "--uniform" => {
                let (k, x) = v.split_once('=').ok_or("--uniform expects IDX=VALUE")?;
                l.uniforms.push((number(k.trim_start_matches('u'), "uniform index")?, parse_u32(x)?));
            }
            "--preamble" => {
                preamble = Some(v.clone());
            }
            "--sampler-heap" => {
                let (va, count) = v.split_once('=').ok_or("--sampler-heap expects VA=COUNT")?;
                sampler_heap = Some((parse_u64(va)?, parse_u32(count)?));
            }
            "--map" | "--alloc" => {
                let (va, x) = v.split_once('=').ok_or("--map/--alloc expects VA=FILE or VA=BYTES")?;
                let b = if option == "--map" { Buffer::File(x.to_string()) } else { Buffer::Allocate(parse_u32(x)? as usize) };
                buffers.push((parse_u64(va)?, b));
            }
            _ => unreachable!(),
        }
        i += 2;
    }
    // Reject invalid geometry/resources before reading shaders/preambles or
    // materializing mapped buffers. simulate_with repeats this check for API callers.
    l.validate()?;
    let core_map: Option<Vec<u32>> = core_map_path.map(|path| {
        if max_waves_explicit { return Err("M2 core map traces every wave; --max-waves is incompatible".into()); }
        let bytes = std::fs::read(&path).map_err(|e| format!("M2 core map {}: {}", path, e))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("M2 core map: {}", e))
    }).transpose()?;
    // Map validation precedes shader/preamble reads and mapped allocations.
    if let Some(cores) = &core_map {
        let preflight = func::Dispatch { code: vec![], grid: l.grid, wg: l.wg, shared_bytes: l.tg_bytes,
            uniforms: l.uniforms.clone(), num_registers: 128, max_steps: l.max_steps as u64, placement: 0,
            launch: cost::SrLaunch { tg_bytes: l.tg_bytes, regs: l.regs, ..Default::default() } };
        mapped_dispatch::validate(&preflight, &l, cores)?;
    }
    let mut code = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    trim_padding(&mut code);
    // Every workgroup on the cores in turn, in the core IDs of the M2 Max: a launch ends with its slowest workgroup, and
    // data-dependent ones differ (Blender Cycles' intersect_closest: 3x within a launch), which a sample of them misses.
    let core_map = if all_workgroups && core_map.is_none() {
        const MASKS: [u32; 4] = [0x3ff, 0x1ff, 0x1ff, 0x3ff];
        let ids: Vec<u32> = (0..4u32).flat_map(|c| (0..16u32).filter(move |i| MASKS[c as usize] & (1 << i) != 0).map(move |i| c * 16 + i)).collect();
        let n = (0..3).map(|i| l.grid[i].div_ceil(l.wg[i]) as usize).product::<usize>();
        Some((0..n).map(|k| ids[k % ids.len()]).collect::<Vec<u32>>())
    } else {
        core_map
    };
    let functional = core_map.is_some() || preamble.is_some() || !buffers.is_empty();
    let mut mem = func::Memory::default();
    mem.sampler_heap = sampler_heap;
    for (va, b) in buffers {
        let bytes = match b {
            Buffer::File(x) => std::fs::read(&x).map_err(|e| format!("{}: {}", x, e))?,
            Buffer::Allocate(n) => vec![0; n],
        };
        l.regions.push((va, bytes.len() as u64));
        mem.map(va, bytes);
    }
    let d = if functional {
        if let Some(pre) = &preamble {
            let code = std::fs::read(pre).map_err(|e| format!("{}: {}", pre, e))?;
            l.uniforms = func::run_preamble(&code, &l.uniforms, &mut mem, 128).map_err(|e| format!("preamble: {}", e))?;
        }
        Some(func::Dispatch { code: code.clone(), grid: l.grid, wg: l.wg, shared_bytes: l.tg_bytes, uniforms: l.uniforms.clone(),
                              num_registers: 128, max_steps: l.max_steps as u64, placement: 0,
                              launch: agx_sim::cost::SrLaunch { tg_bytes: l.tg_bytes, regs: l.regs, ..Default::default() } })
    } else {
        None
    };
    if let Some(cores) = &core_map {
        let r = mapped_dispatch::simulate(d.as_ref().unwrap(), &mem, &l, p, cores)?;
        let s = &r.summary;
        return Ok(serde_json::json!({
            "time_us":s.time_ns/1000.0,"core_cycles":s.core_cycles,"clock_mhz":l.clock_mhz,
            "workgroups":s.workgroups,"groups_per_wg":s.groups_per_wg,"group_cap_per_core":s.group_cap_per_core,
            "wgs_per_core":s.wgs_per_core,"busiest_core_wgs":s.busiest_core_wgs,"waves":s.waves,
            "launch_core_us":s.launch_core_ns/1000.0,"simulated_waves":s.simulated_waves,"launch_us":l.launch_ns/1000.0,
            "warnings":s.warnings,"paths":"functional","core_map":"m2","mapped_workgroups":cores.len(),
            "active_cores":r.cores.len(),"critical_core":r.critical_core,"global_device_segments":r.global_device_segments,
            "device_footprint_complete":r.device_footprint_complete,"footprint_scope":"device-load-store-atomic",
            "resident_slc_assumed":r.resident_slc_assumed,"waves_exhaustive":true,
            "wave_order_assumption":"linear-workgroup-id","slot_assumption":"group-index-within-workgroup",
            "memory_assumption":"isolated-workgroup-snapshots","cache_assumption":"cold-per-wave-with-footprint-based-SLC-residency",
            "cores":r.cores.iter().map(|c| serde_json::json!({"id":c.id,"workgroups":c.workgroups,
                "core_cycles":c.cycles,"launch_core_us":c.launch_ns/1000.0,"completion_us":c.time_ns/1000.0,
                "waves":c.waves.iter().map(|w| serde_json::json!({"index":w.index,"workgroups":w.workgroups,
                    "cycles":w.cycles,"sharing":w.sharing})).collect::<Vec<_>>() })).collect::<Vec<_>>()
        }));
    }
    let prepared = d.as_ref().map(func::PreparedDispatch::new).transpose()?;
    let tracer = |wg: [u32; 3]| -> Result<Vec<cost::Trace>, String> {
        prepared.as_ref().unwrap().trace_workgroup(&mem, wg)
    };
    let r = dispatch::simulate_with(&code, &l, p, if functional { Some(&tracer) } else { None })?;
    let mut output = serde_json::json!({
        "time_us": r.time_ns / 1000.0, "core_cycles": r.core_cycles, "clock_mhz": l.clock_mhz,
        "workgroups": r.workgroups, "groups_per_wg": r.groups_per_wg, "group_cap_per_core": r.group_cap_per_core,
        "wgs_per_core": r.wgs_per_core, "launch_core_us": r.launch_core_ns / 1000.0, "busiest_core_wgs": r.busiest_core_wgs, "waves": r.waves,
        "simulated_waves": r.simulated_waves, "launch_us": l.launch_ns / 1000.0, "code_lines": r.code_lines, "warnings": r.warnings,
        "paths": if functional { "functional" } else { "cost-emulation" },
    });
    if let Some(peak) = l.busiest_core_wgs_override {
        output["busiest_core_wgs_input"] = serde_json::json!(peak);
    }
    Ok(output)
}

fn trim_padding(code: &mut Vec<u8>) {
    // AGX instructions are halfword aligned. Byte-wise stripping also removed
    // the high byte of the final 0x0088 stop and left a truncated instruction.
    while code.len() > 2 && code[code.len() - 2..] == [0, 0] {
        code.truncate(code.len() - 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_preserves_complete_stop() {
        for pad in [0, 2, 16, 128] {
            let mut code = vec![0x88, 0];
            code.extend(vec![0; pad]);
            trim_padding(&mut code);
            assert_eq!(code, vec![0x88, 0]);
            let d = func::Dispatch { code, grid: [1; 3], wg: [1; 3], shared_bytes: 0,
                uniforms: vec![], num_registers: 8, max_steps: 1, placement: 0, launch: Default::default() };
            assert_eq!(func::run(&d, &mut func::Memory::default()).unwrap().instructions, 1);
        }
    }
}

fn parse_u64(s: &str) -> Result<u64, String> {
    if let Some(h) = s.strip_prefix("0x") { u64::from_str_radix(h, 16).map_err(|_| format!("invalid u64 {:?}", s)) } else { number(s, "u64") }
}

fn parse_u32(s: &str) -> Result<u32, String> {
    if let Some(h) = s.strip_prefix("0x") { u32::from_str_radix(h, 16).map_err(|_| format!("invalid u32 {:?}", s)) } else { number(s, "u32") }
}

