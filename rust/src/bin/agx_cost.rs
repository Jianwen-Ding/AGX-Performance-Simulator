//! `agx-cost code.bin [--groups G] [--annotate] [--units real|ref] [--allow-unknown] [--load-latency N]`
//!         `[--uniform IDX=VALUE ...] [--wg x[,y,z]] [--wg-id x[,y,z]] [--simd-group G] [--grid X[,Y,Z]]`
//!         `[--regs HALVES] [--tg-bytes N] [--map VA=FILE ...] [--alloc VA=BYTES ...] [--preamble PRE.bin]`
//!         `[--agx-strict GPU] [--core-throughput]`
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
//! `--agx-strict GPU` (g13, g13x, g14, g14d; or `AGX_STRICT=GPU`) fails on any instruction Mesa's compiler would
//! not emit for that GPU (`agx_sim::isa::mesa`).
//!
//! The JSON line always has `"occupancy"`: resident SIMD groups, workgroups and threads per core
//! (`dispatch::occupancy`), from `--regs` or else from the highest register the code names, for the `--wg` and
//! `--tg-bytes` of the launch (default: 32 threads, no threadgroup memory). `cycles` stays one path of `--groups` groups.
//!
//! `--core-throughput` adds `"core_throughput"`: threads per microsecond of one core running as many workgroups as fit,
//! to compare compilations at different register counts (fewer spills against more resident threads). A model
//! estimate, not validated against hardware: see `core_throughput`.

use agx_sim::cost::{self, GroupCtx, MemModel, Params, SrLaunch, Trace, REAL_CYCLES_PER_UNIT};
use agx_sim::{dispatch, func, isa, py_round_repr};
use std::collections::HashSet;
use std::process::exit;

fn usage() -> ! {
    eprintln!("usage: agx-cost code.bin [--groups G] [--load-latency CYCLES] [--units real|ref] [--allow-unknown] [--annotate]");
    eprintln!("                [--uniform IDX=VALUE] [--wg x,y,z] [--wg-id x,y,z] [--simd-group G] [--grid X,Y,Z]");
    eprintln!("                [--regs HALVES] [--tg-bytes N] [--map VA=FILE] [--alloc VA=BYTES] [--preamble PRE.bin]");
    eprintln!("                [--agx-strict g13|g13x|g14|g14d] [--core-throughput]");
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
    let mut traces = functional_traces(code, l, max_steps)?;
    if group as usize >= traces.len() {
        return Err(format!("workgroup {:?} has {} SIMD groups", wg_id, traces.len()));
    }
    Ok((traces.swap_remove(group as usize), "functional"))
}

/// Every SIMD group of the launch's workgroup, from the functional simulator.
fn functional_traces(code: &[u8], l: &Launch, max_steps: usize) -> Result<Vec<cost::Trace>, String> {
    let wg = l.wg.unwrap_or([32, 1, 1]);
    let wg_id = l.wg_id.unwrap_or([0; 3]);
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
    func::PreparedDispatch::new(&d)?.trace_workgroup(&mem, wg_id)
}

/// Every SIMD group of the costed workgroup: `costed` alone without launch inputs (one 32-lane group), none when no
/// workgroup fits (`occ` has checked the sizes, which may not fit a u32).
fn workgroup_traces(code: &[u8], l: &Launch, costed: &Trace, occ: &Occupancy, max_steps: usize) -> Result<Vec<Trace>, String> {
    if occ.workgroups == 0 {
        return Ok(vec![]);
    }
    if !l.any() {
        return Ok(vec![costed.clone()]);
    }
    if l.functional() {
        return functional_traces(code, l, max_steps);
    }
    let wg = l.wg.unwrap_or([32, 1, 1]);
    (0..occ.groups_per_wg).map(|group| {
        let ctx = GroupCtx { wg_size: wg, wg_id: l.wg_id.unwrap_or([0; 3]), group, core: 0, slot: group, uniforms: l.uniforms.clone(), launch: l.sr_launch() };
        cost::trace_group(code, &ctx, max_steps)
    }).collect()
}

/// One past the highest general-purpose register half the code names (read or written, reachable or not, as a compiler
/// counts them): uniforms and special registers do not occupy the register file.
fn inferred_halves(code: &[u8], ins: &[cost::Instr]) -> Result<u32, String> {
    let mut top = 0;
    for r in ins {
        // `decode` works out these from fields that are not operands (an atomic's value), so take its sets as well
        for &h in r.dst.iter().chain(&r.src).chain(&r.late_src) {
            top = top.max(h + 1);
        }
        // not operand encodings: the 0x78 barrier and stack_adjust, whose stack_store fields hold a size
        if r.name == "barrier_4groups" || r.name == "stack_adjust" {
            continue;
        }
        let n0 = isa::opcode_to_number(&code[r.pc..code.len().min(r.pc + 16)]);
        let Some(d) = isa::get_instruction_descriptor(n0) else { continue };
        let n = if r.size >= 16 { n0 } else { n0 & ((1u128 << (8 * r.size)) - 1) };
        let mut halves = vec![];
        for o in d.decode_operands(n)? {
            o.regs_of(&mut halves);
        }
        if let Some(&h) = halves.iter().max() {
            top = top.max(h + 1);
        }
    }
    Ok(top)
}

/// Resident work on one core for a register count, workgroup size and threadgroup memory (`dispatch::occupancy`).
#[derive(Clone, Debug, PartialEq)]
struct Occupancy {
    /// halves `dispatch::occupancy` is given: `--regs` as declared, or the inferred count rounded up to the launch's unit
    halves: u32,
    source: &'static str,
    inferred_halves: u32,
    threads: u64,
    groups_per_wg: u32,
    /// SIMD groups the registers and the workgroup size allow
    group_cap: u32,
    /// whole workgroups that fit
    workgroups: u32,
    limited_by: &'static str,
    warnings: Vec<String>,
}

impl Occupancy {
    fn new(declared: u32, inferred_halves: u32, wg: [u32; 3], tg_bytes: u32) -> Occupancy {
        // SR19 reports the allocation as max(8, roundup(nr_gprs, 8)) (playground sr-resource-registers)
        let (halves, source) = if declared != 0 { (declared, "declared") } else { (inferred_halves.max(8).div_ceil(8) * 8, "inferred") };
        let launch = dispatch::Launch { regs: halves, wg, tg_bytes, ..Default::default() };
        let (group_cap, workgroups) = dispatch::occupancy(&launch);
        let threads = wg.iter().try_fold(1u64, |n, &v| n.checked_mul(v as u64)).filter(|&t| (1..=1024).contains(&t)).unwrap_or(0);
        let groups_per_wg = threads.div_ceil(32) as u32;
        // the register term alone: a workgroup size with no cap of its own and no threadgroup memory
        let register_cap = dispatch::occupancy(&dispatch::Launch { regs: halves, wg: [128, 1, 1], ..Default::default() }).0;
        let limited_by = if threads == 0 {
            "invalid_workgroup"
        } else if groups_per_wg > 0 && workgroups < group_cap / groups_per_wg {
            "threadgroup_memory"
        } else if group_cap < register_cap {
            "workgroup_size"
        } else if group_cap == 96 {
            "hardware_group_cap"
        } else {
            "registers"
        };
        let mut warnings = vec![];
        if declared != 0 && declared < inferred_halves {
            warnings.push(format!("--regs {} is below the {} halves the code names", declared, inferred_halves));
        }
        if declared % 8 != 0 {
            warnings.push(format!("--regs {} is not a multiple of 8: occupancy was measured at multiples of 8 and the launch rounds up (SR19)", declared));
        }
        if halves > 248 {
            warnings.push(format!("{} halves: occupancy was measured up to 248; the launch field wraps to 0 at 256, whose result is unknown", halves));
        }
        Occupancy { halves, source, inferred_halves, threads, groups_per_wg, group_cap, workgroups, limited_by, warnings }
    }

    fn resident_groups(&self) -> u32 {
        self.workgroups * self.groups_per_wg
    }

    fn resident_threads(&self) -> u64 {
        self.workgroups as u64 * self.threads
    }

    fn json(&self) -> String {
        format!(
            "{{\"register_halves\": {}, \"registers\": {}, \"register_source\": \"{}\", \"inferred_register_halves\": {}, \"workgroup_threads\": {}, \"simd_groups_per_workgroup\": {}, \"simd_group_cap_per_core\": {}, \"resident_simd_groups_per_core\": {}, \"resident_workgroups_per_core\": {}, \"resident_threads_per_core\": {}, \"limited_by\": \"{}\", \"warnings\": {}}}",
            self.halves, self.halves.div_ceil(2), self.source, self.inferred_halves, self.threads, self.groups_per_wg, self.group_cap,
            self.resident_groups(), self.workgroups, self.resident_threads(), self.limited_by, json_list(self.warnings.iter().cloned()),
        )
    }
}

/// One wave of a core: every resident workgroup starting together, as `dispatch::simulate`.
#[derive(Clone, Debug)]
struct CoreThroughput {
    clock_mhz: f64,
    threads: u64,
    /// real cycles with the memory hierarchy: `wave` all resident workgroups, `alone` one of them
    wave: f64,
    alone: f64,
    /// the same without it (device accesses as L1 hits): `agx_cost --groups`, and agx-bench's `tput`, generalised to
    /// SIMD groups that differ
    wave_l1: f64,
    alone_l1: f64,
    /// launching the wave's work on the core, which overlaps execution
    launch_us: f64,
    warnings: Vec<String>,
}

impl CoreThroughput {
    fn wave_us(&self, cycles: f64) -> f64 {
        (cycles / self.clock_mhz).max(self.launch_us)
    }

    fn json(&self) -> String {
        let r = |x: f64, nd: usize| py_round_repr(x, nd);
        format!(
            "{{\"threads_per_us\": {}, \"threads_per_us_l1_only\": {}, \"threads_per_wave\": {}, \"clock_mhz\": {}, \"wave_real_cycles\": {}, \"alone_real_cycles\": {}, \"contention\": {}, \"wave_real_cycles_l1_only\": {}, \"alone_real_cycles_l1_only\": {}, \"contention_l1_only\": {}, \"launch_us\": {}, \"launch_bound\": {}, \"validated_on_hardware\": false, \"warnings\": {}}}",
            r(self.threads as f64 / self.wave_us(self.wave), 2), r(self.threads as f64 / self.wave_us(self.wave_l1), 2), self.threads, r(self.clock_mhz, 1),
            r(self.wave, 1), r(self.alone, 1), r(self.wave / self.alone, 3), r(self.wave_l1, 1), r(self.alone_l1, 1), r(self.wave_l1 / self.alone_l1, 3),
            r(self.launch_us, 3), self.launch_us > self.wave / self.clock_mhz, json_list(self.warnings.iter().cloned()),
        )
    }
}

/// A trace's accesses as workgroup `k` of a wave makes them: device data `k` stretches on, and each SIMD group's stack
/// block its own (`func`: one block per slot, 2^28 bytes apart).
fn own_data(t: &Trace, k: u64, groups_per_wg: u64, stretch: u64, device: &HashSet<usize>, stack: &HashSet<usize>) -> Vec<(u32, Vec<u64>)> {
    t.mem.iter().map(|(step, segs)| {
        let pc = t.pcs[*step as usize];
        let by = if device.contains(&pc) {
            k.wrapping_mul(stretch)
        } else if stack.contains(&pc) {
            (k * groups_per_wg) << (28 - 6)
        } else {
            0
        };
        (*step, segs.iter().map(|s| s.wrapping_add(by)).collect())
    }).collect()
}

/// Simulate one core at `occ`'s occupancy with the group simulation `dispatch::simulate` uses.
///
/// Every resident workgroup runs the costed workgroup's path, so a bound in the launch inputs (`tid < n`) cannot empty
/// the others. Its device accesses are moved to data of its own (the next stretch of the same size), on a GPU with every
/// core as busy. This is an estimate: read-only data the workgroups share (a BVH) makes the memory figure pessimistic, and
/// `_l1_only` then bounds it from above. Unvalidated against hardware.
fn core_throughput(ins: &[cost::Instr], wg: &[Trace], occ: &Occupancy, p: &Params) -> Result<CoreThroughput, String> {
    if occ.workgroups == 0 {
        return Err(format!("no {}-thread workgroup fits a core with {} register halves and this threadgroup memory", occ.threads, occ.halves));
    }
    if wg.len() != occ.groups_per_wg as usize {
        return Err(format!("{} SIMD groups traced for a workgroup of {}", wg.len(), occ.groups_per_wg));
    }
    let launch = dispatch::Launch::default();
    let pcs = |a: &str, b: &str| -> HashSet<usize> { ins.iter().filter(|i| i.name == a || i.name == b).map(|i| i.pc).collect() };
    let (device, stack) = (pcs("device_load", "device_store"), pcs("stack_load", "stack_store"));
    let mut segments: HashSet<u64> = HashSet::new();
    for t in wg {
        for (step, segs) in &t.mem {
            if device.contains(&t.pcs[*step as usize]) {
                segments.extend(segs.iter().copied());
            }
        }
    }
    let stretch = match (segments.iter().min(), segments.iter().max()) {
        (Some(lo), Some(hi)) => hi - lo + 1,
        _ => 0,
    };
    let footprint = segments.len() as f64 * 64.0 * occ.workgroups as f64 * launch.cores as f64;
    let mut mm = MemModel { fetch_line_ns: dispatch::fetch_line_ns(), icache_bytes: dispatch::icache_bytes(), clock_mhz: launch.clock_mhz,
                            active_cores: launch.cores as f64, warm: false, dram_gbps: launch.dram_gbps, ..Default::default() };
    mm.resident_slc = footprint <= mm.slc_bytes as f64;
    let run = |workgroups: u64, memory: bool| -> Result<f64, String> {
        let pick = || (0..workgroups).flat_map(|k| wg.iter().map(move |t| (k, t)));
        let seqs: Vec<&[usize]> = pick().map(|(_, t)| t.pcs.as_slice()).collect();
        let masked: Vec<&[u32]> = pick().map(|(_, t)| t.masked.as_slice()).collect();
        let wg_of: Vec<u32> = pick().map(|(k, _)| k as u32).collect();
        if !memory {
            return Ok(cost::simulate_groups(ins, &seqs, Some(&masked), None, p, false, Some(&wg_of))?.0);
        }
        let owned: Vec<Vec<(u32, Vec<u64>)>> = pick().map(|(k, t)| own_data(t, k, wg.len() as u64, stretch, &device, &stack)).collect();
        let mems: Vec<&[(u32, Vec<u64>)]> = owned.iter().map(|m| m.as_slice()).collect();
        Ok(cost::simulate_groups(ins, &seqs, Some(&masked), Some((mems.as_slice(), &mm)), p, false, Some(&wg_of))?.0)
    };
    let n = occ.workgroups as u64;
    let mut warnings = vec![];
    if wg.iter().any(|t| t.skipped.contains("stack_load")) {
        // the emulation reads spills back as zero: spilled counters end loops early, or never
        warnings.push("the path skips stack_load, so spilled values read as zero and its control flow may be wrong: use --map / --alloc (the functional path) to compare spilling code".to_string());
    }
    let launch_ns = (n as f64 * launch.wg_launch_ns).max(occ.resident_groups() as f64 * launch.group_launch_ns);
    Ok(CoreThroughput {
        clock_mhz: launch.clock_mhz,
        threads: occ.resident_threads(),
        wave: run(n, true)?,
        alone: run(1, true)?,
        wave_l1: run(n, false)?,
        alone_l1: run(1, false)?,
        launch_us: launch_ns / 1000.0,
        warnings,
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut code_path = None;
    let (mut groups, mut units, mut allow_unknown, mut annotate, mut load_latency) = (1usize, "real".to_string(), false, false, None);
    let mut l = Launch::default();
    let mut want_throughput = false;
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
            "--agx-strict" => {
                let g = agx_sim::isa::mesa::MesaGen::parse(&value()).unwrap_or_else(|e| { eprintln!("agx-cost: {}", e); exit(2) });
                agx_sim::isa::mesa::set_mode(Some(g));
            }
            "--core-throughput" => want_throughput = true,
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
        let (cyc, stats) = cost::simulate_trace(&ins, &tr, &p, groups, true)?;
        let stats = stats.unwrap();
        let scale = if units == "real" { 1.0 } else { 1.0 / REAL_CYCLES_PER_UNIT };
        let mut unmeasured: Vec<String> = ins.iter().filter(|r| r.unmeasured).map(|r| r.name.clone()).collect();
        unmeasured.sort();
        unmeasured.dedup();
        let paths = paths.map(|k| format!(", \"paths\": \"{}\"", k)).unwrap_or_default();
        let occ = Occupancy::new(l.regs, inferred_halves(&code, &ins)?, l.wg.unwrap_or([32, 1, 1]), l.tg_bytes);
        let mut occupancy = format!(", \"occupancy\": {}", occ.json());
        if want_throughput {
            // in real cycles whatever `--units` says, and never a reason to lose the fields above
            let estimate = workgroup_traces(&code, &l, &tr, &occ, 500_000).and_then(|wg| core_throughput(&ins, &wg, &occ, &p));
            let field = estimate.map(|t| t.json()).unwrap_or_else(|e| format!("{{\"error\": {}}}", serde_json::to_string(&e).unwrap()));
            occupancy += &format!(", \"core_throughput\": {}", field);
        }
        println!(
            "{{\"instructions\": {}, \"dynamic\": {}, \"groups\": {}, \"units\": {}, \"cycles\": {}, \"emulator_skipped\": {}, \"warnings\": {}, \"unmeasured_classes\": {}{}{}}}",
            ins.len(),
            tr.pcs.len(),
            groups,
            serde_json::to_string(&units).unwrap(),
            py_round_repr(cyc * scale, 1),
            json_list(tr.skipped.iter().cloned()),
            json_list(tr.warnings.iter().cloned()),
            json_list(unmeasured),
            paths,
            occupancy
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

#[cfg(test)]
mod tests {
    use super::*;
    use agx_sim::kernels::{bench_loop, e, mov, store};

    const STOP: [u8; 2] = [0x88, 0];

    fn occ(halves: u32, wg: [u32; 3], tg: u32) -> Occupancy {
        Occupancy::new(halves, 0, wg, tg)
    }

    fn fitted() -> Params {
        let mut p = Params::new().unwrap();
        cost::fit_branch(&mut p).unwrap();
        p
    }

    fn estimate(code: &[u8], o: &Occupancy) -> CoreThroughput {
        let p = fitted();
        let ins = cost::decode(code, &p).unwrap();
        let wg: Vec<Trace> = (0..o.groups_per_wg).map(|group| {
            let ctx = GroupCtx { wg_size: [o.threads as u32, 1, 1], wg_id: [0; 3], group, core: 0, slot: group, uniforms: vec![(4, 0x10000), (5, 0)], launch: SrLaunch { regs: o.halves, ..Default::default() } };
            cost::trace_group(code, &ctx, 500_000).unwrap()
        }).collect();
        core_throughput(&ins, &wg, o, &p).unwrap()
    }

    #[test]
    fn register_counts_set_the_resident_groups() {
        // 128 registers: 12 groups; 80 registers: 20 (32-lane workgroups: one group each)
        let o = occ(256, [32, 1, 1], 0);
        assert_eq!((o.group_cap, o.workgroups, o.resident_groups(), o.resident_threads()), (12, 12, 12, 384));
        assert_eq!(o.limited_by, "registers");
        let o = occ(160, [32, 1, 1], 0);
        assert_eq!((o.group_cap, o.workgroups, o.resident_groups(), o.resident_threads()), (20, 20, 20, 640));
        assert_eq!(o.limited_by, "registers");
        // the measured table (playground 2026-10-11-occupancy), 128-lane workgroups: only whole workgroups count
        for (halves, cap, wgs) in [(16, 96, 24), (32, 96, 24), (40, 82, 20), (48, 68, 17), (64, 52, 13), (80, 40, 10), (96, 34, 8), (128, 26, 6), (160, 20, 5), (192, 16, 4), (248, 12, 3)] {
            let o = occ(halves, [128, 1, 1], 0);
            assert_eq!((o.group_cap, o.workgroups, o.resident_groups()), (cap, wgs, wgs * 4), "{} halves", halves);
        }
    }

    #[test]
    fn small_workgroups_hit_their_own_caps() {
        let o = occ(16, [32, 1, 1], 0);
        assert_eq!((o.group_cap, o.workgroups, o.resident_threads(), o.limited_by), (55, 55, 1760, "workgroup_size"));
        // a partly filled group is still a group, and still one slot
        let o = occ(16, [8, 1, 1], 0);
        assert_eq!((o.group_cap, o.workgroups, o.resident_groups(), o.resident_threads()), (55, 55, 55, 440));
        let o = occ(16, [64, 1, 1], 0);
        assert_eq!((o.group_cap, o.workgroups, o.resident_groups(), o.resident_threads(), o.limited_by), (84, 42, 84, 2688, "workgroup_size"));
        let o = occ(16, [8, 8, 1], 0);
        assert_eq!((o.group_cap, o.workgroups), (84, 42));
        // above those caps the registers decide
        assert_eq!(occ(64, [32, 1, 1], 0).group_cap, 52);
        assert_eq!(occ(48, [64, 1, 1], 0).group_cap, 68);
        assert_eq!(occ(16, [128, 1, 1], 0).limited_by, "hardware_group_cap");
    }

    #[test]
    fn threadgroup_memory_and_bad_launches() {
        let o = occ(16, [128, 1, 1], 12288);
        assert_eq!((o.group_cap, o.workgroups, o.resident_groups(), o.limited_by), (96, 5, 20, "threadgroup_memory"));
        for o in [occ(16, [2048, 1, 1], 0), occ(16, [u32::MAX; 3], 0), occ(4096, [32, 1, 1], 0)] {
            assert_eq!(o.workgroups, 0);
            assert_eq!(o.resident_threads(), 0);
        }
        // never a division by zero
        let err = core_throughput(&[], &[], &occ(4096, [32, 1, 1], 0), &fitted()).unwrap_err();
        assert!(err.contains("fits"), "{}", err);
    }

    #[test]
    fn registers_are_inferred_from_the_code_and_declarations_win() {
        let p = fitted();
        let scan = |code: &[u8]| inferred_halves(code, &cost::decode(code, &p).unwrap()).unwrap();
        assert_eq!(scan(&STOP), 0);
        // r79 is halves 158 and 159: 80 registers, 160 halves
        assert_eq!(scan(&[e("iadd r79, r0, 0"), STOP.to_vec()].concat()), 160);
        assert_eq!(scan(&[e("iadd r127, r0, 0"), STOP.to_vec()].concat()), 256);
        // a 16-bit half, a source, and a 64-bit pair
        assert_eq!(scan(&[e("mov r3h, 1"), STOP.to_vec()].concat()), 8);
        assert_eq!(scan(&[e("iadd r1, r9, 0"), STOP.to_vec()].concat()), 20);
        assert_eq!(scan(&[e("iadd r1_r2, r10_r11, 0"), STOP.to_vec()].concat()), 24);
        // a stored register, uniforms not counted
        assert_eq!(scan(&[store(30, 4, 2), STOP.to_vec()].concat()), 62);
        assert_eq!(scan(&[e("iadd r1, u60, 0"), STOP.to_vec()].concat()), 4);
        // padding after the stop is not code
        assert_eq!(scan(&[mov(9, 1), STOP.to_vec(), vec![0; 16]].concat()), 20);

        let o = Occupancy::new(0, 154, [32, 1, 1], 0);
        assert_eq!((o.halves, o.source, o.group_cap), (160, "inferred", 20));
        assert_eq!(Occupancy::new(0, 0, [32, 1, 1], 0).halves, 8);
        // between the measured multiples of 8 the launch's unit decides: 161 is allocated as 168
        assert_eq!(Occupancy::new(0, 161, [32, 1, 1], 0).group_cap, 18);
        let o = Occupancy::new(256, 160, [32, 1, 1], 0);
        assert_eq!((o.halves, o.source, o.group_cap), (256, "declared", 12));
        assert!(o.warnings.iter().any(|w| w.contains("measured up to 248")));
        assert!(Occupancy::new(160, 200, [32, 1, 1], 0).warnings.iter().any(|w| w.contains("below")));
        assert!(Occupancy::new(161, 0, [32, 1, 1], 0).warnings.iter().any(|w| w.contains("multiple of 8")));
        assert!(Occupancy::new(160, 160, [32, 1, 1], 0).warnings.is_empty());
    }

    #[test]
    fn throughput_is_threads_over_the_simulated_wave() {
        let code = [bench_loop("fadd32", 1, 4, 2000), STOP.to_vec()].concat();
        let o = occ(160, [32, 1, 1], 0);
        let t = estimate(&code, &o);
        assert_eq!((t.threads, t.clock_mhz), (640, 1398.0));
        assert!(t.wave >= t.alone && t.wave_l1 >= t.alone_l1, "{:?}", t);
        assert_eq!(t.launch_us, 2.0); // 20 workgroups at 100 ns
        let j: serde_json::Value = serde_json::from_str(&t.json()).unwrap();
        let want = 640.0 / (t.wave / 1398.0);
        assert!((j["threads_per_us"].as_f64().unwrap() - want).abs() < 0.01, "{}", j);
        assert_eq!(j["validated_on_hardware"], false);
        assert_eq!(j["launch_bound"], false);
    }

    #[test]
    fn the_l1_figure_is_the_groups_option() {
        // what `--groups 20` (agx-bench's tput) computes for identical groups
        let code = [bench_loop("fadd32", 1, 4, 500), STOP.to_vec()].concat();
        let p = fitted();
        let ins = cost::decode(&code, &p).unwrap();
        let tr = cost::trace(&code, 32, 500_000).unwrap();
        let groups = cost::simulate_trace(&ins, &tr, &p, 20, false).unwrap().0;
        let t = core_throughput(&ins, std::slice::from_ref(&tr), &occ(160, [32, 1, 1], 0), &p).unwrap();
        assert_eq!(t.wave_l1, groups);
        assert_eq!(t.alone_l1, cost::simulate_trace(&ins, &tr, &p, 1, false).unwrap().0);
    }

    #[test]
    fn more_resident_groups_help_latency_bound_code_and_not_alu_bound_code() {
        // a long-latency chain per group: groups hide each other's latency, so threads per us scale with occupancy
        let latency = [bench_loop("fmul32", 1, 4, 2000), STOP.to_vec()].concat();
        let (few, many) = (estimate(&latency, &occ(256, [32, 1, 1], 0)), estimate(&latency, &occ(160, [32, 1, 1], 0)));
        let gain = |a: &CoreThroughput, b: &CoreThroughput| (b.threads as f64 / b.wave_us(b.wave)) / (a.threads as f64 / a.wave_us(a.wave));
        assert!(gain(&few, &many) > 1.0, "{:?} {:?}", few, many);
        assert!(many.wave / many.alone >= few.wave / few.alone);
        // scheduler-saturating chains: the core is already full, so more residents add contention, not throughput
        let busy = [bench_loop("fadd32", 8, 8, 2000), STOP.to_vec()].concat();
        let (few, many) = (estimate(&busy, &occ(256, [32, 1, 1], 0)), estimate(&busy, &occ(160, [32, 1, 1], 0)));
        assert!(gain(&few, &many) < 1.67, "{:?} {:?}", few, many); // 20 / 12: never the full ratio
        assert!(many.wave > few.wave, "{:?} {:?}", few, many);
    }

    #[test]
    fn stack_blocks_and_device_data_are_each_workgroups_own() {
        let device = HashSet::from([0usize]);
        let stack = HashSet::from([8usize]);
        let t = Trace { pcs: vec![0, 8, 16], mem: vec![(0, vec![100, 101]), (1, vec![1 << 22]), (2, vec![7])], ..Default::default() };
        // workgroup 0 is as traced
        assert_eq!(own_data(&t, 0, 2, 10, &device, &stack), t.mem);
        // workgroup 3 of 2-group workgroups: device 3 stretches on, stack 6 slots on; atomics and threadgroup words stay
        assert_eq!(own_data(&t, 3, 2, 10, &device, &stack), vec![(0, vec![130, 131]), (1, vec![7 << 22]), (2, vec![7])]);
    }

    #[test]
    fn a_stack_load_the_emulation_skips_is_reported() {
        let p = fitted();
        let code = [e("iadd r1, r2, 0"), STOP.to_vec()].concat();
        let ins = cost::decode(&code, &p).unwrap();
        let mut tr = cost::trace(&code, 32, 100).unwrap();
        let o = occ(160, [32, 1, 1], 0);
        assert!(core_throughput(&ins, std::slice::from_ref(&tr), &o, &p).unwrap().warnings.is_empty());
        tr.skipped.insert("stack_load".to_string());
        let t = core_throughput(&ins, std::slice::from_ref(&tr), &o, &p).unwrap();
        assert!(t.warnings[0].contains("--alloc"), "{:?}", t.warnings);
        assert!(t.json().contains("stack_load"));
    }

    #[test]
    fn each_workgroup_gets_its_own_data_and_its_own_barrier_scope() {
        // one store of r5 at r2: every group alone hits its own lines
        let code = [mov(2, 0), mov(5, 1), store(5, 4, 2), STOP.to_vec()].concat();
        let o = occ(64, [64, 1, 1], 0);
        let t = estimate(&code, &o);
        assert_eq!(o.groups_per_wg, 2);
        assert_eq!(t.threads, o.resident_threads());
        assert!(t.wave.is_finite() && t.wave > 0.0);
        // a 4-group barrier in 128-lane workgroups: completes, rather than waiting on the whole core's groups
        let code = [vec![0x78, 0], STOP.to_vec()].concat();
        let o = occ(64, [128, 1, 1], 0);
        let t = estimate(&code, &o);
        assert!(t.wave.is_finite() && t.wave < 100_000.0, "{:?}", t);
        assert_eq!(t.threads, 13 * 128);
    }
}
