//! Dispatch-level model: one compute dispatch on the whole GPU.
//!
//! * Occupancy (playground novel-results/2026-10-11-occupancy): resident SIMD groups per core =
//!   min(96, 2 * floor(3328 / (2 * register halves))); 32-lane workgroups cap near 55 groups and 64-lane
//!   ones near 84; whole workgroups only. Threadgroup memory: `tg_capacity`
//!   = 64 KiB per core, measured: floor(65536 / declared bytes) workgroups (playground occupancy finding).
//! * Workgroups are split evenly over the GPU clusters (38 cores: 10, 9, 9, 10) and then over the cores of
//!   each cluster, so the busiest core gets ceil(ceil(n / 4) / 9) workgroups (wg_placement.py: exact for
//!   38..608 workgroups of 64 and 256 lanes). That core sets the time.
//! * That core runs its workgroups in waves of `wgs_per_core`; each simulated wave traces every SIMD group
//!   with its own geometry (`cost::trace_group`) and runs `cost::simulate_groups` with the memory model.
//!   With many waves, `max_waves` of them are simulated (the first, the last, the middle, then 1, 3, 7, ...) and
//!   the others interpolated linearly between the nearest simulated full waves: a grid whose work is concentrated
//!   in its first workgroups (threads past a data-dependent count exit at once: Geekbench 7's sort_keypoints,
//!   compute_keypoints, nonmax_suppression) is not taken to cost its first wave every time.
//! * Launching costs the busiest core max(`wg_launch_ns` per workgroup, `group_launch_ns` per SIMD group)
//!   (empty kernels over 4M-16M threads); it overlaps execution, so the core time is the larger of the two.
//! * time = core time + `launch_ns` (fixed per-dispatch overhead, 5.6 us for a 32-thread empty dispatch).

use crate::cost::{self, GroupCtx, MemModel, Params, Trace};

/// Dynamic paths of every SIMD group of one workgroup (by workgroup id), in place of the cost model's own emulation;
/// `func::trace_workgroups` supplies them for code that reads real memory (driver preambles, root tables, data).
pub type Tracer<'a> = dyn Fn([u32; 3]) -> Result<Vec<Trace>, String> + Sync + 'a;

#[derive(Clone, Debug)]
pub struct Launch {
    /// global size in threads (x, y, z)
    pub grid: [u32; 3],
    /// workgroup size in threads
    pub wg: [u32; 3],
    /// declared registers in 16-bit halves
    pub regs: u32,
    pub tg_bytes: u32,
    pub tg_capacity: u32,
    pub cores: u32,
    /// Optional measured peak workgroup count on one core. This controls the
    /// representative core's waves; it does not prescribe workgroup identities,
    /// simultaneous residency, active cores or a full placement map.
    pub busiest_core_wgs_override: Option<u64>,
    pub clock_mhz: f64,
    pub launch_ns: f64,
    pub wg_launch_ns: f64,
    pub group_launch_ns: f64,
    /// per distinct 64-byte line of code the busiest core's first wave executes: the instruction fetch of a command
    /// whose caches start cold, overlapping execution (0 in the direct-DRM profile, whose calibration includes it; 104 ns for the CUDA driver's
    /// commands: straight-line code 294 -> 4134 bytes, +6.25 us, playground novel-results/2026-11-01-code-fetch)
    pub fetch_line_ns: f64,
    /// stack (scratch) per thread the kernel uses, and the fixed cost of a command whose kernel has one (the scratch
    /// set up by the launch's helper program: 0 in the drm profile; 5.0 us for the CUDA driver's: real one-group stack kernels
    /// take 7.7-8.9 us in all at a warm clock (results/2026-11-07-stack-slots), while an empty kernel over 38 to
    /// 15,200 workgroups +7.1-8.1 us with --stack 64, playground novel-results/2026-11-01-stack-launch)
    pub stack_bytes: u32,
    pub stack_launch_ns: f64,
    /// per SIMD-group slot of the busiest core that a stack-using kernel occupies in a command (resident workgroups x
    /// groups): the first stack use of each slot costs this, one slot at a time per core, before its group runs (bfs
    /// Kernel2 at 1-1954 workgroups of 512 threads: +1.0-1.3 us per slot up to the 96 resident; 0 in the drm profile;
    /// runtime 1150 ns; results/2026-11-07-stack-slots). AGX_STACK_SLOT_NS overrides
    pub stack_slot_ns: f64,
    /// instruction-cache refetch for kernels whose executed code exceeds the cache: when a wave's distinct 64-byte
    /// lines exceed `icache_lines`, the simulation fetches through a per-core LRU of that many lines (a fall-through
    /// miss `refetch_line_ns`, a taken jump's `fetch_jump_ns`, the group waiting) in place of `fetch_line_ns` per
    /// distinct line. 0: off (drm profile). runtime: 168 lines (12 KiB with the 8-byte-crossing loss of 6-byte streams,
    /// playground novel-results/2026-10-20-icache), 64 / 200 ns (2026-11-07-uncached-jump-fetch). AGX_ICACHE_LINES overrides
    pub icache_lines: usize,
    /// the instruction cache's replacement: one of this many least recently used lines (`cost::MemModel::icache_victims`)
    pub icache_victims: usize,
    pub refetch_line_ns: f64,
    pub fetch_jump_ns: f64,
    /// GPU-wide store bandwidth (GB/s), a floor on the time of a dispatch's writes (0: none, the drm profile): stores
    /// that fit the system-level cache went at no cost in the model, measured at ~280-340 GB/s whatever their size
    /// (1520 x 256 threads writing 1.5 / 6 / 24 MB: 9.9 / 22.0 / 72.1 us, after the workgroups' launch; runtime profile
    /// 365, playground novel-results/2026-11-01-store-bandwidth)
    pub store_gbps: f64,
    /// fixed cost of a command whose kernel stores to device memory (the writes reaching memory before the command
    /// ends): 0 in drm; 700 ns in runtime (one workgroup storing 1 KB: 3.0 us against the empty kernel's 2.3)
    pub store_launch_ns: f64,
    /// the waves after the first start with their accesses cached when the dispatch's data fits the system-level cache
    /// (false in drm, whose calibration it would change; true in runtime: Geekbench's captures median 9% -> 8%, llama.cpp's
    /// within 10% 7 -> 9 of 11)
    pub later_waves_warm: bool,
    /// with `later_waves_warm`: only the first wave's data is warm for later waves; the rest of the data a previous
    /// command left behind is at L2 latency but its first touch in the dispatch goes at the DRAM rate (reads back to
    /// back never exceed ~380 GB/s; results/2026-11-01-fft-shared-captures). The runtime profile sets it
    /// (AGX_PARTIAL_WARM=0 / 1 overrides)
    pub partial_warm: bool,
    /// simulate each wave chained with the next in every slot (`sim_wave`); the runtime profile sets it
    pub chain_waves: bool,
    /// cores per cluster (workgroups are split evenly over clusters first); empty = round-robin over `cores`
    pub clusters: Vec<u32>,
    pub uniforms: Vec<(usize, u32)>,
    pub mem: bool,
    /// the buffers the launch's memory maps (start, bytes), when known (agx_dispatch --map): the GPU-wide footprint
    /// estimate of each is capped at its size
    pub regions: Vec<(u64, u64)>,
    /// effective DRAM bandwidth of a whole dispatch (GB/s): 360 fits the streaming and reduction dispatches;
    /// steady per-core streaming (mem_probes) reaches ~410
    pub dram_gbps: f64,
    /// Representative-core path only: simulate at most this many distinct waves (first, last, middle, then 1, 3, 7,
    /// 15, ...); others are interpolated. The supplied-map path is exhaustive.
    pub max_waves: usize,
    pub max_steps: usize,
}

impl Default for Launch {
    fn default() -> Self {
        Launch {
            grid: [32, 1, 1],
            wg: [32, 1, 1],
            regs: 64,
            tg_bytes: 0,
            tg_capacity: 64 << 10,
            cores: 38,
            busiest_core_wgs_override: None,
            clock_mhz: 1398.0,
            launch_ns: 5600.0,
            wg_launch_ns: 100.0,
            group_launch_ns: 13.6,
            fetch_line_ns: 0.0,
            stack_bytes: 0,
            stack_launch_ns: 0.0,
            stack_slot_ns: 0.0,
            icache_lines: 0,
            icache_victims: 1,
            refetch_line_ns: 64.0,
            fetch_jump_ns: 200.0,
            store_gbps: 0.0,
            store_launch_ns: 0.0,
            later_waves_warm: false,
            partial_warm: false,
            chain_waves: false,
            clusters: vec![10, 9, 9, 10],
            uniforms: vec![],
            mem: true,
            regions: Vec::new(),
            dram_gbps: std::env::var("AGX_DRAM_GBPS").ok().and_then(|v| v.parse().ok()).unwrap_or(360.0),
            max_waves: 8,
            max_steps: 20_000_000,
        }
    }
}

struct Geometry {
    dims: [u64; 3],
    workgroups: u64,
    threads: u32,
}

impl Launch {
    /// Check the model's input domain before decoding, tracing or allocating
    /// workgroup state. This is not a claim of hardware support for every input.
    pub fn validate(&self) -> Result<(), String> {
        self.geometry().map(|_| ())
    }

    fn geometry(&self) -> Result<Geometry, String> {
        if self.grid.contains(&0) { return Err("grid dimensions must be nonzero (empty dispatches are not modelled)".into()); }
        let threads = self.wg.iter().try_fold(1u32, |n, &v| n.checked_mul(v)).ok_or("workgroup size overflows")?;
        if threads == 0 || threads > 1024 { return Err("workgroup must contain 1..1024 threads with nonzero dimensions".into()); }
        if self.cores == 0 || self.clusters.contains(&0) { return Err("core and cluster counts must be nonzero".into()); }
        if !self.clock_mhz.is_finite() || self.clock_mhz <= 0.0 { return Err("clock_mhz must be finite and positive".into()); }
        if !self.dram_gbps.is_finite() || self.dram_gbps <= 0.0 { return Err("dram_gbps must be finite and positive".into()); }
        for (name, v) in [("launch_ns", self.launch_ns), ("wg_launch_ns", self.wg_launch_ns), ("group_launch_ns", self.group_launch_ns)] {
            if !v.is_finite() || v < 0.0 { return Err(format!("{} must be finite and nonnegative", name)); }
        }
        if self.uniforms.iter().any(|&(u, _)| u >= 256) { return Err("uniform register index exceeds u255".into()); }
        if self.tg_bytes > self.tg_capacity { return Err("threadgroup memory exceeds the configured per-core capacity".into()); }
        let (cap, resident) = occupancy(self);
        let gpw = threads.div_ceil(32);
        if resident == 0 { return Err(format!("workgroup of {} SIMD groups exceeds the {}-group occupancy limit", gpw, cap)); }
        let dims: [u64; 3] = std::array::from_fn(|i| self.grid[i].div_ceil(self.wg[i]) as u64);
        let workgroups = dims.iter().try_fold(1u64, |n, &d| n.checked_mul(d)).ok_or("workgroup count exceeds u64")?;
        if let Some(peak) = self.busiest_core_wgs_override {
            if peak < workgroups.div_ceil(self.cores as u64) || peak > workgroups {
                return Err("busiest_core_wgs must lie between ceil(workgroups / cores) and workgroups".into());
            }
        }
        Ok(Geometry { dims, workgroups, threads })
    }
}

fn wg_coords(dims: [u64; 3], k: u64) -> [u32; 3] {
    [(k % dims[0]) as u32, ((k / dims[0]) % dims[1]) as u32, (k / dims[0] / dims[1]) as u32]
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub workgroups: u64,
    pub groups_per_wg: u32,
    pub group_cap_per_core: u32,
    pub wgs_per_core: u32,
    pub busiest_core_wgs: u64,
    pub waves: u64,
    /// (wave index, workgroups in it, core cycles)
    pub simulated_waves: Vec<(u64, u64, f64)>,
    pub core_cycles: f64,
    pub launch_core_ns: f64,
    /// distinct 64-byte code lines the first wave executed (the fetch cost, `fetch_line_ns` each)
    pub code_lines: u64,
    pub time_ns: f64,
    pub warnings: Vec<String>,
}

/// (resident SIMD groups per core allowed by registers and workgroup size, resident workgroups per core).
/// Invalid/impossible resource declarations have zero resident workgroups; never panic on them.
pub fn occupancy(l: &Launch) -> (u32, u32) {
    let halves = l.regs.max(8) as u64;
    let mut cap = 96u32.min((2 * (3328 / (2 * halves))) as u32);
    let Some(threads) = l.wg.iter().try_fold(1u32, |n, &v| n.checked_mul(v)) else { return (0, 0); };
    if threads == 0 || threads > 1024 { return (0, 0); }
    if threads <= 32 {
        cap = cap.min(55);
    } else if threads <= 64 {
        cap = cap.min(84);
    }
    let gpw = threads.div_ceil(32);
    let mut wgs = cap / gpw;
    if l.tg_bytes > 0 {
        wgs = wgs.min(l.tg_capacity / l.tg_bytes);
    }
    (cap, wgs)
}

/// Workgroups per core guaranteed to make progress together in the bounded
/// B5 probe. This is a concurrency bound for cooperative launches, not the
/// physical occupancy used by the throughput wave model above.
pub fn co_resident_workgroups(l: &Launch) -> u32 {
    let Some(threads) = l.wg.iter().try_fold(1u32, |n, &v| n.checked_mul(v)) else { return 0; };
    if threads == 0 || threads > 1024 { return 0; }
    let halves = (l.regs.max(8) as u64).div_ceil(8) * 8;
    let cap = (2 * (1664 / halves)).min(96) as u32;
    let fit = cap / threads.div_ceil(32);
    let mut guaranteed = if fit <= 1 { fit } else { (fit - 1).min(15) };
    if l.tg_bytes > 0 {
        guaranteed = guaranteed.min(l.tg_capacity / l.tg_bytes);
    }
    guaranteed
}

/// Cold instruction fetch, ns per 64 bytes, on top of issue: 8.85 ns per 8-byte instruction of straight-line code on the M2 Max, warm, back to
/// back, from 64 KB to 4 MB of code and 1 or 12 SIMD groups alike (scripts/probe_fetch_scaling.py). A launch's kernel is
/// fetched again each launch. Off by default: the calibrated launch profiles cost the fetch per launch instead
/// (`Launch::fetch_line_ns`, 104 ns per line in the runtime profile, fitted on 56 captures). AGX_FETCH_LINE_NS=62 turns
/// this in-simulation fetch on (it stays off when the launch profile costs the fetch).
pub fn fetch_line_ns() -> f64 {
    std::env::var("AGX_FETCH_LINE_NS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)
}

/// AGX_REFILL=1: a workgroup starts as soon as a finished one has freed its registers, in place of waves that each wait for
/// their slowest workgroup. Not established on hardware: off by default until it is compared with measured launches.
pub fn refill() -> bool {
    std::env::var("AGX_REFILL").is_ok_and(|v| v == "1")
}

/// When `durations` (in start order) are done with `slots` of them running at once, each starting when the first slot
/// frees: list scheduling, whose result is at most the sum of the waves' slowest and at least the longest.
pub fn refill_makespan(durations: &[f64], slots: usize) -> f64 {
    let mut free = vec![0.0f64; slots.max(1)];
    let mut end = 0.0f64;
    for &d in durations {
        let (i, &t) = free.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        free[i] = t + d;
        end = end.max(free[i]);
    }
    end
}

/// The latest finish among each workgroup's SIMD groups, by workgroup (`wg_of` as `cost::simulate_groups`).
pub fn workgroup_finish(finish: &[f64], wg_of: &[u32], count: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; count];
    for (&f, &w) in finish.iter().zip(wg_of) {
        out[w as usize] = out[w as usize].max(f);
    }
    out
}

/// The instruction cache's capacity for MemModel::icache_bytes. AGX_ICACHE_BYTES overrides it.
pub fn icache_bytes() -> usize {
    std::env::var("AGX_ICACHE_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

pub fn simulate(code: &[u8], l: &Launch, p: &Params) -> Result<Report, String> {
    simulate_with(code, l, p, None)
}

/// `simulate` with the SIMD groups' paths from `tracer` (None: `cost::trace_group`).
///
/// A grid of at most one workgroup per core ends with its slowest workgroup, which need not be the first: every
/// workgroup is simulated (each alone on its core) and the slowest sets the time. Rodinia particlefilter's `kernel`
/// (8 x 128 threads, each searching the CDF for its particle: thread i loops ~i times) ran 88 us on the GPU while its
/// first workgroup alone takes 13 us. AGX_ALL_WGS=0 simulates the first workgroup only.
pub fn simulate_with(code: &[u8], l: &Launch, p: &Params, tracer: Option<&Tracer>) -> Result<Report, String> {
    let Geometry { workgroups: n, .. } = l.geometry()?;
    let (_, r) = occupancy(l);
    let cores = l.cores as u64;
    let one_per_core = l.busiest_core_wgs_override.is_none() && r > 0 && cores > 0 && n > 1 && n <= cores
        && busiest_core_wgs(n, l) <= 1 && std::env::var("AGX_ALL_WGS").map_or(true, |v| v != "0");
    // Likewise a grid that is resident all at once (one wave) ends with its slowest core: each core's workgroups
    // (off, off + cores, ...) are simulated and the slowest core sets the time. Cycles' intersect_closest (330 x 64
    // threads, BVH traversal): 325-386 us over the cores, 335 for core 0. AGX_ALL_CORES=0: core 0 only.
    let one_wave = l.busiest_core_wgs_override.is_none() && r > 0 && cores > 1 && n > cores
        && n.div_ceil(cores) <= r as u64 && std::env::var("AGX_ALL_CORES").map_or(true, |v| v != "0");
    if !one_per_core && !one_wave {
        let off = std::env::var("AGX_CORE_OFF").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        return simulate_core(code, l, p, tracer, off);
    }
    // one core at a time (each simulation is parallel inside; 38 at once ran the host out of memory on heartwall).
    // The cores are sampled evenly by cost: about 2e7 simulated core cycles in all (heartwall, 15M cycles per core,
    // 60 s of simulation each: 2 cores; intersect_closest, 0.47M: all 38). AGX_CORE_CYCLE_BUDGET overrides.
    let first = simulate_core(code, l, p, tracer, 0)?;
    let budget = std::env::var("AGX_CORE_CYCLE_BUDGET").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(2e7);
    let k = n.min(cores);
    let samples = ((budget / first.core_cycles.max(1.0)) as u64).clamp(2, k);
    let offs: Vec<u64> = (1..samples).map(|j| j * k / samples).filter(|&o| o > 0).collect();
    let reps = std::iter::once(Ok(first)).chain(offs.iter().map(|&off| simulate_core(code, l, p, tracer, off)));
    let mut best: Option<Report> = None;
    for rep in reps {
        let rep = rep?;
        if best.as_ref().map_or(true, |b| rep.time_ns > b.time_ns) {
            best = Some(rep);
        }
    }
    Ok(best.unwrap())
}

/// `simulate_with` for the busiest core holding workgroups off, off + cores, ... (off < cores)
fn simulate_core(code: &[u8], l: &Launch, p: &Params, tracer: Option<&Tracer>, core_off: u64) -> Result<Report, String> {
    let Geometry { dims, workgroups: n, threads } = l.geometry()?;
    let ins = cost::decode(code, p)?;
    let prepared = tracer.is_none().then(|| cost::PreparedTrace::new(code));
    let gpw = threads.div_ceil(32);
    let (cap, r) = occupancy(l);
    let mut rep = Report { workgroups: n, groups_per_wg: gpw, group_cap_per_core: cap, wgs_per_core: r, ..Default::default() };
    if gpw > cap {
        return Err(format!("workgroup of {} SIMD groups exceeds the {}-group occupancy limit", gpw, cap));
    }
    let cores = l.cores as u64;
    // all workgroups resident at once: static cluster split; otherwise cores take new workgroups as
    // they free slots, which balances the load (608 and 4096 workgroups: n / cores fits better)
    let m_static = busiest_core_wgs(n, l);
    let estimated = if m_static <= r as u64 { m_static } else { n.div_ceil(cores) };
    let m = l.busiest_core_wgs_override.unwrap_or(estimated);
    rep.busiest_core_wgs = m;
    let waves = m.div_ceil(r as u64).max(1);
    rep.waves = waves;
    // the in-simulation cold fetch (MemModel::fetch_line_ns) is off when the launch profile costs the code fetch
    // itself (`Launch::fetch_line_ns`, the runtime profile), so that the fetch is not counted twice
    let mm = MemModel { fetch_line_ns: if l.fetch_line_ns > 0.0 { 0.0 } else { fetch_line_ns() }, icache_bytes: icache_bytes(), clock_mhz: l.clock_mhz, active_cores: (cores.min(n)) as f64, warm: false, dram_gbps: l.dram_gbps, ..Default::default() };
    let device_pcs: std::collections::HashSet<usize> =
        ins.iter().filter(|i| i.name == "device_load" || i.name == "device_store").map(|i| i.pc).collect();
    let store_pcs: std::collections::HashSet<usize> = ins.iter().filter(|i| i.name == "device_store").map(|i| i.pc).collect();
    // the GPU-wide bytes the dispatch writes, from the first wave (for the store bandwidth floor)
    let written = std::sync::Mutex::new(0.0f64);
    let wg_coords = |k| wg_coords(dims, k);
    let first_wave_lines = std::sync::Mutex::new(std::collections::HashSet::<usize>::new());
    // the first wave's device segments, which later waves find in the caches (`partial_warm`)
    let first_wave_segs = std::sync::Mutex::new(None::<std::sync::Arc<std::collections::HashSet<u64>>>);
    let dram_share_cache = std::sync::Mutex::new(None::<f64>);
    let partial_warm = std::env::var("AGX_PARTIAL_WARM").map_or(l.partial_warm, |v| v == "1");
    let refetched = std::sync::atomic::AtomicBool::new(false);
    let sim_wave = |w: u64| -> Result<(u64, f64, Vec<String>, Vec<f64>), String> {
        let count = (m - w * r as u64).min(r as u64);
        let build = |first: u64| -> Result<(Vec<Trace>, Vec<u32>, Vec<String>), String> {
        let mut traces = vec![];
        let mut wg_of = vec![];
        let mut warn = vec![];
        // (workgroup, group, slot) of every SIMD group; their traces are independent (emulated in parallel)
        let jobs: Vec<(u64, u32, u32)> = (0..count).flat_map(|i| (0..gpw).map(move |g| (i, g, i as u32 * gpw + g))).collect();
        let wg_at = |i: u64| {
            // A measured peak can exceed the balanced estimate. Spread its
            // representative workgroups distinctly instead of clamping every
            // extra group onto the final grid coordinate. This is a sample of
            // workgroup paths, not reconstruction of the observed placement.
            let k = if l.busiest_core_wgs_override.is_some() {
                (((first + i) as u128 * n as u128) / m as u128) as u64
            } else {
                (first + i).saturating_mul(cores).saturating_add(core_off).min(n - 1)
            };
            wg_coords(k)
        };
        let traced: Vec<Result<Trace, String>> = match tracer {
            Some(tr) => {
                let wgs: Vec<u64> = (0..count).collect();
                let mut out = vec![];
                for r in crate::par::map(&wgs, |&i| tr(wg_at(i))) {
                    match r {
                        Ok(ts) if ts.len() == gpw as usize => out.extend(ts.into_iter().map(Ok)),
                        Ok(ts) => return Err(format!("tracer returned {} groups for a {}-group workgroup", ts.len(), gpw)),
                        Err(e) => return Err(e),
                    }
                }
                out
            }
            None => crate::par::map(&jobs, |&(i, g, slot)| {
                let ctx = GroupCtx { wg_size: l.wg, wg_id: wg_at(i), group: g, core: 0, slot, uniforms: l.uniforms.clone(), launch: cost::SrLaunch { tg_bytes: l.tg_bytes, regs: l.regs, ..Default::default() } };
                prepared.as_ref().unwrap().trace_group(&ctx, l.max_steps)
            }),
        };
        for (&(i, _, _), t) in jobs.iter().zip(traced) {
            let t = t?;
            for x in &t.warnings {
                if !warn.contains(x) {
                    warn.push(x.clone());
                }
            }
            traces.push(t);
            wg_of.push(i as u32);
        }
        Ok((traces, wg_of, warn))
        };
        let (mut traces, wg_of, mut warn) = build(w * r as u64)?;
        let next_wave = |traces: &mut Vec<Trace>, warn: &mut Vec<String>| -> Result<u64, String> {
        // `chain_waves`: each slot runs this wave's workgroup and then the next wave's (the GPU starts a workgroup in
        // a slot as soon as the previous one ends, so one wave's loads run under the other's tail), simulated
        // together, half the cycles each (AGX_CHAIN_WAVES=0 / 1 overrides)
        // AGX_CHAIN_DEPTH=k (experiment, default 2): k waves per slot, 1/k of the cycles each
        let depth = std::env::var("AGX_CHAIN_DEPTH").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(2).max(2);
        let mut chained = 1u64;
        if std::env::var("AGX_CHAIN_WAVES").map_or(l.chain_waves, |v| v == "1") {
            while chained < depth && w + chained < waves && m - (w + chained) * r as u64 >= count {
                let (next, _, warn2) = build((w + chained) * r as u64)?;
                warn.extend(warn2);
                for (t, t2) in traces.iter_mut().zip(next) {
                    let off = t.pcs.len() as u32;
                    t.pcs.extend(t2.pcs);
                    t.mem.extend(t2.mem.into_iter().map(|(s, v)| (s + off, v)));
                    t.device_segments.extend(t2.device_segments.into_iter().map(|(s, v)| (s + off, v)));
                    t.masked.extend(t2.masked.into_iter().map(|s| s + off));
                }
                chained += 1;
            }
        }
            Ok(chained)
        };
        if w == 0 {
            let mut lines = first_wave_lines.lock().unwrap();
            lines.extend(traces.iter().flat_map(|t| t.pcs.iter().map(|&pc| pc / 64)));
        }
        let seqs: Vec<&[usize]> = traces.iter().map(|t| t.pcs.as_slice()).collect();
        let mems: Vec<&[(u32, Vec<u64>)]> = traces.iter().map(|t| t.mem.as_slice()).collect();
        let masked: Vec<&[u32]> = traces.iter().map(|t| t.masked.as_slice()).collect();
        // GPU-wide device footprint: this wave's distinct segments, scaled to the busiest core's workgroups and
        // to every active core; if it fits the system-level cache, the data is taken to be resident there
        let mut distinct: std::collections::HashSet<u64> = Default::default();
        for t in &traces {
            for (step, segs) in &t.mem {
                if device_pcs.contains(&t.pcs[*step as usize]) {
                    distinct.extend(segs.iter().copied());
                }
            }
        }
        let scale = (m as f64 / count.max(1) as f64) * cores.min(n) as f64;
        let footprint = distinct.len() as f64 * 64.0 * scale;
        let mut region_lines: Vec<u64> = vec![];
        if !l.regions.is_empty() {
            // a buffer's lines cannot exceed the buffer: workgroups whose tiles overlap (halos, a downsampling
            // kernel's 2x2 reads, Halide's local_laplacian gPyramid2: 70 MiB estimated for 15.6 MiB of buffers) count
            // shared lines once per workgroup otherwise. Lines outside every region are scaled as before.
            let mut per = vec![0u64; l.regions.len() + 1];
            for &sg in &distinct {
                let a = sg << 6;
                let k = l.regions.iter().position(|&(b, len)| a >= b && a < b + len).unwrap_or(l.regions.len());
                per[k] += 1;
            }
            region_lines = per;
        }
        if w == 0 && l.store_gbps > 0.0 {
            let mut stored: std::collections::HashSet<u64> = Default::default();
            for t in &traces {
                for (step, segs) in &t.mem {
                    if store_pcs.contains(&t.pcs[*step as usize]) {
                        stored.extend(segs.iter().copied());
                    }
                }
            }
            *written.lock().unwrap() = stored.len() as f64 * 64.0 * (m as f64 / count.max(1) as f64) * cores.min(n) as f64;
        }
        // data shared between cores: mean overlap of this core's lines with workgroups on other cores
        let mut sharing = 0.0;
        if l.mem && n > 1 {
            // (workgroups of this wave on other cores: the wave's first one on this core is (w r) cores + core_off.
            // Comparing a later wave with the grid's first workgroups found no sharing and doubled the footprint
            // estimate of Halide's stencil_chain stages from wave 1 on: 29.5 MiB against 18 MiB of buffers)
            let k0 = (w * r as u64).saturating_mul(cores).saturating_add(core_off);
            let others: Vec<u64> = [1u64, 2, cores / 2, cores - 1].iter().map(|&d| k0 + d).filter(|&k| k < n && k > 0).collect();
            // the other workgroups are traced and compared independently: concurrently, summed in the same order
            let ovs: Vec<f64> = match tracer {
                Some(t) => crate::par::map(&others, |&k| t(wg_coords(k)).map(|tr| cost::segment_overlap(&traces, &tr, &device_pcs)))
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?,
                None => {
                    let mut ovs = vec![];
                    for &k in &others {
                        let gs: Vec<u32> = (0..gpw).collect();
                        let tr = crate::par::map(&gs, |&g| {
                            let ctx = GroupCtx { wg_size: l.wg, wg_id: wg_coords(k), group: g, core: 1, slot: g, uniforms: l.uniforms.clone(), launch: cost::SrLaunch { tg_bytes: l.tg_bytes, regs: l.regs, ..Default::default() } };
                            prepared.as_ref().unwrap().trace_group(&ctx, l.max_steps)
                        }).into_iter().collect::<Result<Vec<_>, _>>()?;
                        ovs.push(cost::segment_overlap(&traces, &tr, &device_pcs));
                    }
                    ovs
                }
            };
            if !ovs.is_empty() {
                sharing = ovs.iter().sum::<f64>() / ovs.len() as f64;
            }
        }
        // DRAM traffic shared with neighbouring workgroups (usually on other cores): for a mid-grid workgroup, each
        // of its lines costs 1 / (1 + the neighbours (x +-1, y +-1) that also read it). Colsum probe with overlapping
        // columns: hardware 0.52x the private time, bandwidth-bound on distinct data (playground results/
        // 2026-11-01-gpu-written-lines). AGX_DRAM_SHARE=0: every core pays all its lines
        let dram_share = {
            let mut cached = dram_share_cache.lock().unwrap();
            if cached.is_none() {
                let mut share = 1.0;
                if l.mem && n > 1 && std::env::var("AGX_DRAM_SHARE").map_or(true, |v| v != "0") {
                    let k0 = n / 2;
                    // A contiguous run of `block` workgroups from mid-grid (they run at about the same time GPU-wide): a
                    // line that several of them read is fetched once, so the share is their distinct lines over the sum
                    // of each one's. The neighbours-only estimate (x +-1, y +-1; AGX_DRAM_SHARE_BLOCK=0) missed lines
                    // shared further away (Halide's 2x2 downsampling with halos, pyramid levels in y or z): local_laplacian
                    // gPyramid2 +33% -> +6%, f5 +54 -> +15%, 41 of 162 held-out launches better, 11 worse
                    // (results/2026-11-07-held-out)
                    let block: u64 = std::env::var("AGX_DRAM_SHARE_BLOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
                    let mut ks = vec![k0];
                    if block > 1 {
                        let start = (n / 2).saturating_sub(block / 2);
                        ks = (start..(start + block).min(n)).collect();
                    } else {
                    for d in [1u64, dims[0]] {
                        if k0 >= d { ks.push(k0 - d); }
                        if k0 + d < n { ks.push(k0 + d); }
                    }
                    ks.dedup();
                    }
                    let trace_wg = |k: u64| -> Result<Vec<Trace>, String> {
                        match tracer {
                            Some(t) => t(wg_coords(k)),
                            None => {
                                let gs: Vec<u32> = (0..gpw).collect();
                                crate::par::map(&gs, |&g| {
                                    let ctx = GroupCtx { wg_size: l.wg, wg_id: wg_coords(k), group: g, core: 1, slot: g, uniforms: l.uniforms.clone(), launch: cost::SrLaunch { tg_bytes: l.tg_bytes, regs: l.regs, ..Default::default() } };
                                    prepared.as_ref().unwrap().trace_group(&ctx, l.max_steps)
                                }).into_iter().collect()
                            }
                        }
                    };
                    let sets: Vec<std::collections::HashSet<u64>> = crate::par::map(&ks, |&k| trace_wg(k))
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()?
                        .iter()
                        .map(|ts| {
                            let mut set = std::collections::HashSet::new();
                            for t in ts {
                                for (step, segs) in &t.mem {
                                    if device_pcs.contains(&t.pcs[*step as usize]) {
                                        set.extend(segs.iter().copied());
                                    }
                                }
                            }
                            set
                        })
                        .collect();
                    if block > 1 {
                        let sum: usize = sets.iter().map(|x| x.len()).sum();
                        let union: std::collections::HashSet<u64> = sets.iter().flatten().copied().collect();
                        if sum > 0 { share = union.len() as f64 / sum as f64; }
                    } else if !sets[0].is_empty() && sets.len() > 1 {
                        share = sets[0].iter().map(|sg| 1.0 / (1.0 + sets[1..].iter().filter(|o| o.contains(sg)).count() as f64)).sum::<f64>()
                            / sets[0].len() as f64;
                    }
                }
                *cached = Some(share);
            }
            cached.unwrap()
        };
        // with sharing, the GPU-wide footprint is smaller than this core's x active cores
        let shared_by = 1.0 + (cores.min(n) as f64 - 1.0) * sharing;
        let footprint = if region_lines.is_empty() {
            footprint / shared_by
        } else {
            region_lines.iter().enumerate().map(|(k, &c)| {
                let est = c as f64 * 64.0 * scale / shared_by;
                if k < l.regions.len() { est.min(l.regions[k].1 as f64) } else { est }
            }).sum()
        };
        if std::env::var("AGX_DEBUG_FOOTPRINT").is_ok_and(|v| v == "1") {
            eprintln!("wave {w}: footprint {:.1} MiB (uncapped {:.1}, slc {:.1} MiB) sharing {sharing:.2} dram_share {dram_share:.2}", footprint / 1048576.0, distinct.len() as f64 * 64.0 * scale / shared_by / 1048576.0, mm.slc_bytes as f64 / 1048576.0);
        }
        // The SLC holds a footprint up to ~14 MiB whole and none of one beyond ~30 MiB, smoothly in between (half-way
        // ~21 MiB, playground novel-results/2026-10-18-slc): the fraction of lines still there falls linearly over
        // 14-30 MiB. The step at slc_bytes (AGX_SLC_SMOOTH=0) sent waves of Halide's conv_layer whose footprint
        // estimate straddled 19 MiB (9.8 / 19.6) to DRAM every other wave: 232k / 491k cycles.
        let (resident, frac) = if std::env::var("AGX_SLC_SMOOTH").map_or(true, |v| v != "0") {
            let mib = footprint / 1048576.0;
            let h = ((30.0 - mib) / 16.0).clamp(0.0, 1.0);
            (h > 0.0, h)
        } else {
            (footprint <= mm.slc_bytes as f64, 1.0)
        };
        // Of a wave's first touches (`first_touch_dram`) only the lines new to the whole dispatch go at the DRAM rate
        // and wait the first-touch latency: the fraction is the estimated footprint (capped by the buffers) over the
        // sum of every core's waves' lines, over the DRAM share (which already divides the rate). The others were read
        // by other cores or earlier waves and wait L2 latency (the GPU-wide L2). Held-out set (159 launches without
        // Cycles): median 9 -> 7%, within 10% 91 -> 111; Halide bilateral_grid +53 -> +27%, local_laplacian's
        // pyramid levels +20-30% -> +-10%. AGX_FIRST_TOUCH_FRAC=0: every first touch is new (round 3); 1: the fraction
        // without the L2 latency; 2: 1 with every first touch at the first-touch latency
        let raw = distinct.len() as f64 * 64.0 * scale;
        let first_touch_frac = if std::env::var("AGX_FIRST_TOUCH_FRAC").map_or(true, |v| v != "0") && raw > 0.0 {
            (footprint / (raw * dram_share.clamp(0.05, 1.0))).clamp(0.0, 1.0)
        } else {
            1.0
        };
        if std::env::var("AGX_DEBUG_FOOTPRINT").is_ok_and(|v| v == "1") {
            eprintln!("wave {w}: first_touch_frac {first_touch_frac:.3}");
        }
        let mm = MemModel { resident_slc: resident, slc_resident_frac: frac, sharing, dram_share, first_touch_frac, ..mm.clone() };
        // `later_waves_warm`: the waves after the first find their data in the caches when it fits the system-level
        // cache (what the first waves and their neighbours loaded); AGX_LATER_WAVES_WARM=1 / 0 forces it
        let warm_later = std::env::var("AGX_LATER_WAVES_WARM").map_or(l.later_waves_warm, |v| v == "1");
        if w == 0 && partial_warm {
            *first_wave_segs.lock().unwrap() = Some(std::sync::Arc::new(distinct.clone()));
        }
        let mm = MemModel { first_touch_dram: partial_warm, ..mm };
        let mm = if w > 0 && mm.resident_slc && warm_later {
            let warm_only = if partial_warm { first_wave_segs.lock().unwrap().clone() } else { None };
            MemModel { warm: true, warm_only, first_touch_dram: partial_warm, ..mm }
        } else {
            mm
        };
        // code beyond the instruction cache: fetched through it in the simulation (the launch's per-line fetch is then
        // not charged, `refetched`)
        let icache_lines = std::env::var("AGX_ICACHE_LINES").ok().and_then(|v| v.parse().ok()).unwrap_or(l.icache_lines);
        let mm = if icache_lines > 0 && l.fetch_line_ns > 0.0 && {
            let lines: std::collections::HashSet<usize> = traces.iter().flat_map(|t| t.pcs.iter().map(|&pc| pc / 64)).collect();
            lines.len() > icache_lines
        } {
            refetched.store(true, std::sync::atomic::Ordering::Relaxed);
            // (a later wave's core has just run the same code: AGX_ICACHE_WARM=0 starts every wave with an empty cache)
            let icache_warm = w > 0 && std::env::var("AGX_ICACHE_WARM").map_or(true, |v| v != "0");
            MemModel { icache_lines, icache_victims: l.icache_victims, icache_warm, fetch_line_ns: l.refetch_line_ns, fetch_jump_ns: l.fetch_jump_ns, icache_bytes: 0, ..mm }
        } else {
            mm
        };
        // (chained after the footprint, the written bytes and the sharing, which are this wave's)
        drop((seqs, mems, masked));
        let chained = next_wave(&mut traces, &mut warn)?;
        let seqs: Vec<&[usize]> = traces.iter().map(|t| t.pcs.as_slice()).collect();
        let mems: Vec<&[(u32, Vec<u64>)]> = traces.iter().map(|t| t.mem.as_slice()).collect();
        let masked: Vec<&[u32]> = traces.iter().map(|t| t.masked.as_slice()).collect();
        let mem = if l.mem { Some((mems.as_slice(), &mm)) } else { None };
        let (c, _, finish) = cost::simulate_groups_finish(&ins, &seqs, Some(&masked), mem, p, false, Some(&wg_of))?;
        // chained waves share a slot: each takes 1/chained of the time, as its workgroups do of their finish times
        let per_wg = workgroup_finish(&finish, &wg_of, count as usize).into_iter().map(|f| f / chained as f64).collect();
        Ok((count, c / chained as f64, warn, per_wg))
    };
    // the first and last waves, the middle, then densely after the first (2^k - 1): data-dependent counts put the
    // work at the start of the grid
    let mut todo: Vec<u64> = vec![0];
    let dense = (1..63).map(|k| (1u64 << k) - 1).take_while(|&w| w < waves);
    for w in [waves - 1, waves / 2].into_iter().chain(dense) {
        if !todo.contains(&w) {
            todo.push(w);
        }
    }
    todo.truncate(l.max_waves.max(1));
    let mut total = 0.0;
    // the simulated waves are independent: run them concurrently (results combined in wave order)
    // the first wave first when later waves need its segments, then the rest concurrently
    let results: Vec<_> = if partial_warm {
        let r0 = sim_wave(todo[0]);
        std::iter::once(r0).chain(crate::par::map(&todo[1..], |&w| sim_wave(w))).collect()
    } else {
        crate::par::map(&todo, |&w| sim_wave(w))
    };
    let mut wave_wgs: Vec<(u64, Vec<f64>)> = vec![];
    for (&w, res) in todo.iter().zip(results) {
        let (count, c, warn, per_wg) = res?;
        wave_wgs.push((w, per_wg));
        for x in warn {
            if !rep.warnings.contains(&x) {
                rep.warnings.push(x);
            }
        }
        rep.simulated_waves.push((w, count, c));
        total += c;
    }
    // a wave not simulated: linear between the nearest simulated full waves (the last one is partial when m is not
    // a multiple of r), else as the nearest one
    let mut full: Vec<(u64, f64)> =
        rep.simulated_waves.iter().filter(|&&(w, count, _)| count == r as u64 || w + 1 < waves).map(|&(w, _, c)| (w, c)).collect();
    full.sort_by_key(|&(w, _)| w);
    // between two simulated full waves s0 < s1: the k = s1 - s0 - 1 waves in between sum to
    // k c0 + (c1 - c0) (1 + ... + k) / (s1 - s0); after the last simulated full wave, as that one
    for pair in full.windows(2) {
        let ((s0, c0), (s1, c1)) = (pair[0], pair[1]);
        let k = (s1 - s0 - 1) as f64;
        total += k * c0 + (c1 - c0) * (k * (k + 1.0) / 2.0) / (s1 - s0) as f64;
    }
    if let Some(&(s_last, c_last)) = full.last() {
        let after = waves - s_last - 1 - todo.iter().filter(|&&w| w > s_last).count() as u64;
        total += c_last * after as f64;
    }
    if refill() && r > 1 {
        // Each wave not simulated: as the nearest simulated one, whose total the interpolation above already stands for.
        wave_wgs.sort_by_key(|x| x.0);
        let mut durations = Vec::with_capacity(m as usize);
        for w in 0..waves {
            let (_, near) = wave_wgs.iter().min_by_key(|(s, _)| s.abs_diff(w)).unwrap();
            let count = (m - w * r as u64).min(r as u64) as usize;
            durations.extend((0..count).map(|i| near[i % near.len()]));
        }
        total = total.min(refill_makespan(&durations, r as usize));
    }
    rep.core_cycles = total;
    rep.launch_core_ns = (m as f64 * l.wg_launch_ns).max((m as u128 * gpw as u128) as f64 * l.group_launch_ns);
    rep.code_lines = first_wave_lines.into_inner().unwrap().len() as u64;
    // the fetch overlaps execution (the slower of the two sets the core's time): straight-line code of one group is
    // fetch-bound, a long kernel's loops hide it (llama.cpp's Q8_0 matrix-vector kernel: 138 lines, 22.8 us of work,
    // 26.2 us measured with the 2.3 us launch)
    let fetch_ns = if refetched.load(std::sync::atomic::Ordering::Relaxed) { 0.0 } else { rep.code_lines as f64 * l.fetch_line_ns };
    let stack_ns = if l.stack_bytes > 0 { l.stack_launch_ns } else { 0.0 };
    let written = written.into_inner().unwrap();
    let writes_ns = if !store_pcs.is_empty() { l.store_launch_ns } else { 0.0 };
    let store_ns = if l.store_gbps > 0.0 { written / l.store_gbps } else { 0.0 };
    // (the stores after the workgroups' launch: 1520 x 256 threads writing 1.5 / 6 / 24 MB, 4 us of launch + the bytes
    // at 365 GB/s + 2.3 us: 10.4 / 22.7 / 72.0 us, measured 9.9 / 22.0 / 72.1)
    // (launch and stores overlap beyond that: Geekbench's gradient, 9,672 workgroups per core, 967 us of launch and
    // ~500 us of stores, measured 1036 us)
    let store_floor = if store_ns > 0.0 { store_ns + rep.launch_core_ns.min(store_ns) } else { 0.0 };
    // ... but only partly: time = max + 0.3 min of the two (one thread, 32 dependent loads to new lines with 0-120 FMAs
    // of straight-line code after each: 0.27-0.34; playground novel-results/2026-11-01-fetch-overlap). The full overlap
    // fitted the Q8_0 kernel only while threadgroup loads were wrongly deferred. AGX_FETCH_OVERLAP overrides 0.3
    let mut core_ns = total / l.clock_mhz * 1000.0;
    if l.stack_bytes > 0 {
        let slot_ns = std::env::var("AGX_STACK_SLOT_NS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(l.stack_slot_ns);
        if slot_ns > 0.0 {
            // the slots are set up one after the other before their groups run (the first one with the command's
            // stack set-up), and the waves run after them
            let slots = m.min(r as u64) as f64 * gpw as f64;
            // (GPU work: measured at the top clock, 1398 MHz, and slower at a lower one; in-run gaussian Fan1, mostly
            // slot set-up, takes 2.8x its warm time like the compute-bound launches)
            core_ns += (slots - 1.0).max(0.0) * slot_ns * 1398.0 / l.clock_mhz;
        }
    }
    let fetch_part = std::env::var("AGX_FETCH_OVERLAP").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.3) * core_ns.min(fetch_ns);
    rep.time_ns = (core_ns.max(fetch_ns) + fetch_part).max(rep.launch_core_ns).max(store_floor) + l.launch_ns + stack_ns + writes_ns;
    if !rep.time_ns.is_finite() { return Err("predicted time is non-finite for this launch".into()); }
    Ok(rep)
}

/// Workgroups on the busiest core: even split over clusters, then over each cluster's cores.
pub fn busiest_core_wgs(n: u64, l: &Launch) -> u64 {
    if l.cores == 0 || l.clusters.contains(&0) { return 0; }
    let k = l.clusters.len() as u64;
    if k == 0 || l.clusters.iter().map(|&c| c as u64).sum::<u64>() != l.cores as u64 {
        return n.div_ceil(l.cores as u64);
    }
    let share = n.div_ceil(k);
    let smallest = *l.clusters.iter().min().unwrap() as u64;
    share.div_ceil(smallest)
}

#[cfg(test)]
mod tests {
    #[test]
    fn refill_fills_the_gap_a_slow_workgroup_leaves() {
        // Two slots: waves take 10 + 10 = 20; with refill the short ones run beside the first long one and the
        // second long one starts at 2.
        assert_eq!(super::refill_makespan(&[10.0, 1.0, 1.0, 10.0], 2), 12.0);
        // One slot: the sum, as waves.
        assert_eq!(super::refill_makespan(&[3.0, 4.0], 1), 7.0);
        // Never less than the longest.
        assert_eq!(super::refill_makespan(&[5.0, 1.0], 8), 5.0);
    }

    #[test]
    fn observed_peak_controls_partial_wave_and_launch_work() {
        let p = Params::new().unwrap();
        let l = Launch { grid: [131072, 1, 1], wg: [128, 1, 1], mem: false, ..Default::default() };
        let base = simulate(&[0x88, 0], &l, &p).unwrap();
        let measured = simulate(&[0x88, 0], &Launch { busiest_core_wgs_override: Some(33), ..l.clone() }, &p).unwrap();
        assert_eq!((base.workgroups, base.busiest_core_wgs, base.wgs_per_core), (1024, 27, 13));
        assert_eq!((measured.busiest_core_wgs, measured.waves), (33, 3));
        // the first, the last (partial) and the middle wave
        assert_eq!(measured.simulated_waves.iter().map(|&(wave, count, _)| (wave, count)).collect::<Vec<_>>(), vec![(0, 13), (2, 7), (1, 13)]);
        assert_eq!(measured.launch_core_ns, 3300.0);
        assert!(measured.time_ns >= base.time_ns);
        let same = simulate(&[0x88, 0], &Launch { busiest_core_wgs_override: Some(27), ..l }, &p).unwrap();
        assert_eq!((same.time_ns, same.core_cycles, same.waves), (base.time_ns, base.core_cycles, base.waves));
    }

    #[test]
    fn observed_peak_rejects_impossible_counts_before_tracing() {
        let l = Launch { grid: [131072, 1, 1], wg: [128, 1, 1], ..Default::default() };
        for count in [0, 26, 1025] {
            let invalid = Launch { busiest_core_wgs_override: Some(count), ..l.clone() };
            let error = simulate(&[], &invalid, &Params::new().unwrap()).unwrap_err();
            assert!(error.contains("busiest_core_wgs"));
        }
        for count in [27, 33, 1024] {
            Launch { busiest_core_wgs_override: Some(count), ..l.clone() }.validate().unwrap();
        }
    }

    #[test]
    fn observed_peak_samples_distinct_workgroups_across_waves() {
        let l = Launch { grid: [131072, 1, 1], wg: [128, 1, 1], mem: false,
            busiest_core_wgs_override: Some(33), ..Default::default() };
        let seen = std::sync::Mutex::new(vec![]);
        let tracer = |wg| {
            seen.lock().unwrap().push(wg);
            Ok(vec![Trace { pcs: vec![0], ..Default::default() }; 4])
        };
        let r = simulate_with(&[0x88, 0], &l, &Params::new().unwrap(), Some(&tracer)).unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(r.busiest_core_wgs, 33);
        // The model samples the first, last and middle waves: 13 + 7 + 13 workgroups, all distinct.
        assert_eq!(seen.len(), 33);
        assert_eq!(seen.iter().collect::<std::collections::HashSet<_>>().len(), 33);
        assert!(seen.iter().all(|w| w[0] < 1024 && w[1] == 0 && w[2] == 0));
    }
    use super::*;

    #[test]
    fn invalid_launches_fail_before_decode_or_external_trace() {
        let mut cases = vec![];
        for wg in [[0, 1, 1], [1, 0, 1], [1, 1, 0], [1025, 1, 1], [65536, 65536, 1]] {
            cases.push((Launch { wg, ..Default::default() }, "workgroup"));
        }
        cases.push((Launch { grid: [0, 1, 1], ..Default::default() }, "grid"));
        cases.push((Launch { grid: [u32::MAX; 3], wg: [1; 3], ..Default::default() }, "workgroup count"));
        cases.push((Launch { cores: 0, ..Default::default() }, "core"));
        cases.push((Launch { clusters: vec![0, 38], ..Default::default() }, "cluster"));
        cases.push((Launch { tg_bytes: 65537, ..Default::default() }, "threadgroup memory"));
        cases.push((Launch { regs: u32::MAX, ..Default::default() }, "occupancy"));
        cases.push((Launch { uniforms: vec![(usize::MAX, 0)], ..Default::default() }, "uniform"));
        for v in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            cases.push((Launch { clock_mhz: v, ..Default::default() }, "clock_mhz"));
            cases.push((Launch { dram_gbps: v, ..Default::default() }, "dram_gbps"));
        }
        for v in [-1.0, f64::NAN, f64::INFINITY] {
            cases.push((Launch { launch_ns: v, ..Default::default() }, "launch_ns"));
            cases.push((Launch { wg_launch_ns: v, ..Default::default() }, "wg_launch_ns"));
            cases.push((Launch { group_launch_ns: v, ..Default::default() }, "group_launch_ns"));
        }
        let p = Params::new().unwrap();
        let no_trace = |_: [u32; 3]| -> Result<Vec<Trace>, String> { panic!("invalid launch called external tracer") };
        for (l, message) in cases {
            let expected = l.validate().unwrap_err();
            assert!(expected.contains(message), "{}", expected);
            assert_eq!(simulate_with(&[0xff], &l, &p, Some(&no_trace)).unwrap_err(), expected);
        }
    }

    #[test]
    fn occupancy_keeps_retained_rules_and_never_invents_a_resident_workgroup() {
        // Playground occupancy finding: G14C, register-pair allocation and 64 KiB shared memory.
        for (regs, cap) in [(16, 96), (32, 96), (40, 82), (48, 68), (56, 58), (64, 52), (72, 46),
                            (80, 40), (96, 34), (128, 26), (160, 20), (192, 16), (248, 12)] {
            assert_eq!(occupancy(&Launch { regs, wg: [128, 1, 1], ..Default::default() }), (cap, cap / 4));
        }
        for (tg_bytes, resident) in [(1024, 24), (4096, 16), (12288, 5), (24576, 2), (32768, 2), (49152, 1), (65536, 1)] {
            assert_eq!(occupancy(&Launch { regs: 16, wg: [128, 1, 1], tg_bytes, ..Default::default() }), (96, resident));
        }
        for wg in [[0; 3], [u32::MAX; 3], [1025, 1, 1]] {
            assert_eq!(occupancy(&Launch { wg, ..Default::default() }), (0, 0));
        }
        assert_eq!(occupancy(&Launch { regs: u32::MAX, ..Default::default() }), (0, 0));
        assert_eq!(occupancy(&Launch { tg_bytes: 65537, ..Default::default() }).1, 0);
        assert_eq!(busiest_core_wgs(1, &Launch { cores: 0, ..Default::default() }), 0);
        assert_eq!(busiest_core_wgs(1, &Launch { clusters: vec![0, 38], ..Default::default() }), 0);
        // Keep existing custom-core fallback, zero-overhead settings and minimum-register normalization.
        let l = Launch { cores: 2, regs: 0, launch_ns: 0.0, wg_launch_ns: 0.0, group_launch_ns: 0.0, ..Default::default() };
        l.validate().unwrap();
        assert_eq!(busiest_core_wgs(5, &l), 3);
    }

    #[test]
    fn co_residency_uses_the_b5_progress_bound_without_changing_wave_occupancy() {
        let one_group = Launch { regs: 16, wg: [32, 1, 1], ..Default::default() };
        assert_eq!(occupancy(&one_group).1, 55);
        assert_eq!(co_resident_workgroups(&one_group), 15);
        let four_groups = Launch { regs: 64, wg: [128, 1, 1], ..Default::default() };
        assert_eq!(occupancy(&four_groups).1, 13);
        assert_eq!(co_resident_workgroups(&four_groups), 12);
        let shared = Launch { tg_bytes: 32768, ..one_group };
        assert_eq!(co_resident_workgroups(&shared), 2);
        let rounded_registers = Launch { regs: 33, wg: [512, 1, 1], ..Default::default() };
        assert_eq!(co_resident_workgroups(&rounded_registers), 4);
    }

    #[test]
    fn large_coordinates_do_not_truncate_to_u32_or_overflow_xy() {
        let dims = [65536, 65536, 2];
        assert_eq!(wg_coords(dims, (1u64 << 32) + 65536 + 7), [7, 1, 1]);
        for k in [0, 65535, 65536, (1u64 << 32) - 1, 1u64 << 32, (1u64 << 33) - 1] {
            let [x, y, z] = wg_coords(dims, k).map(u64::from);
            assert_eq!(x + dims[0] * (y + dims[1] * z), k);
        }
        let l = Launch { grid: [65536, 65536, 2], wg: [1; 3], cores: 1, clusters: vec![], mem: false, ..Default::default() };
        let seen = std::sync::Mutex::new(vec![]);
        let tracer = |wg| { seen.lock().unwrap().push(wg); Ok(vec![Trace { pcs: vec![0], ..Default::default() }]) };
        let report = simulate_with(&[0x88, 0], &l, &Params::new().unwrap(), Some(&tracer)).unwrap();
        assert_eq!(report.workgroups, 1u64 << 33);
        assert_eq!(report.simulated_waves.len(), 8);
        assert!(seen.lock().unwrap().iter().any(|w| w[2] == 1));
        assert!(report.time_ns.is_finite()); // synthetic coordinate check, not hardware timing validation
    }

    #[test]
    fn finite_inputs_cannot_silently_serialize_an_infinite_time() {
        let l = Launch { clock_mhz: f64::MIN_POSITIVE, mem: false, ..Default::default() };
        assert!(simulate(&[0x88, 0], &l, &Params::new().unwrap()).unwrap_err().contains("non-finite"));
    }
}

