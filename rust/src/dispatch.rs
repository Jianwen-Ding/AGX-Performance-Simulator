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
    /// cores per cluster (workgroups are split evenly over clusters first); empty = round-robin over `cores`
    pub clusters: Vec<u32>,
    pub uniforms: Vec<(usize, u32)>,
    pub mem: bool,
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
            clusters: vec![10, 9, 9, 10],
            uniforms: vec![],
            mem: true,
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
/// fetched again each launch. AGX_FETCH_LINE_NS overrides it (0: off).
pub fn fetch_line_ns() -> f64 {
    std::env::var("AGX_FETCH_LINE_NS").ok().and_then(|v| v.parse().ok()).unwrap_or(62.0)
}

pub fn simulate(code: &[u8], l: &Launch, p: &Params) -> Result<Report, String> {
    simulate_with(code, l, p, None)
}

/// `simulate` with the SIMD groups' paths from `tracer` (None: `cost::trace_group`).
pub fn simulate_with(code: &[u8], l: &Launch, p: &Params, tracer: Option<&Tracer>) -> Result<Report, String> {
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
    let mm = MemModel { fetch_line_ns: fetch_line_ns(), clock_mhz: l.clock_mhz, active_cores: (cores.min(n)) as f64, warm: false, dram_gbps: l.dram_gbps, ..Default::default() };
    let device_pcs: std::collections::HashSet<usize> =
        ins.iter().filter(|i| i.name == "device_load" || i.name == "device_store").map(|i| i.pc).collect();
    let wg_coords = |k| wg_coords(dims, k);
    let sim_wave = |w: u64| -> Result<(u64, f64, Vec<String>), String> {
        let first = w * r as u64;
        let count = (m - first).min(r as u64);
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
                (first + i).saturating_mul(cores).min(n - 1)
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
        let footprint = distinct.len() as f64 * 64.0 * (m as f64 / count.max(1) as f64) * cores.min(n) as f64;
        // data shared between cores: mean overlap of this core's lines with workgroups on other cores
        let mut sharing = 0.0;
        if l.mem && n > 1 {
            let others: Vec<u64> = [1u64, 2, cores / 2, cores - 1].iter().copied().filter(|&k| k < n && k > 0).collect();
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
        // with sharing, the GPU-wide footprint is smaller than this core's x active cores
        let footprint = footprint / (1.0 + (cores.min(n) as f64 - 1.0) * sharing);
        let mm = MemModel { resident_slc: footprint <= mm.slc_bytes as f64, sharing, ..mm.clone() };
        let mem = if l.mem { Some((mems.as_slice(), &mm)) } else { None };
        let c = cost::simulate_groups(&ins, &seqs, Some(&masked), mem, p, false, Some(&wg_of))?.0;
        Ok((count, c, warn))
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
    let results = crate::par::map(&todo, |&w| sim_wave(w));
    for (&w, res) in todo.iter().zip(results) {
        let (count, c, warn) = res?;
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
    rep.core_cycles = total;
    rep.launch_core_ns = (m as f64 * l.wg_launch_ns).max((m as u128 * gpw as u128) as f64 * l.group_launch_ns);
    rep.time_ns = (total / l.clock_mhz * 1000.0).max(rep.launch_core_ns) + l.launch_ns;
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

