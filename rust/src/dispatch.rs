//! Dispatch-level model: one compute dispatch on the whole GPU.
//!
//! * Occupancy (playground novel-results/2026-10-11-occupancy): resident SIMD groups per core =
//!   min(96, 2 * floor(3328 / (2 * register halves))); 32-lane workgroups cap near 55 groups and 64-lane
//!   ones near 84; whole workgroups only. Threadgroup memory per core is not measured: `tg_capacity`
//!   (default 64 KiB) is an assumption.
//! * Workgroups are split evenly over the GPU clusters (38 cores: 10, 9, 9, 10) and then over the cores of
//!   each cluster, so the busiest core gets ceil(ceil(n / 4) / 9) workgroups (wg_placement.py: exact for
//!   38..608 workgroups of 64 and 256 lanes). That core sets the time.
//! * That core runs its workgroups in waves of `wgs_per_core`; each simulated wave traces every SIMD group
//!   with its own geometry (`cost::trace_group`) and runs `cost::simulate_groups` with the memory model.
//!   With many waves only the first and the last are simulated and the rest extrapolated from the first.
//! * Launching costs the busiest core max(`wg_launch_ns` per workgroup, `group_launch_ns` per SIMD group)
//!   (empty kernels over 4M-16M threads); it overlaps execution, so the core time is the larger of the two.
//! * time = core time + `launch_ns` (fixed per-dispatch overhead, 5.6 us for a 32-thread empty dispatch).

use crate::cost::{self, GroupCtx, MemModel, Params};

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
    pub clock_mhz: f64,
    pub launch_ns: f64,
    pub wg_launch_ns: f64,
    pub group_launch_ns: f64,
    /// cores per cluster (workgroups are split evenly over clusters first); empty = round-robin over `cores`
    pub clusters: Vec<u32>,
    pub uniforms: Vec<(usize, u32)>,
    pub mem: bool,
    /// simulate at most this many distinct waves (first, last); others are extrapolated
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
            clock_mhz: 1398.0,
            launch_ns: 5600.0,
            wg_launch_ns: 100.0,
            group_launch_ns: 13.6,
            clusters: vec![10, 9, 9, 10],
            uniforms: vec![],
            mem: true,
            max_waves: 2,
            max_steps: 20_000_000,
        }
    }
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

/// (resident SIMD groups per core allowed by registers and workgroup size, resident workgroups per core)
pub fn occupancy(l: &Launch) -> (u32, u32) {
    let halves = l.regs.max(8);
    let mut cap = 96u32.min(2 * (3328 / (2 * halves)));
    let threads = l.wg[0] * l.wg[1] * l.wg[2];
    if threads <= 32 {
        cap = cap.min(55);
    } else if threads <= 64 {
        cap = cap.min(84);
    }
    let gpw = threads.div_ceil(32);
    let mut wgs = (cap / gpw).max(1);
    if l.tg_bytes > 0 {
        wgs = wgs.min((l.tg_capacity / l.tg_bytes).max(1));
    }
    (cap, wgs)
}

pub fn simulate(code: &[u8], l: &Launch, p: &Params) -> Result<Report, String> {
    let ins = cost::decode(code, p)?;
    let dims: Vec<u32> = (0..3).map(|i| l.grid[i].div_ceil(l.wg[i].max(1))).collect();
    let n = dims.iter().map(|&d| d as u64).product::<u64>();
    let threads = l.wg[0] * l.wg[1] * l.wg[2];
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
    let m = if m_static <= r as u64 { m_static } else { n.div_ceil(cores) };
    rep.busiest_core_wgs = m;
    let waves = m.div_ceil(r as u64).max(1);
    rep.waves = waves;
    let mm = MemModel { clock_mhz: l.clock_mhz, active_cores: (cores.min(n)) as f64, warm: false, ..Default::default() };
    let wg_coords = |k: u64| -> [u32; 3] {
        let k = k as u32;
        [k % dims[0], (k / dims[0]) % dims[1], k / (dims[0] * dims[1])]
    };
    let sim_wave = |w: u64| -> Result<(u64, f64, Vec<String>), String> {
        let first = w * r as u64;
        let count = (m - first).min(r as u64);
        let mut traces = vec![];
        let mut slot = 0u32;
        let mut warn = vec![];
        for i in 0..count {
            let wg_index = ((first + i) * cores).min(n - 1); // representative workgroups spread over the grid
            for g in 0..gpw {
                let ctx = GroupCtx { wg_size: l.wg, wg_id: wg_coords(wg_index), group: g, core: 0, slot, uniforms: l.uniforms.clone() };
                slot += 1;
                let t = cost::trace_group(code, &ctx, l.max_steps)?;
                for x in &t.warnings {
                    if !warn.contains(x) {
                        warn.push(x.clone());
                    }
                }
                traces.push(t);
            }
        }
        let seqs: Vec<&[usize]> = traces.iter().map(|t| t.pcs.as_slice()).collect();
        let mems: Vec<&[(u32, Vec<u64>)]> = traces.iter().map(|t| t.mem.as_slice()).collect();
        let mem = if l.mem { Some((mems.as_slice(), &mm)) } else { None };
        let c = cost::simulate_groups(&ins, &seqs, mem, p, false)?.0;
        Ok((count, c, warn))
    };
    let mut todo: Vec<u64> = vec![0];
    if waves > 1 {
        todo.push(waves - 1);
    }
    todo.truncate(l.max_waves.max(1));
    let mut first_cycles = 0.0;
    let mut total = 0.0;
    for &w in &todo {
        let (count, c, warn) = sim_wave(w)?;
        for x in warn {
            if !rep.warnings.contains(&x) {
                rep.warnings.push(x);
            }
        }
        rep.simulated_waves.push((w, count, c));
        if w == 0 {
            first_cycles = c;
        }
        total += c;
    }
    // waves not simulated cost as much as the first
    let simulated: u64 = todo.len() as u64;
    total += first_cycles * (waves - simulated.min(waves)) as f64;
    rep.core_cycles = total;
    rep.launch_core_ns = (m as f64 * l.wg_launch_ns).max((m * gpw as u64) as f64 * l.group_launch_ns);
    rep.time_ns = (total / l.clock_mhz * 1000.0).max(rep.launch_core_ns) + l.launch_ns;
    Ok(rep)
}

/// Workgroups on the busiest core: even split over clusters, then over each cluster's cores.
pub fn busiest_core_wgs(n: u64, l: &Launch) -> u64 {
    let k = l.clusters.len() as u64;
    if k == 0 || l.clusters.iter().map(|&c| c as u64).sum::<u64>() != l.cores as u64 {
        return n.div_ceil(l.cores as u64);
    }
    let share = n.div_ceil(k);
    let smallest = *l.clusters.iter().min().unwrap() as u64;
    share.div_ceil(smallest)
}
