//! Supplied M2 core partitions with exhaustive, isolated functional traces.
//!
//! Linear-ID order and occupancy-sized waves are model assumptions. Core maps
//! do not establish SR17 slots, concurrent residency, cache state or frequency.
use crate::{cost::{self, MemModel, Params}, dispatch::{self, Launch, Report}, func::{self, Dispatch, Memory, PreparedDispatch}};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug)]
pub struct Wave {
    pub index: usize,
    pub workgroups: Vec<u64>,
    pub cycles: f64,
    /// Mean fraction of this wave's device segments touched by other cores.
    pub sharing: f64,
}

#[derive(Debug)]
pub struct Core {
    pub id: u32,
    pub workgroups: Vec<u64>,
    pub waves: Vec<Wave>,
    pub cycles: f64,
    pub launch_ns: f64,
    /// Core completion estimate before fixed dispatch overhead.
    pub time_ns: f64,
}

#[derive(Debug)]
pub struct MappedReport {
    /// Critical core's execution/launch estimates; busiest_core_wgs is the
    /// maximum count across all cores, not necessarily this critical core.
    pub summary: Report,
    pub cores: Vec<Core>,
    pub critical_core: u32,
    /// Exact union over every isolated functional trace (not measured traffic).
    pub global_device_segments: usize,
    pub resident_slc_assumed: bool,
    /// False if texture/image/stack accesses lack functional memory traces.
    pub device_footprint_complete: bool,
}

/// Preflight before reading shaders, running preambles or allocating buffers.
/// `d` can have empty code; only launch metadata is inspected here.
pub fn validate(d: &Dispatch, l: &Launch, cores: &[u32]) -> Result<(), String> {
    if l.busiest_core_wgs_override.is_some() {
        return Err("M2 core map and busiest_core_wgs are mutually exclusive".into());
    }
    l.validate()?;
    if l.cores != 38 { return Err("M2 core map requires the 38-core M2 Max profile".into()); }
    if d.grid != l.grid || d.wg != l.wg || d.shared_bytes != l.tg_bytes || d.uniforms != l.uniforms
        || d.launch.regs != l.regs || d.launch.tg_bytes != l.tg_bytes {
        return Err("M2 core map functional and cost launch metadata disagree".into());
    }
    func::validate_m2_core_map(d, cores)
}

/// Trace every workgroup once, partition by supplied core, and evaluate every
/// occupancy-sized wave in linear workgroup order. Launch::max_waves applies
/// only to the older representative-core path; no waves are extrapolated here.
/// Each workgroup starts from its own memory snapshot. Cross-workgroup effects
/// and temporal overlap between cores are not reconstructed by this model.
pub fn simulate(d: &Dispatch, mem: &Memory, l: &Launch, p: &Params, cores: &[u32]) -> Result<MappedReport, String> {
    validate(d, l, cores)?;
    let prepared = PreparedDispatch::with_m2_core_map(d, cores)?;
    let ins = cost::decode(&d.code, p)?;
    let footprint_complete = !ins.iter().any(|i| i.name.starts_with("texture_")
        || i.name.starts_with("image_") || i.name.starts_with("stack_")
        || matches!(i.name.as_str(), "tex_state_store" | "sampler_state_store"));
    let segments_of = |t: &cost::Trace| t.device_segments.iter()
        .flat_map(|(_, segments)| segments.iter().copied()).collect::<HashSet<_>>();
    let dims = std::array::from_fn::<_, 3, _>(|i| d.grid[i].div_ceil(d.wg[i]) as u64);
    let coords = |k: usize| { let k = k as u64;
        [(k % dims[0]) as u32, ((k / dims[0]) % dims[1]) as u32, (k / dims[0] / dims[1]) as u32] };
    let mut partitions = BTreeMap::<u32, Vec<usize>>::new();
    for (i, &core) in cores.iter().enumerate() { partitions.entry(core).or_default().push(i); }
    let threads = d.wg.iter().product::<u32>();
    let gpw = threads.div_ceil(32);
    let (cap, resident) = dispatch::occupancy(l);
    let ids: Vec<_> = (0..cores.len()).collect();
    let traces = crate::par::map(&ids, |&i| prepared.trace_workgroup(mem, coords(i)))
        .into_iter().collect::<Result<Vec<_>, _>>()?;
    let mut warnings = vec![];
    if !footprint_complete {
        warnings.push("M2 core map footprint omits untraced texture/image/stack accesses; SLC residency not assumed".into());
    }
    let mut segment_sets = BTreeMap::<u32, HashSet<u64>>::new();
    let mut global = HashSet::new();
    for (&core, ts) in cores.iter().zip(&traces) {
        let segments = segment_sets.entry(core).or_default();
        for t in ts {
            if !t.skipped.is_empty() { return Err(format!("M2 core map trace skipped instructions: {:?}", t.skipped)); }
            for warning in &t.warnings { if !warnings.contains(warning) { warnings.push(warning.clone()); } }
            let captured = segments_of(t);
            segments.extend(captured.iter().copied()); global.extend(captured);
        }
    }
    let mm = MemModel { fetch_line_ns: dispatch::fetch_line_ns(), icache_bytes: dispatch::icache_bytes(), clock_mhz: l.clock_mhz, active_cores: partitions.len() as f64,
        dram_gbps: l.dram_gbps, warm: false, ..Default::default() };
    let resident_slc = footprint_complete && global.len() as u128 * 64 <= mm.slc_bytes as u128;
    let mut out = vec![];
    // Sequential waves bound temporary scheduler state; instruction metadata and
    // immutable traces remain shared. No GPU work is involved.
    for (&id, wgs) in &partitions {
        let mut waves = vec![];
        let mut total = 0.0;
        let mut durations = vec![];
        for (index, chunk) in wgs.chunks(resident as usize).enumerate() {
            let ts: Vec<_> = chunk.iter().flat_map(|&i| traces[i].iter()).collect();
            let seqs: Vec<&[usize]> = ts.iter().map(|t| t.pcs.as_slice()).collect();
            let mems: Vec<&[(u32, Vec<u64>)]> = ts.iter().map(|t| t.mem.as_slice()).collect();
            let masked: Vec<&[u32]> = ts.iter().map(|t| t.masked.as_slice()).collect();
            let mut segments = HashSet::new();
            for t in &ts { segments.extend(segments_of(t)); }
            let sharing = if segments.is_empty() || partitions.len() == 1 { 0.0 } else {
                segment_sets.iter().filter(|(&other, _)| other != id).map(|(_, set)|
                    segments.intersection(set).count() as f64 / segments.len() as f64).sum::<f64>() / (partitions.len() - 1) as f64
            };
            let wave_mem = MemModel { sharing, resident_slc, ..mm.clone() };
            let wg_of: Vec<u32> = (0..chunk.len()).flat_map(|i| std::iter::repeat_n(i as u32, gpw as usize)).collect();
            let (cycles, _, finish) = cost::simulate_groups_finish(&ins, &seqs, Some(&masked), l.mem.then_some((mems.as_slice(), &wave_mem)), p, false, Some(&wg_of))?;
            total += cycles;
            durations.extend(dispatch::workgroup_finish(&finish, &wg_of, chunk.len()));
            waves.push(Wave { index, workgroups: chunk.iter().map(|&i| i as u64).collect(), cycles, sharing });
        }
        if dispatch::refill() && resident > 1 {
            total = dispatch::refill_makespan(&durations, resident as usize);
        }
        let count = wgs.len() as f64;
        let launch_ns = (count * l.wg_launch_ns).max(count * gpw as f64 * l.group_launch_ns);
        let time_ns = (total / l.clock_mhz * 1000.0).max(launch_ns);
        if !time_ns.is_finite() { return Err("M2 core map predicted time is non-finite".into()); }
        out.push(Core { id, workgroups: wgs.iter().map(|&i| i as u64).collect(), waves, cycles: total, launch_ns, time_ns });
    }
    // BTreeMap iteration breaks exact ties deterministically by the smaller ID.
    let critical = out.iter().enumerate().fold(0, |best, (i, c)| if c.time_ns > out[best].time_ns { i } else { best });
    let core = &out[critical];
    let time_ns = core.time_ns + l.launch_ns;
    if !time_ns.is_finite() { return Err("M2 core map predicted time is non-finite".into()); }
    let summary = Report { workgroups: cores.len() as u64, groups_per_wg: gpw, group_cap_per_core: cap, wgs_per_core: resident,
        busiest_core_wgs: partitions.values().map(|w| w.len() as u64).max().unwrap(), waves: core.waves.len() as u64,
        simulated_waves: core.waves.iter().map(|w| (w.index as u64, w.workgroups.len() as u64, w.cycles)).collect(),
        core_cycles: core.cycles, launch_core_ns: core.launch_ns, code_lines: 0, time_ns, warnings };
    Ok(MappedReport { summary, critical_core: core.id, cores: out, global_device_segments: global.len(), resident_slc_assumed: resident_slc, device_footprint_complete: footprint_complete })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{e, e2, get_sr, mov, store};

    fn fixture(code: Vec<u8>, n: u32, wg: u32) -> (Dispatch, Launch) {
        let l = Launch { grid: [n, 1, 1], wg: [wg, 1, 1], ..Default::default() };
        let d = Dispatch { code, grid: l.grid, wg: l.wg, shared_bytes: 0, uniforms: vec![],
            num_registers: 32, max_steps: 10000, placement: 0,
            launch: cost::SrLaunch { regs: 64, ..Default::default() } };
        (d, l)
    }

    #[test]
    fn mapped_waves_keep_interleaved_core_partitions_and_exact_footprint() {
        let n = 80 * 32;
        let code = [get_sr(80, 1), get_sr(20, 2), mov(5, 0),
            e("device_load 0, i32, x, r4, u10_u11, r5, unsigned"), e("wait 0"),
            e("imadd r3, r2, r4, r1"), store(2, 8, 3), e("wait 0"), e("stop")].concat();
        let (mut d, mut l) = fixture(code, n, 32);
        d.uniforms = vec![(8, 0x1000), (9, 0), (10, 0), (11, 16)];l.uniforms = d.uniforms.clone();
        let cores: Vec<_> = (0..80).map(|i| if i % 4 == 0 { 1 } else { 0 }).collect();
        let plane = std::sync::Arc::new(vec![0x5a; n as usize * 4]);
        let mut mem = Memory::default();
        for core in 0..2 { mem.regions.push((0x1000 + core * n as u64 * 4, plane.clone())); }
        mem.map(1u64 << 36, n.to_le_bytes().to_vec());
        let report = simulate(&d, &mem, &l, &Params::new().unwrap(), &cores).unwrap();
        assert_eq!(report.cores.len(), 2); // active, not configured 38
        assert_eq!(report.summary.busiest_core_wgs, 60);
        assert_eq!(report.global_device_segments, 161); // 80 * 2 store lines + one shared load
        assert!(report.device_footprint_complete && report.resident_slc_assumed);
        assert!(report.summary.warnings.is_empty());
        let mut visits = vec![];
        for c in &report.cores {
            let expected: Vec<_> = (0..80).filter(|&i| cores[i] == c.id).map(|i| i as u64).collect();
            assert_eq!(c.workgroups, expected);
            assert_eq!(c.waves.iter().flat_map(|w| w.workgroups.iter().copied()).collect::<Vec<_>>(), expected);
            for wave in &c.waves {
                assert!(wave.workgroups.iter().all(|&i| cores[i as usize] == c.id));
                assert!((wave.sharing - 1.0 / (wave.workgroups.len() * 2 + 1) as f64).abs() < 1e-12);
                visits.extend(wave.workgroups.iter().copied());
            }
            assert_eq!(c.cycles, c.waves.iter().map(|w| w.cycles).sum::<f64>());
        }
        visits.sort_unstable();assert_eq!(visits, (0..80).collect::<Vec<_>>());
        assert_eq!(report.cores[0].waves.iter().map(|w| w.workgroups.len()).collect::<Vec<_>>(), vec![52, 8]);
        assert_eq!(report.cores[1].waves.len(), 1);
        assert!(mem.regions[..2].iter().all(|(_, p)| std::sync::Arc::ptr_eq(p, &plane)));
        assert!(plane.iter().all(|&v| v == 0x5a));
        assert_eq!(mem.read(1u64 << 36, 4).unwrap(), n as u64);
    }

    #[test]
    fn mapped_atomic_word_addresses_normalize_to_64_byte_segments() {
        let code = [get_sr(80, 5), mov(6, 1), crate::kernels::atomic_add(8, 5, 6), e("wait 0"), e("stop")].concat();
        let (mut d, mut l) = fixture(code, 32, 32);
        d.uniforms = vec![(8, 0x1000), (9, 0)];l.uniforms = d.uniforms.clone();
        let mut mem = Memory::default();mem.map(0x1000, vec![0; 128]);
        let r = simulate(&d, &mem, &l, &Params::new().unwrap(), &[0]).unwrap();
        assert_eq!(r.global_device_segments, 2);assert!(r.device_footprint_complete);
        assert_eq!(mem.words(0x1000, 32).unwrap(), vec![0; 32]);
    }

    #[test]
    fn mapped_atomic_byte_ranges_cover_crossing_and_unaligned_accesses() {
        for (bytes, base, op, variant) in [(4, 0x103fu32, 0, 2), (8, 0x103c, 3, 1), (8, 0x103f, 3, 3)] {
            let original = crate::kernels::atomic_add(8, 5, 6);
            let number = crate::isa::opcode_to_number(&original);
            let desc = crate::isa::get_instruction_descriptor(number).unwrap();
            let mut fields: std::collections::HashMap<_, _> = desc.decode_fields(number).into_iter().map(|(k, v)| (k, v as i64)).collect();
            fields.insert("op", op);fields.insert("u5", variant);
            let raw = desc.encode_fields(&fields).unwrap();
            let atomic: Vec<_> = (0..8).map(|i| (raw >> (8 * i)) as u8).collect();
            let code = [mov(5, 0), mov(6, 0), mov(7, 0), atomic, e("wait 0"), e("stop")].concat();
            let (mut d, mut l) = fixture(code, 32, 32);
            d.uniforms = vec![(8, base), (9, 0)];l.uniforms = d.uniforms.clone();
            let mut mem = Memory::default();mem.map(0x1000, vec![0; 128]);
            let ts = func::trace_workgroups_with_m2_core_map(&d, &mem, &[[0; 3]], &[0]).unwrap();
            let expected: HashSet<_> = (0..bytes).map(|i| (base as u64 + i) / 64).collect();
            assert_eq!(ts[0][0].device_segments, vec![(3, vec![64, 65])]);
            assert_eq!(ts[0][0].mem, vec![(3, vec![base as u64 / 4; 32])]);
            let r = simulate(&d, &mem, &l, &Params::new().unwrap(), &[0]).unwrap();
            assert_eq!(r.global_device_segments, expected.len());
            assert_eq!(mem.read_bytes(0x1000, 128).unwrap(), vec![0; 128]);
        }
    }

    #[test]
    fn mapped_critical_core_can_have_fewer_workgroups() {
        // Core 0: eleven one-trip workgroups. Core 1: one 129-trip workgroup.
        let mut code = get_sr(20, 4);code.extend(mov(6, 128));code.extend(mov(7, 1));
        code.extend(e("imadd r4, r4, r6, r7"));code.extend(mov(3, 0));
        code.extend(e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1"));
        let mut lp = e("iadd r3, 1, r3.sx");lp.extend(e("while_icmp r0l, ult, r3, r4, 1"));
        lp.extend(e(&format!("jmp_exec_any pc-{}", lp.len())));code.extend(lp);
        code.extend(e("pop_exec r0l, 1"));code.extend(e("stop"));
        let (d, mut l) = fixture(code, 12 * 32, 32);
        l.mem = false;l.wg_launch_ns = 0.0;l.group_launch_ns = 0.0;
        let mut cores = vec![0; 12];cores[11] = 1;
        let r = simulate(&d, &Memory::default(), &l, &Params::new().unwrap(), &cores).unwrap();
        assert_eq!(r.summary.busiest_core_wgs, 11);
        assert_eq!(r.critical_core, 1);assert_eq!(r.cores[1].workgroups, vec![11]);
        assert!(r.cores[1].cycles > r.cores[0].cycles);
        assert_eq!(r.summary.core_cycles, r.cores[1].cycles);
    }

    #[test]
    fn mapped_validation_precedes_decode_and_rejects_conflicting_context() {
        let (mut d, l) = fixture(vec![0xff], 12 * 128, 128);
        let cores = vec![0; 12];let p = Params::new().unwrap();
        assert!(simulate(&d, &Memory::default(), &l, &p, &cores[..11]).unwrap_err().contains("entries"));
        let mut bad = cores.clone();bad[0] = 25;
        assert!(simulate(&d, &Memory::default(), &l, &p, &bad).unwrap_err().contains("disabled"));
        bad[0] = 16;
        assert!(simulate(&d, &Memory::default(), &l, &p, &bad).unwrap_err().contains("spans clusters"));
        let peak = Launch { busiest_core_wgs_override: Some(1), ..l.clone() };
        assert!(simulate(&d, &Memory::default(), &peak, &p, &cores).unwrap_err().contains("mutually exclusive"));
        let custom = Launch { cores: 2, ..l.clone() };
        assert!(simulate(&d, &Memory::default(), &custom, &p, &cores).unwrap_err().contains("38-core"));
        d.launch.regs = 32;
        assert!(simulate(&d, &Memory::default(), &l, &p, &cores).unwrap_err().contains("metadata disagree"));
    }

    #[test]
    fn mapped_descriptor_reads_alone_mark_the_footprint_incomplete() {
        for (name, slot) in [("tex_state_store", "TS"), ("sampler_state_store", "SS")] {
            let desc = crate::isa::desc_by_name(name).unwrap();
            let fields = [(slot, 0), ("B", 3), ("O", 0)].into_iter().collect();
            let raw = desc.encode_fields(&fields).unwrap();
            let mut code: Vec<_> = (0..8).map(|i| (raw >> (8 * i)) as u8).collect();code.extend(e("stop"));
            let (mut d, mut l) = fixture(code, 32, 32);
            d.uniforms = vec![(2, 0x1000), (3, 0)];l.uniforms = d.uniforms.clone();
            let mut mem = Memory::default();mem.map(0x1000, vec![0; 32]);
            let r = simulate(&d, &mem, &l, &Params::new().unwrap(), &[0]).unwrap();
            assert!(!r.device_footprint_complete && !r.resident_slc_assumed);
            assert_eq!(r.global_device_segments, 0); // descriptor reads are not yet traced
            assert!(r.summary.warnings.iter().any(|w| w.contains("footprint omits")));
            assert!(mem.tex_state.is_empty() && mem.samp_state.is_empty());
        }
    }

    #[test]
    fn mapped_two_dimensional_coordinates_and_single_core_sharing() {
        let (mut d, mut l) = fixture(e("stop"), 32, 32);
        d.grid = [32, 12, 1];l.grid = d.grid;
        let cores = [0, 16, 32, 48, 0, 16, 32, 48, 0, 16, 32, 48];
        let p = Params::new().unwrap();
        let r = simulate(&d, &Memory::default(), &l, &p, &cores).unwrap();
        assert_eq!(r.cores.iter().map(|c| c.workgroups.clone()).collect::<Vec<_>>(),
            vec![vec![0, 4, 8], vec![1, 5, 9], vec![2, 6, 10], vec![3, 7, 11]]);
        let single = simulate(&d, &Memory::default(), &l, &p, &[0; 12]).unwrap();
        assert_eq!(single.cores.len(), 1);assert_eq!(single.cores[0].waves[0].sharing, 0.0);
    }
}
