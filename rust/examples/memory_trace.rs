//! Checked memory tracing: [workgroups] [loads] [word stride] [repeats].
//! Both tracers must produce every expected PC and segment; no GPU is used.
use agx_sim::{cost::{self, GroupCtx, Trace}, func::{self, Dispatch, Memory}, kernels::{e, get_sr, mov}, par};
use std::time::Instant;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let groups = *a.first().unwrap_or(&128);
    let loads = *a.get(1).unwrap_or(&128);
    let stride = *a.get(2).unwrap_or(&1);
    let repeats = *a.get(3).unwrap_or(&7);
    assert!(groups > 0 && loads > 0 && stride > 0 && repeats > 0);
    let mut code = vec![];
    let mut pcs = vec![];
    let mut emit = |bytes: Vec<u8>| { pcs.push(code.len()); code.extend(bytes); };
    emit(get_sr(80, 2));
    emit(mov(6, stride as u32));
    emit(e("imadd r2, r2, r6, 0"));
    emit(e("iadd r24, u8, 0"));
    emit(e("iadd r25, u9, 0"));
    for _ in 0..loads { emit(e("device_load 0, i32, x, r4, r24_r25, r2, unsigned")); }
    emit(e("wait 0"));
    emit(vec![0x88, 0]);
    let base = cost::TABLE_VA as u64;
    let uniforms = vec![(8, base as u32), (9, (base >> 32) as u32)];
    let contexts: Vec<_> = (0..groups as u32).map(|x| GroupCtx { wg_size: [32, 1, 1], wg_id: [x, 0, 0],
        group: 0, core: 0, slot: x % 96, uniforms: uniforms.clone(), launch: Default::default() }).collect();
    let d = Dispatch { code, grid: [groups as u32 * 32, 1, 1], wg: [32, 1, 1], shared_bytes: 0,
        uniforms, num_registers: 32, max_steps: pcs.len() as u64, placement: 0, launch: Default::default() };
    let mut memory = Memory::default();
    memory.map(base, vec![0; 4 * groups * 32 * stride]);
    // The approximate cost interpreter's zero table holds 65,536 words.
    assert!(groups * 32 * stride <= 65536);
    let wgs: Vec<_> = contexts.iter().map(|c| c.wg_id).collect();
    for kind in ["cost", "functional"] {
        let mut samples = vec![];
        for _ in 0..=repeats {
            let start = Instant::now();
            let traces: Vec<Trace> = if kind == "cost" {
                let prepared = cost::PreparedTrace::new(&d.code);
                par::map(&contexts, |ctx| prepared.trace_group(ctx, pcs.len()).unwrap())
            } else {
                func::PreparedDispatch::new(&d).unwrap().trace_workgroups(&memory, &wgs).unwrap()
                    .into_iter().map(|mut wg| { assert_eq!(wg.len(), 1); wg.remove(0) }).collect()
            };
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            for (g, tr) in traces.iter().enumerate() {
                assert_eq!(tr.pcs, pcs);
                assert!(tr.skipped.is_empty() && tr.warnings.is_empty());
                let mut expected: Vec<_> = (0..32).map(|t| (base + 4 * ((g * 32 + t) * stride) as u64) / 64).collect();
                expected.sort_unstable(); expected.dedup();
                assert_eq!(tr.mem.len(), loads);
                for (i, (step, segments)) in tr.mem.iter().enumerate() {
                    assert_eq!(*step as usize, 5 + i);
                    assert_eq!(segments, &expected);
                }
            }
        }
        samples.remove(0);
        samples.sort_by(f64::total_cmp);
        println!("kind={kind} groups={groups} loads={loads} stride={stride} repeats={repeats} median_ms={:.3} instructions={} accesses={}",
            samples[samples.len() / 2], groups * pcs.len(), groups * loads);
    }
}
