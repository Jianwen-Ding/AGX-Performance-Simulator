//! Checked benchmark of independent workgroup tracing and decode-cache allocation.
//! cargo run --release --example functional_trace -- [workgroups] [steps] [padding KiB] [repeats]
//! AGX_THREADS controls tracing parallelism. Padding is unreachable and must never be decoded.
use agx_sim::{cost::Trace, func::{self, Dispatch, Memory}, kernels};
use std::{hint::black_box, time::Instant};

fn signature(traces: &[Vec<Trace>]) -> (usize, u64) {
    let mut count = 0;
    let mut sum = 0;
    for wg in traces { for tr in wg {
        count += tr.pcs.len();
        assert!(tr.mem.is_empty() && tr.skipped.is_empty() && tr.warnings.is_empty());
        for &pc in &tr.pcs { sum += pc as u64; }
    } }
    (count, sum)
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let groups = *a.first().unwrap_or(&128);
    let steps = *a.get(1).unwrap_or(&128);
    let padding = *a.get(2).unwrap_or(&64);
    let repeats = *a.get(3).unwrap_or(&5);
    assert!(groups > 0 && steps > 0 && repeats > 0);
    let mut code = kernels::get_sr(80, 2);
    for _ in 0..steps { code.extend(kernels::e("iadd r2, r2, 1")); }
    code.extend([0x88, 0]);
    let mut pcs = vec![0];
    pcs.extend((0..=steps).map(|i| 4 + 8 * i)); // adds and stop
    let expected = (groups * 2 * pcs.len(), groups as u64 * 2 * pcs.iter().map(|&p| p as u64).sum::<u64>());
    code.extend(vec![0xff; padding * 1024]);
    let d = Dispatch { code, grid: [groups as u32 * 64, 1, 1], wg: [64, 1, 1], shared_bytes: 0,
        uniforms: vec![], num_registers: 8, max_steps: (steps + 2) as u64, placement: 0, launch: Default::default() };
    let wgs: Vec<_> = (0..groups as u32).map(|x| [x, 0, 0]).collect();
    let mut samples = vec![];
    for _ in 0..=repeats {
        let start = Instant::now();
        let traces = func::trace_workgroups(black_box(&d), &Memory::default(), &wgs).unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(signature(&traces), expected);
        // All groups take the same static path, even with distinct global ids.
        for wg in &traces { assert_eq!(wg.len(), 2); for tr in wg { assert_eq!(tr.pcs, pcs); } }
        samples.push(elapsed);
    }
    samples.remove(0); // warm-up
    samples.sort_by(f64::total_cmp);
    println!("workgroups={groups} steps={steps} padding_kib={padding} repeats={repeats} median_ms={:.3} instructions={} checksum={}",
        samples[samples.len() / 2], expected.0, expected.1);
}
