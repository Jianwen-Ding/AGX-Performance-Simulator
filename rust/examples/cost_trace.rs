//! Checked cost-model tracing benchmark: [groups] [adds] [padding KiB] [repeats].
//! AGX_THREADS controls independent per-group tracing; no hardware is used.
//! AGX_BENCH_CODE optionally saves the compact program for full-dispatch comparisons.
use agx_sim::{cost::{self, GroupCtx}, kernels, par};
use std::{hint::black_box, time::Instant};

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let groups = *a.first().unwrap_or(&128);
    let adds = *a.get(1).unwrap_or(&128);
    let padding = *a.get(2).unwrap_or(&64);
    let repeats = *a.get(3).unwrap_or(&7);
    assert!(groups > 0 && adds > 0 && repeats > 0);
    let mut code = kernels::get_sr(80, 2);
    for _ in 0..adds { code.extend(kernels::e("iadd r2, r2, 1")); }
    code.extend([0x88, 0]);
    if let Ok(path) = std::env::var("AGX_BENCH_CODE") { std::fs::write(path, &code).unwrap(); }
    let mut expected = vec![0];
    expected.extend((0..=adds).map(|i| 4 + 8 * i));
    code.extend(vec![0xff; padding * 1024]);
    let contexts: Vec<_> = (0..groups as u32).map(|x| GroupCtx {
        wg_size: [32, 1, 1], wg_id: [x, 0, 0], group: 0, core: 0, slot: x % 96,
        uniforms: vec![], launch: Default::default(),
    }).collect();
    let mut samples = vec![];
    for _ in 0..=repeats {
        let start = Instant::now();
        let traces = par::map(&contexts, |ctx| cost::trace_group(black_box(&code), ctx, adds + 2).unwrap());
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        for tr in traces {
            assert_eq!(tr.pcs, expected);
            assert!(tr.mem.is_empty() && tr.skipped.is_empty() && tr.warnings.is_empty());
        }
    }
    samples.remove(0);
    samples.sort_by(f64::total_cmp);
    println!("groups={groups} adds={adds} padding_kib={padding} repeats={repeats} median_ms={:.3} instructions={}",
        samples[samples.len() / 2], groups * expected.len());
}
