//! Checked memory execution benchmark; [groups] [loads per group] [repeats].
//! No GPU or timing-model calibration is involved.
use agx_sim::{func::{self, Dispatch, Memory}, kernels::{e, get_sr, store}};
use std::{hint::black_box, time::Instant};

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let groups = *a.first().unwrap_or(&1024);
    let loads = *a.get(1).unwrap_or(&128);
    let repeats = *a.get(2).unwrap_or(&7);
    assert!(groups > 0 && loads > 0 && repeats > 0);
    let n = groups * 32;
    let input: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(0x9e3779b9)).collect();
    let mut code = get_sr(80, 2);
    for _ in 0..loads { code.extend(e("device_load 0, i32, x, r4, u8_u9, r2, unsigned")); }
    code.extend(e("wait 0"));
    code.extend(store(4, 6, 2));
    code.extend([0x88, 0]);
    let d = Dispatch { code, grid: [n as u32, 1, 1], wg: [32, 1, 1], shared_bytes: 0,
        uniforms: vec![(6, 0x10000), (7, 0), (8, 0x1000000), (9, 0)],
        num_registers: 8, max_steps: (loads + 4) as u64, placement: 0, launch: Default::default() };
    let mut samples = vec![];
    for _ in 0..=repeats {
        let mut mem = Memory::default();
        mem.map(0x10000, vec![0xcd; n * 4]);
        mem.map(0x1000000, input.iter().flat_map(|v| v.to_le_bytes()).collect());
        let start = Instant::now();
        let stats = func::run(black_box(&d), &mut mem).unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(stats.instructions, (groups * (loads + 4)) as u64);
        assert_eq!(stats.groups, groups as u64);
        assert_eq!(mem.words(0x10000, n).unwrap(), input);
    }
    samples.remove(0);
    samples.sort_by(f64::total_cmp);
    println!("groups={groups} loads={loads} repeats={repeats} median_ms={:.3} instructions={} checked_words={n}",
        samples[samples.len() / 2], groups * (loads + 4));
}
