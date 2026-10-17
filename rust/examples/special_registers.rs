//! Checked CPU functional get_sr benchmark: [groups] [reads/group] [repeats].
use agx_sim::{func::{self, Dispatch, Memory}, kernels::{e, get_sr, store}};
use std::time::Instant;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let groups = *a.first().unwrap_or(&256);
    let reads = *a.get(1).unwrap_or(&128);
    let repeats = *a.get(2).unwrap_or(&7);
    assert!(groups > 0 && reads > 0 && repeats > 0);
    let n = groups * 32;
    for sr in [7u32, 18, 19, 80, 56, 57, 58, 59] {
        for active in [32usize, 13] {
            let mut code = get_sr(80, 2);
            code.extend(get_sr(52, 5));
            code.extend(e(&format!("if_icmp r0l, ult, r5, {active}, 1")));
            for _ in 0..reads { code.extend(get_sr(sr, 4)); }
            code.extend(store(4, 6, 2));
            code.extend([0x88, 0]);
            let d = Dispatch { code, grid: [n as u32, 1, 1], wg: [32, 1, 1], shared_bytes: 513,
                uniforms: vec![(6, 0x10000), (7, 0)], num_registers: 8, max_steps: (reads + 5) as u64,
                placement: 0, launch: agx_sim::cost::SrLaunch { tg_bytes: 513, regs: 65, ..Default::default() } };
            let expected: Vec<u32> = (0..n).map(|i| {
                let lane = i % 32;
                if lane >= active { return 0xcdcdcdcd; }
                match sr {
                    7 => 1, 18 => 768, 19 => 72, 80 => i as u32,
                    56 => (lane % 4) as u32,
                    57 => (active - lane / 4 * 4).min(4) as u32,
                    58 => lane as u32, 59 => active as u32, _ => unreachable!(),
                }
            }).collect();
            let mut samples = vec![];
            for _ in 0..=repeats {
                let mut mem = Memory::default();
                mem.map(0x10000, vec![0xcd; n * 4]);
                let start = Instant::now();
                let stats = func::run(&d, &mut mem).unwrap();
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(stats.groups, groups as u64);
                assert_eq!(stats.instructions, (groups * (reads + 5)) as u64);
                assert_eq!(mem.words(0x10000, n).unwrap(), expected);
            }
            samples.remove(0); samples.sort_by(f64::total_cmp);
            println!("sr={sr} active={active} groups={groups} reads={reads} repeats={repeats} median_ms={:.3} checked_words={n}", samples[samples.len() / 2]);
        }
    }
}
