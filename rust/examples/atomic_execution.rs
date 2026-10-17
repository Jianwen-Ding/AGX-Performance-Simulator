//! Checked functional atomic execution: [groups] [atomics/group] [lanes] [repeats] [source register].
//! Device and threadgroup cases use independent per-lane counters, not GPU timing.
use agx_sim::{func::{self, Dispatch, Memory}, kernels::{atomic_add_ret, e, get_sr, mov, store, tg, tg_atomic_add}};
use std::time::Instant;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let groups = *a.first().unwrap_or(&512);
    let steps = *a.get(1).unwrap_or(&128);
    let lanes = *a.get(2).unwrap_or(&32);
    let repeats = *a.get(3).unwrap_or(&7);
    let source = *a.get(4).unwrap_or(&6);
    assert!(groups > 0 && steps > 0 && repeats > 0 && matches!(lanes, 1 | 32));
    assert!(matches!(source, 6 | 31)); // 31 exposes the old unused-CAS-register bug
    let n = groups * lanes;
    for kind in ["device", "threadgroup"] {
        let mut code = get_sr(80, 2);
        if kind == "threadgroup" {
            code.extend(get_sr(58, 5));
            code.extend(mov(7, 4));
            code.extend(e("imadd r5, r5, r7, 0"));
        }
        code.extend(mov(source as u32, 1));
        for _ in 0..steps {
            code.extend(if kind == "device" { atomic_add_ret(4, 2, source as u32) } else { tg_atomic_add(4, 5, source as u32, true) });
        }
        if kind == "threadgroup" {
            code.extend(tg(true, "i32", 8, Some(5), 0, 1));
            code.extend(store(8, 8, 2)); // expose each lane's final shared-memory counter
        }
        code.extend(store(4, 6, 2)); // every lane's final atomic return
        code.extend([0x88, 0]);
        let expected_steps = steps + if kind == "device" { 4 } else { 9 };
        let d = Dispatch { code, grid: [n as u32, 1, 1], wg: [lanes as u32, 1, 1], shared_bytes: 128,
            uniforms: vec![(6, 0x10000), (7, 0), (8, 0x1000000), (9, 0)], num_registers: (source + 1).max(16),
            max_steps: expected_steps as u64, placement: 0, launch: Default::default() };
        let mut samples = vec![];
        for _ in 0..=repeats {
            let mut mem = Memory::default();
            mem.map(0x10000, vec![0xcd; n * 4]);
            mem.map(0x1000000, vec![0; n * 4]);
            let start = Instant::now();
            let stats = func::run(&d, &mut mem).unwrap();
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(stats.groups, groups as u64);
            assert_eq!(stats.instructions, (groups * expected_steps) as u64);
            assert_eq!(mem.words(0x10000, n).unwrap(), vec![(steps - 1) as u32; n]);
            assert_eq!(mem.words(0x1000000, n).unwrap(), vec![steps as u32; n]);
        }
        samples.remove(0);
        samples.sort_by(f64::total_cmp);
        println!("kind={kind} groups={groups} steps={steps} lanes={lanes} repeats={repeats} median_ms={:.3} checked_words={} atomics={}",
            samples[samples.len() / 2], n * 2, n * steps);
    }
}
