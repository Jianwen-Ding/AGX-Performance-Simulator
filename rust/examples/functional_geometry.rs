//! Checked CPU dispatch setup/barrier benchmark; no hardware timing.
use agx_sim::func::{self, Dispatch, Memory};
use std::time::Instant;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let workgroups = *a.first().unwrap_or(&2048);
    let repeats = *a.get(1).unwrap_or(&7);
    assert!(workgroups > 0 && repeats > 0);
    for threads in [1u32, 32, 256] {
        for barriers in [0, 8] {
            for placement in [0, 17] {
                let mut code = [0x78, 0].repeat(barriers);
                code.extend([0x88, 0]);
                let d = Dispatch { code, grid: [threads * workgroups as u32, 1, 1], wg: [threads, 1, 1],
                    shared_bytes: 0, uniforms: vec![], num_registers: 8, max_steps: (barriers + 1) as u64,
                    placement, launch: Default::default() };
                let groups = workgroups as u64 * threads.div_ceil(32) as u64;
                let mut samples = vec![];
                for _ in 0..=repeats {
                    let mut memory = Memory::default();
                    let start = Instant::now();
                    let stats = func::run(&d, &mut memory).unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(stats.groups, groups);
                    assert_eq!(stats.instructions, groups * (barriers + 1) as u64);
                    assert!(!stats.placement_read && memory.regions.is_empty());
                    assert_eq!(memory.writer, ((workgroups as u64 - 1) << 16) | (threads.div_ceil(32) - 1) as u64);
                    samples.push(elapsed);
                }
                samples.remove(0);
                samples.sort_by(f64::total_cmp);
                println!("threads={threads} barriers={barriers} placement={placement} workgroups={workgroups} repeats={repeats} median_ms={:.6} checked_instructions={}", samples[samples.len()/2], groups * (barriers + 1) as u64);
            }
        }
    }
}
