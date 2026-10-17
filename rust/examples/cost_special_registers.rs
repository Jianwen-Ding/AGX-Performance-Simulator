//! Checked CPU cost-SR benchmark or JSONL value replay through synthetic address probes.
//! No memory contents/GPU timing are inferred: a lane/value pair maps injectively
//! to a captured 64-byte segment, value*32+lane, using a 64-bit byte address.
use agx_sim::{cost::{self, GroupCtx, PreparedTrace, SrLaunch}, kernels::{e, e2, get_sr, mov}};
use serde_json::{json, Value};
use std::{io::{self, BufRead}, time::Instant};

const PRELOAD: u32 = 0x5a5a1234;
const HIGH: u32 = 0xa5a5edcb;

fn program(sr: u32, width: &str, mask: u32, reads: usize) -> (Vec<u8>, Vec<usize>, usize) {
    let mut code = vec![];
    let mut pcs = vec![];
    let mut add = |bytes: Vec<u8>| { pcs.push(code.len()); code.extend(bytes); };
    add(get_sr(52, 2));
    // Arbitrary masks, not just contiguous prefixes; every excluded lane adds
    // one nested predicate. The probe never reads source data from memory.
    for t in 0..32 { if mask & (1 << t) == 0 { add(e(&format!("if_icmp r0l, nueq, r2, {t}, 1"))); } }
    add(mov(4, PRELOAD)); add(mov(5, HIGH));
    let mut read = get_sr(sr, 4);
    match width {
        "half" => read[1] &= !1, // clear Dt=2 (bit 8); D=8 is r4l
        "odd" => read[1] |= 2, // D=9, Dt=2 still writes r4 only
        "word" => {},
        _ => panic!("invalid width"),
    }
    for _ in 0..reads { add(read.clone()); }
    add(mov(20, 2048)); add(mov(21, 64));
    add(e("imadd r22, r2, r21, 0")); add(mov(23, 0)); add(mov(26, 0));
    for r in [4, 5] {
        add(e(&format!("imadd r24_r25, r{r}, r20, r22_r23")));
        add(e2(&format!("device_store 0, i32, x, r{r}, r24_r25, r26, unsigned, lsl 0, 0"),
            &format!("device_store 0, i32, x, r{r}, r24_r25, r26, unsigned, 0")));
    }
    add(vec![0x88, 0]);
    let steps = pcs.len();
    (code, pcs, steps)
}

fn check(trace: &cost::Trace, pcs: &[usize], expected: &[Option<u32>; 32], mask: u32, high_only: bool) {
    assert_eq!(trace.pcs, pcs);
    assert!(trace.warnings.is_empty(), "{:?}", trace.warnings);
    // Stores deliberately target synthetic unmapped addresses; their addresses
    // are captured before the approximate emulator skips the write itself.
    assert!(trace.skipped.iter().all(|s| s == "device_store"), "{:?}", trace.skipped);
    assert_eq!(trace.mem.len(), 2);
    for (i, high) in [false, true].into_iter().enumerate() {
        if high_only && !high { continue; } // retained odd-width run observed the adjacent register only
        let mut segments = vec![];
        for t in 0..32 {
            assert_eq!(expected[t].is_some(), mask & (1 << t) != 0);
            if let Some(v) = expected[t] { segments.push(32 * if high { HIGH as u64 } else { v as u64 } + t as u64); }
        }
        segments.sort_unstable();
        assert_eq!(trace.mem[i].0 as usize, pcs.len() - if high { 2 } else { 4 });
        assert_eq!(trace.mem[i].1, segments, "probe {i}");
    }
}

fn replay() {
    let mut cases = 0usize;
    for line in io::stdin().lock().lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let n = |key: &str| u32::try_from(v[key].as_u64().expect(key)).unwrap();
        let sr = n("selector"); assert!(sr < 256 && sr != 113);
        let mask = n("mask"); let width = v["width"].as_str().unwrap();
        let expected: [Option<u32>; 32] = v["expected"].as_array().unwrap().iter().map(|x|
            if x.is_null() { None } else { Some(u32::try_from(x.as_u64().unwrap()).unwrap()) }).collect::<Vec<_>>().try_into().unwrap();
        let ctx = GroupCtx { wg_size: [32, 1, 1], wg_id: [n("workgroup"), 0, 0], group: 0, core: 0, slot: 0,
            uniforms: vec![], launch: SrLaunch { tg_bytes: n("tg_bytes"), regs: n("regs"), context: Some(n("context")), pipeline: Some(n("pipeline")), ..Default::default() } };
        let (code, pcs, steps) = program(sr, width, mask, 1);
        check(&PreparedTrace::new(&code).trace_group(&ctx, steps).unwrap(), &pcs, &expected, mask, v["high_only"].as_bool().unwrap_or(false));
        cases += 1;
    }
    assert!(cases > 0, "empty replay");
    println!("{}", json!({"cases":cases,"passed":true}));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--replay") { assert_eq!(args.len(), 1); replay(); return; }
    let a: Vec<usize> = args.iter().map(|s| s.parse().unwrap()).collect();
    let groups = *a.first().unwrap_or(&128); let reads = *a.get(1).unwrap_or(&128); let repeats = *a.get(2).unwrap_or(&7);
    assert!(groups > 0 && reads > 0 && repeats > 0);
    // Only established full-mask 32-bit cases benchmark against the old binary.
    // Newly supported half/divergent behavior must not be timed as if old output were correct.
    for sr in [7, 18, 19, 80, 56, 57, 58, 59] {
        let (code, pcs, steps) = program(sr, "word", u32::MAX, reads);
        let prepared = PreparedTrace::new(&code);
        let mut samples = vec![];
        for _ in 0..=repeats {
            let start = Instant::now();
            let traces: Vec<_> = (0..groups).map(|x| {
                let ctx = GroupCtx { wg_size: [32, 1, 1], wg_id: [x as u32, 0, 0], group: 0, core: 0, slot: 0,
                    uniforms: vec![], launch: SrLaunch { tg_bytes: 513, regs: 65, ..Default::default() } };
                prepared.trace_group(&ctx, steps).unwrap()
            }).collect();
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            for (x, tr) in traces.iter().enumerate() {
                let expected = std::array::from_fn(|t| Some(match sr {
                    7 => 1, 18 => 768, 19 => 72, 80 => x as u32 * 32 + t as u32,
                    56 => t as u32 % 4, 57 => 4, 58 => t as u32, 59 => 32, _ => unreachable!(),
                }));
                check(tr, &pcs, &expected, u32::MAX, false);
            }
        }
        samples.remove(0); samples.sort_by(f64::total_cmp);
        println!("sr={sr} groups={groups} reads={reads} repeats={repeats} median_ms={:.6} checked_lanes={}", samples[samples.len()/2], groups*32);
    }
}
