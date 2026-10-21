//! Instructions issued with an empty execution mask (playground results/2026-10-20-b31-empty-mask-cost): the B31
//! probe programs, costed by the model. Masked ALU work costs about as little as a jump over it; masked memory
//! instructions keep part of their cost.

use agx_sim::cost::{self, Params};
use agx_sim::kernels::{e, e2, mov, store, tg};
use agx_sim::func::{self, Dispatch, Memory};

/// One instance of the probe's if body class (b31.py CLS).
fn class(cls: &str) -> Vec<u8> {
    match cls {
        "iadd" => e("iadd r10, r10, 1"),
        "fmadd32" => e("fmadd32 r11, r11, r12, r13"),
        "load" => [e("device_load 0, i32, x, r14, u8_u9, r5, unsigned"), e("wait 0")].concat(),
        "store" => store(15, 8, 6),
        "tgload" => [tg(true, "i32", 16, Some(22), 0, 1), e("wait 0")].concat(),
        "shuffle" => e("simd_shuffle r17, r17, r18l"),
        _ => panic!("class"),
    }
}

/// b31.py body: `trips` trips of a counted loop, each with an if whose body is `k` copies of `cls`.
/// mode: base (no if), empty (no lane enters, no jump), skip (jmp_exec_none over the body), active (every lane).
fn probe(cls: &str, k: usize, mode: &str, trips: u32) -> Vec<u8> {
    let mut b = [mov(3, 0), mov(4, trips), mov(5, 0), mov(22, 0), mov(29, 0xffffffff), mov(10, 0), mov(11, 0), mov(12, 0x3f800000),
        mov(13, 0x3f800000), mov(17, 0), mov(18, 1), mov(26, 1000), e("iadd r6, r1, r26"), mov(15, 0x77777777)].concat();
    let inner = class(cls).repeat(k);
    let trip = match mode {
        "base" => vec![],
        "active" => [mov(27, 1000), e("if_icmp r0l, ult, r1, r27, 1"), inner, e("pop_exec r0l, 1")].concat(),
        _ => {
            let j = format!("jmp_exec_none pc+{}", 6 + inner.len());
            let jump = if mode == "skip" { e2(&j, &j) } else { vec![] };
            [e("if_icmp r0l, ueq, r1, r29, 1"), jump, inner, e("pop_exec r0l, 1")].concat()
        }
    };
    let lp = [trip, e("iadd r3, 1, r3.sx"), e("while_icmp r0l, ult, r3, r4, 1")].concat();
    let back = format!("jmp_exec_any pc-{}", lp.len());
    b.extend(e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1"));
    b.extend(lp);
    b.extend(e2(&back, &back));
    b.extend(e("pop_exec r0l, 1"));
    b.extend(e("iadd r2, r3, 0"));
    b
}

fn params() -> Params {
    let mut p = Params::new().unwrap();
    cost::fit_branch(&mut p).unwrap();
    p
}

/// Model cycles of one SIMD group running `code`.
fn cycles(code: &[u8], p: &Params) -> f64 {
    let ins = cost::decode(code, p).unwrap();
    let tr = cost::trace(code, 32, 1_000_000).unwrap();
    cost::simulate_trace(&ins, &tr, p, 1, false).unwrap().0
}

/// Model cycles per trip (difference of two trip counts, so the program's set-up cancels).
fn per_trip(cls: &str, k: usize, mode: &str, p: &Params) -> f64 {
    (cycles(&probe(cls, k, mode, 120), p) - cycles(&probe(cls, k, mode, 20), p)) / 100.0
}

#[test]
fn trace_records_the_empty_mask_steps() {
    // if_icmp (no lane passes), 3 iadds, pop_exec, stop: the iadds and the pop_exec issue with an empty mask
    let code = [e("if_icmp r0l, ueq, r1, r29, 1"), e("iadd r10, r10, 1").repeat(3), e("pop_exec r0l, 1"), vec![0x88, 0]].concat();
    let code = [mov(29, 0xffffffff), code].concat();
    let tr = cost::trace(&code, 32, 100).unwrap();
    assert_eq!(tr.pcs.len(), 7);
    assert_eq!(tr.masked, vec![2, 3, 4, 5]);
    // all lanes active: nothing masked
    let code = [mov(29, 1000), e("if_icmp r0l, ult, r1, r29, 1"), e("pop_exec r0l, 1"), vec![0x88, 0]].concat();
    assert!(cost::trace(&code, 32, 100).unwrap().masked.is_empty());
    // the functional simulator's trace records the same steps
    let code = [mov(29, 0xffffffff), e("if_icmp r0l, ueq, r1, r29, 1"), e("iadd r10, r10, 1").repeat(3), e("pop_exec r0l, 1"), vec![0x88, 0]].concat();
    let d = Dispatch { code: code.clone(), grid: [32, 1, 1], wg: [32, 1, 1], shared_bytes: 0, uniforms: vec![], num_registers: 32,
        max_steps: 100, placement: 0, launch: Default::default() };
    let f = func::trace_workgroups(&d, &Memory::default(), &[[0, 0, 0]]).unwrap().remove(0).remove(0);
    let c = cost::trace(&code, 32, 100).unwrap();
    assert_eq!(f.pcs, c.pcs);
    assert_eq!(f.masked, c.masked);
}

#[test]
fn masked_alu_costs_about_the_same_as_jumping_over_it() {
    let p = params();
    // the backend leaves out jmp_exec_none over at most 4 ALU instructions: with no lane active, the jump-less body
    // must not cost much more than the jump over it (measured: the same within ~1 ns per instruction)
    for cls in ["iadd", "fmadd32", "shuffle"] {
        for k in [1, 2, 4] {
            let (empty, skip) = (per_trip(cls, k, "empty", &p), per_trip(cls, k, "skip", &p));
            assert!(empty <= skip + 1.5 * k as f64, "{} x{}: empty {:.1} vs skip {:.1} cycles per trip", cls, k, empty, skip);
        }
        // and K = 32 masked instructions cost at most ~1-2 cycles each over the skipped body (measured 0-2 ns)
        let (empty, skip, active) = (per_trip(cls, 32, "empty", &p), per_trip(cls, 32, "skip", &p), per_trip(cls, 32, "active", &p));
        assert!(empty - skip < 2.0 * 32.0, "{}: empty {:.1} skip {:.1}", cls, empty, skip);
        assert!(empty < active, "{}: empty {:.1} active {:.1}", cls, empty, active);
    }
}

#[test]
fn short_if_body_without_the_jump_costs_about_the_same_as_with_it() {
    // a loop trip: if_icmp (no lane passes), [jmp_exec_none over the body], up to 4 dependent ALU ops, pop_exec
    // (the bodies the backend no longer jumps over), on 1 and 4 SIMD groups
    let p = params();
    let cost_of = |cls: &str, k: usize, mode: &str, groups: usize, use_mask: bool| {
        let run = |trips: u32| {
            let code = probe(cls, k, mode, trips);
            let ins = cost::decode(&code, &p).unwrap();
            let tr = cost::trace(&code, 32, 100_000).unwrap();
            if use_mask {
                cost::simulate_trace(&ins, &tr, &p, groups, false).unwrap().0
            } else {
                cost::simulate(&ins, &tr.pcs, &p, groups, false).unwrap().0
            }
        };
        (run(120) - run(20)) / 100.0
    };
    for cls in ["iadd", "fmadd32"] {
        for groups in [1, 4] {
            for k in [1, 2, 3, 4] {
                let (jump, no_jump) = (cost_of(cls, k, "skip", groups, true), cost_of(cls, k, "empty", groups, true));
                // charging the masked instructions their full latency (the trace without its mask list)
                let full_latency = cost_of(cls, k, "empty", groups, false);
                println!("{} x{} {} groups: jump {:.1}  no jump {:.1}  (full latency {:.1}) cycles per trip", cls, k, groups, jump, no_jump, full_latency);
                assert!(no_jump <= jump + 1.0, "{} x{} {} groups: no jump {:.1} vs jump {:.1}", cls, k, groups, no_jump, jump);
                assert!(no_jump <= full_latency, "{} x{} {} groups: {:.1} vs full latency {:.1}", cls, k, groups, no_jump, full_latency);
                // one issue cycle per masked instruction over the empty if
                if k > 1 {
                    let fewer = cost_of(cls, k - 1, "empty", groups, true);
                    assert!((no_jump - fewer - 1.0).abs() < 0.6, "{} x{} {} groups: {:.1} vs {:.1}", cls, k, groups, no_jump, fewer);
                }
            }
        }
    }
}

#[test]
fn masked_memory_keeps_part_of_its_cost() {
    let p = params();
    let ns = 1.398; // cycles per ns at the top clock
    for (cls, lo, hi) in [("load", 8.0, 14.0), ("tgload", 10.0, 16.0), ("store", 8.0, 14.0)] {
        let (empty, skip, active) = (per_trip(cls, 32, "empty", &p), per_trip(cls, 32, "skip", &p), per_trip(cls, 32, "active", &p));
        let extra = (empty - skip) / 32.0 / ns;
        assert!(extra > lo && extra < hi, "{}: masked {:.1} ns per instruction over the skipped body", cls, extra);
        if cls != "store" {
            assert!(empty < active, "{}: empty {:.1} active {:.1}", cls, empty, active);
        }
    }
}

#[test]
fn unmasked_programs_cost_the_same_with_or_without_the_mask_list() {
    let p = params();
    for (cls, mode) in [("iadd", "active"), ("load", "active"), ("store", "skip"), ("tgload", "base")] {
        let code = probe(cls, 8, mode, 20);
        let ins = cost::decode(&code, &p).unwrap();
        let tr = cost::trace(&code, 32, 100_000).unwrap();
        let plain = cost::simulate(&ins, &tr.pcs, &p, 4, false).unwrap().0;
        assert_eq!(cost::simulate_trace(&ins, &tr, &p, 4, false).unwrap().0, plain, "{} {}", cls, mode);
    }
}

/// `cargo test --release --test empty_mask -- --ignored --nocapture`: the model's numbers for the B31 table
/// (ns per instruction over the base loop at the top clock, K = 32).
#[test]
#[ignore]
fn print_b31_table() {
    let p = params();
    let base = per_trip("iadd", 32, "base", &p);
    println!("base {:.1} cycles per trip", base);
    for cls in ["iadd", "fmadd32", "load", "store", "tgload", "shuffle"] {
        let row: Vec<String> = ["empty", "skip", "active"].iter()
            .map(|m| format!("{} {:6.2}", m, (per_trip(cls, 32, m, &p) - base) / 32.0 / 1.398)).collect();
        println!("{:8} {}", cls, row.join("  "));
    }
}
