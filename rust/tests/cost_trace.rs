use agx_sim::{cost::{self, GroupCtx, PreparedTrace, Trace}, kernels::{e, e2, get_sr, mov}, par};

fn context() -> GroupCtx {
    GroupCtx { wg_size: [32, 1, 1], wg_id: [0; 3], group: 0, core: 0, slot: 0,
        uniforms: vec![], launch: Default::default() }
}

fn same(a: &Trace, b: &Trace) {
    assert_eq!(a.pcs, b.pcs);
    assert_eq!(a.mem, b.mem);
    assert_eq!(a.skipped, b.skipped);
    assert_eq!(a.warnings, b.warnings);
}

#[test]
fn cost_trace_counts_barriers_and_rejects_invalid_fetches() {
    for code in [vec![0x78, 0, 0x78, 9, 0x88, 0], vec![0x68, 0, 0x78, 0, 0x88, 0]] {
        for run in [cost::trace, cost::trace_reference] {
            assert_eq!(run(&code, 32, 3).unwrap().pcs, vec![0, 2, 4]);
            for limit in [0, 1, 2] { assert!(run(&code, 32, limit).unwrap_err().contains("step limit")); }
        }
    }
    // An all-barrier program used to bypass the check entirely.
    assert!(cost::trace(&[0x78, 0, 0x78, 9], 32, 1).unwrap_err().contains("step limit"));
    for n in 1..4 {
        assert!(cost::trace(&get_sr(80, 2)[..n], 32, 10).unwrap_err().contains("truncated"));
    }
    for text in ["jmp_exec_any pc-2", "jmp_exec_any pc+100", "jmp_exec_any pc+1"] {
        assert!(cost::trace(&e(text), 32, 10).unwrap_err().contains("jump at"));
    }
    let mut code = e("jmp_exec_none pc+100");
    code.extend([0x88, 0, 0xff]);
    assert_eq!(cost::trace(&code, 32, 2).unwrap().pcs.len(), 2);
    let mut ctx = context();
    ctx.uniforms = vec![(usize::MAX, 0)];
    assert!(PreparedTrace::new(&[0x88, 0]).trace_group(&ctx, 1).unwrap_err().contains("uniform register"));
}

#[test]
fn shared_cost_trace_keeps_geometry_uniforms_masks_and_taint_independent() {
    let mut code = mov(3, 0);
    code.extend(get_sr(0, 7));
    code.extend(e("iadd r4, u4, 0"));
    code.extend(e("iadd r4, r4, r7"));
    code.extend(e2("if_fcmp r0l, eq, 0.0, 0.0, 1", "push_exec r0l, 1"));
    let mut lp = e("iadd r3, 1, r3.sx");
    lp.extend(e("while_icmp r0l, ult, r3, r4, 1"));
    lp.extend(e(&format!("jmp_exec_any pc-{}", lp.len())));
    code.extend(lp);
    code.extend(e("pop_exec r0l, 1"));
    code.extend(get_sr(80, 2));
    // The approximate interpreter supports register bases, not the uniform-base load form.
    code.extend(e("iadd r24, u8, 0"));
    code.extend(e("iadd r25, u9, 0"));
    code.extend(e("device_load 0, i32, x, r6, r24_r25, r2, unsigned"));
    code.extend(e("if_icmp r0l, ueq, r6, 0, 1"));
    code.extend(e("pop_exec r0l, 1"));
    code.extend([0x88, 0, 0xff]); // never decode the unreachable malformed suffix
    let prepared = PreparedTrace::new(&code);
    let mut contexts = vec![];
    for x in 0..16 {
        let base = if x % 2 == 0 { cost::TABLE_VA as u64 } else { 0 }; // alternate valid/skipped loads
        for group in 0..2 {
            let mut ctx = context();
            ctx.wg_size = [48, 1, 1];
            ctx.wg_id = [x, 0, 0];
            ctx.group = group;
            ctx.uniforms = vec![(4, 2 + x % 3), (8, base as u32), (9, (base >> 32) as u32)];
            contexts.push(ctx);
        }
    }
    par::set_local_threads(4);
    for _ in 0..2 { // warm shared decodes must not retain execution state
        let traces = par::map(&contexts, |ctx| prepared.trace_group(ctx, 1000).unwrap());
        for (ctx, tr) in contexts.iter().zip(&traces) {
            same(tr, &cost::trace_group(&code, ctx, 1000).unwrap());
            assert_eq!(tr.warnings.is_empty(), ctx.wg_id[0] % 2 == 0);
            assert_eq!(tr.skipped.is_empty(), ctx.wg_id[0] % 2 == 0);
            let base = if ctx.wg_id[0] % 2 == 0 { cost::TABLE_VA as u64 } else { 0 };
            let first = ctx.wg_id[0] * 48 + ctx.group * 32;
            let mut segs: Vec<_> = (0..ctx.lanes()).map(|t| (base + 4 * (first as u64 + t as u64)) / 64).collect();
            segs.sort_unstable(); segs.dedup();
            assert_eq!(tr.mem.len(), 1);
            assert_eq!(tr.mem[0].1, segs);
        }
        assert_ne!(traces[0].pcs.len(), traces[2].pcs.len());
    }
    par::set_local_threads(0);
    // A failed call cannot poison the shared decode for later calls.
    assert!(prepared.trace_group(&contexts[0], 1).is_err());
    same(&prepared.trace_group(&contexts[0], 1000).unwrap(), &cost::trace_group(&code, &contexts[0], 1000).unwrap());
}
