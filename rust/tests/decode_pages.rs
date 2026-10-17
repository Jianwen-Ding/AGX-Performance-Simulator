//! Synthetic cache-boundary regressions, not new hardware recordings.
use agx_sim::{cost::{self, GroupCtx, PreparedTrace}, func::{self, Dispatch, Memory, PreparedDispatch}, kernels::{e, get_sr, store}};

#[test]
fn jumps_and_cross_boundary_instructions_keep_exact_paths_and_data() {
    for target in [8190, 8192, 8702, 8704, 65534] {
        let mut code = e(&format!("jmp_exec_any pc+{target}"));
        code.resize(target, 0xff); // unreachable bytes must not be decoded
        let mut pcs = vec![0, target];
        code.extend(get_sr(80, 2));
        pcs.push(code.len());
        code.extend(e("iadd r3, r2, 1"));
        pcs.push(code.len());
        code.extend(store(3, 6, 2));
        pcs.push(code.len());
        code.extend([0x88, 0, 0xff]); // odd malformed suffix remains unreachable
        let d = Dispatch { code, grid: [64, 1, 1], wg: [32, 1, 1], shared_bytes: 0,
            uniforms: vec![(6, 0x10000), (7, 0)], num_registers: 8,
            max_steps: 5, placement: 0, launch: Default::default() };
        let mut mem = Memory::default();
        mem.map(0x10000, vec![0xcd; 64 * 4]);
        let stats = func::run(&d, &mut mem).unwrap();
        assert_eq!(stats.instructions, 10);
        assert_eq!(mem.words(0x10000, 64).unwrap(), (1..=64).collect::<Vec<_>>());
        let prepared = PreparedDispatch::new(&d).unwrap();
        let cost = PreparedTrace::new(&d.code);
        for _ in 0..2 {
            let traces = prepared.trace_workgroups(&mem, &[[0; 3], [1, 0, 0]]).unwrap();
            for (x, wg) in traces.iter().enumerate() {
                assert_eq!(wg[0].pcs, pcs);
                let ctx = GroupCtx { wg_size: d.wg, wg_id: [x as u32, 0, 0], group: 0, core: 0, slot: 0,
                    uniforms: d.uniforms.clone(), launch: Default::default() };
                let tr = cost.trace_group(&ctx, 5).unwrap();
                assert_eq!(tr.pcs, pcs);
                assert_eq!(tr.mem, wg[0].mem);
                let expected = vec![(3, vec![(0x10000 + x as u64 * 128) / 64, (0x10040 + x as u64 * 128) / 64])];
                assert_eq!(tr.mem, expected);
            }
        }
    }
}

#[test]
fn failed_tail_decodes_do_not_alias_neighboring_or_prefix_cells() {
    let target = 16384;
    let mut code = e(&format!("jmp_exec_any pc+{target}"));
    code.resize(target, 0xff);
    code.extend(&get_sr(80, 2)[..3]); // truncated instruction in a lazy page
    for run in [cost::trace, cost::trace_reference] {
        assert!(run(&code, 32, 10).unwrap_err().contains("truncated"));
    }
    let prepared = PreparedTrace::new(&code);
    let ctx = GroupCtx { wg_size: [32, 1, 1], wg_id: [0; 3], group: 0, core: 0, slot: 0,
        uniforms: vec![], launch: Default::default() };
    for _ in 0..2 { assert!(prepared.trace_group(&ctx, 10).unwrap_err().contains("truncated")); }
    // No active lane takes the jump; prefix decode must remain intact after errors.
    let empty = GroupCtx { wg_size: [1, 1, 1], group: 1, ..ctx };
    assert!(prepared.trace_group(&empty, 1).unwrap_err().contains("step limit"));
}
