//! Executor representation/input guards, not new hardware observations.
use agx_sim::{func::{self, Dispatch, Memory, PreparedDispatch}, kernels::{get_sr, store}};

fn dispatch(grid: [u32; 3], wg: [u32; 3]) -> Dispatch {
    Dispatch { code: vec![0x88, 0], grid, wg, shared_bytes: 0, uniforms: vec![],
        num_registers: 8, max_steps: 1, placement: 17, launch: Default::default() }
}

#[test]
fn invalid_coordinates_fail_before_execution_including_empty_dispatches() {
    let mut d = dispatch([9, 5, 3], [4, 2, 2]);
    d.code = vec![0xff]; // rejection must precede decoding
    let prepared = PreparedDispatch::new(&d).unwrap();
    for wg in [[3, 0, 0], [0, 3, 0], [0, 0, 2], [u32::MAX; 3]] {
        assert!(prepared.trace_workgroup(&Memory::default(), wg).unwrap_err().contains("outside dispatch"));
        // Invalid member at the end must be found before decoding the first member.
        assert!(prepared.trace_workgroups(&Memory::default(), &[[0; 3], wg]).unwrap_err().contains("outside dispatch"));
    }
    for axis in 0..3 {
        let mut grid = [u32::MAX; 3]; grid[axis] = 0;
        let d = dispatch(grid, [1; 3]);
        let prepared = PreparedDispatch::new(&d).unwrap();
        assert!(prepared.trace_workgroups(&Memory::default(), &[]).unwrap().is_empty());
        assert!(prepared.trace_workgroup(&Memory::default(), [0; 3]).unwrap_err().contains("outside dispatch"));
        let stats = func::run(&d, &mut Memory::default()).unwrap();
        assert_eq!((stats.groups, stats.instructions), (0, 0));
    }
}

#[test]
fn workgroup_capacity_is_checked_without_wrapping_and_boundary_tags_are_exact() {
    // Exactly 2^48 workgroups fit in the existing 48+16-bit writer tag.
    let mut d = dispatch([65536, 65536, 65536], [1; 3]);
    let p = PreparedDispatch::new(&d).unwrap();
    assert_eq!(p.trace_workgroup(&Memory::default(), [65535; 3]).unwrap()[0].pcs, vec![0]);
    // Placement and execution also accept the highest representable index.
    d.code = get_sr(20, 4); d.code.extend(store(4, 6, 2)); d.code.extend([0x88, 0]);
    d.uniforms = vec![(6, 0x10000), (7, 0)]; d.max_steps = 3;
    let mut mem = Memory::default(); mem.map(0x10000, vec![0; 4]);
    let tr = PreparedDispatch::new(&d).unwrap().trace_workgroup(&mem, [65535; 3]).unwrap();
    assert_eq!(tr[0].pcs.len(), 3);
    for grid in [[65537, 65536, 65536], [u32::MAX; 3]] {
        let d = dispatch(grid, [1; 3]);
        assert!(PreparedDispatch::new(&d).err().unwrap().contains("48-bit"));
        let mut mem = mem.clone();
        assert!(func::run(&d, &mut mem).unwrap_err().contains("48-bit"));
        assert_eq!(mem.read(0x10000, 4).unwrap(), 0);
    }
}

#[test]
fn simd_writer_tags_do_not_merge_across_barrier_resumes() {
    let mut d = dispatch([128, 1, 1], [64, 1, 1]);
    d.code = get_sr(53, 4); d.code.extend(store(4, 6, 2));
    d.code.extend([0x78, 0]); d.code.extend(store(4, 6, 2)); d.code.extend([0x88, 0]);
    d.uniforms = vec![(6, 0x10000), (7, 0)]; d.max_steps = 5;
    let mut mem = Memory::default(); mem.map(0x10000, vec![0; 4]); mem.log = Some(Default::default());
    let stats = func::run(&d, &mut mem).unwrap();
    assert_eq!((stats.groups, stats.instructions), (4, 20));
    assert_eq!(mem.log.as_ref().unwrap()[&0x10000], vec![(0, 0), (1, 1), (65536, 0), (65537, 1)]);
}

#[test]
fn partial_three_dimensional_edges_and_racing_writer_tags_stay_distinct() {
    let mut d = dispatch([3, 3, 3], [2, 2, 2]);
    d.code = get_sr(0, 4); d.code.extend(store(4, 6, 2)); d.code.extend([0x78, 0]);
    d.code.extend(get_sr(1, 4)); d.code.extend(store(4, 6, 2)); d.code.extend([0x88, 0]);
    d.uniforms = vec![(6, 0x10000), (7, 0)]; d.max_steps = 6;
    let mut mem = Memory::default(); mem.map(0x10000, vec![0; 4]); mem.log = Some(Default::default());
    let stats = func::run(&d, &mut mem).unwrap();
    assert_eq!((stats.groups, stats.instructions), (8, 48));
    let writers = &mem.log.as_ref().unwrap()[&0x10000];
    assert_eq!(writers, &(0..8).map(|w| (w << 16, ((w / 2) % 2) as u8)).collect::<Vec<_>>());
    let traces = PreparedDispatch::new(&d).unwrap().trace_workgroups(&mem, &[[0, 0, 0], [1, 1, 1]]).unwrap();
    for tr in traces { assert_eq!(tr.len(), 1); assert_eq!(tr[0].pcs.len(), 6); }
}
