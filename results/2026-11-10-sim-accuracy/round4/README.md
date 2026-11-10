# 2026-11-10 round 4: GPU-wide L2 capacity (MMQ busy-core slowdown), stack probes, soft_max / rms_norm occupancy

Hardware finding: apple-documentation-playground novel-results/2026-11-10-l2-gpu-capacity (the L2 is one GPU-wide
capacity of ~6-7 MB; per-core working sets of 256 KB slow beyond ~24 busy cores; the cores' stacks, 6 MB apart,
also alias at 24). Raw logs and job scripts in `hw/` (queue numbers 300-335). Probes: `probes/*.ptx` (hand-written
PTX, compiled by zoc, captured with agx_bench_kernel + ZLUDA_AGX_CAPTURE, replayed by `gsweep.py` at smaller grids).
`fitprobe.sh TD TS`: the probes' simulated times for an AGX_L2_GPU setting.

## Model changes (all experiments, default off)

- `AGX_L2_GPU=BYTES` (+ `AGX_L2_GPU_STACK` for stack lines): the busy cores' L2 working sets share a GPU-wide
  capacity T. A core's working set W = distinct lines of its last 4 x (L2 lines) touches; an L2 hit misses (SLC)
  with probability 1 - T / (cores using distinct data x W), the same lines each time. cost.rs `WorkingSet`.
- `AGX_PLACED_BUSY=1`: a placed core's wave k shares the memory system with the cores that run a k-th workgroup (38
  one-resident workgroups: 34 cores, then 4), not with min(38, n).
- `AGX_L2_TOTAL=BYTES`: the per-core L2 LRU sized T / busy cores (a cliff: an LRU over a cyclic working set loses
  all of it at once; stack probe 709 -> 1453 us where hw goes 737 -> 1045 -> 1184; not useful).
- `gsweep.py`: NOSIM=1 (hw only), REGS= (override regs.txt), HWBIN= (agx_run_hw build).

RESULTS_PLACEHOLDER
