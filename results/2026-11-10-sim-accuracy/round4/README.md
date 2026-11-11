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

## Hardware vs model

| case | hw | model before | AGX_L2_GPU=6815744 |
|---|---|---|---|
| stack probe A (256 KB/core) 16 / 24 / 32 cores | 743 / 1045 / 1184 | 709 / 709 / 713 | 709 / 709 / 1304 |
| stack probe E (128 KB/core) 32 cores | 866 | 707 | 752 |
| stack probe C (tiny) 32 cores | 760 | 708 | 751 |
| MMQ Q5_K grid 16 / 24 / 32 / 34 | 1645 / 1855 / 2109 / 2146 | 1505-1545 (flat) | 1527 / 1540 / 1848 / 1917 |
| MMQ Q5_K full (38) | 3703 | 2876 (-22%) | 3712 (+0%) |
| MMQ Q6_K grid 17 / 34 | 2693 / 3260 | 2812 / 2830 | 2811 / 3730 |
| MMQ Q6_K full (38) | 5526 | 5225 (-5%) | 7185 (+30%); with AGX_PLACED_BUSY 7015 |

Eval with AGX_L2_GPU=6815744 (`eval/ws65`, compared with round 3 `fin1`): tg, tinygrad and fa identical; pp median 8,
mean 11.6 -> 11.7, p90 27 -> 30; held-out (159; heartwall's simulation failed under host load) median 5, mean 7.7,
p90 18, 118 within 10% (round 3: 5.5 / 7.6 / 18 / 119). Not adopted, because the pp p90 and the held-out mean get
worse. Q6_K overshoots because the chained wave-0 simulation runs both workgroups of a 2-wave core under 34 cores'
L2 pressure.

Other measurements: soft_max pp occupancy is right (hw 425 / 300 / 267 us at 116 / 96 / 64 register halves, sim
497 / 333 / 293). The model charges a constant ~0.37 us too much per workgroup, and the convert prologue is not the
cause: converts are within 3% (probe_tg_alu patterns v / u). rms_norm is under at every grid (each added workgroup
per core costs ~1.0 us on hw and 0.65 in the model). Both are still open.
