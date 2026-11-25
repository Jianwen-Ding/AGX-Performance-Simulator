# Store time is additive and paid per byte written (2026-11-25, hardware + model experiment)

## Hardware ([hw-probe.txt](hw-probe.txt))

`scripts/probe_hot_lines.py`, BCAST=0, ITER=1, N=8192 workgroups of 128 threads, one load pair + add + store per
thread, warm (b2b 5000), two repeats agreeing within 0.1 us. The store's cost is the time with the store minus
NOSTORE=1. PRE / POST add n independent iadds before the loads / after the store.

| Store cost (us) | PRE 0 | PRE 256 | PRE 1024 | POST 256 | POST 1024 |
|---|---|---|---|---|---|
| i32, 4.2 MB | 10.1 | 8.1 | 8.0 | 5.7 | 6.7 |
| i16, 2.1 MB (`STORE16=1`, Halide stencil's store form) | 5.2 | 4.4 | 4.6 | | |

- Compute does not hide store time: the kernel at PRE 1024 runs 170 us, compute-bound, and still pays 8 us for 4 MB
  of stores (~450-520 GB/s).
- The cost is per byte: half the bytes cost half the time, whatever the width.

## Model: `AGX_STORE_DRAIN=8 AGX_STORE_DRAIN_ISSUE=1 AGX_STORE_DRAIN_GBPS=450` (experiment, off by default)

The 2026-11-11 drain experiment (stores stall the core's issue at the drain rate) fitted llama's kernels, but it was
rejected because the Halide stencil stages went to +48%. That came from the charge, not from overlap:

- Stencil stores are 64 B over 2-3 partial 64 B segments, and the stores of a thread rewrite segments the core already
  stored. The drain charged segments, about 5x the bytes.
- New: each trace records the bytes the active lanes access (`Trace::mem_bytes`, passed to the cost model in
  `MemModel::mem_bytes`). The drain charges min(segments, bytes / 64), and only segments this core has not stored
  before (`MemState::drained`). An in-place accumulation (non_local_means_sum_s1) then costs nothing extra.
- The probe's store costs with the model are 11.7 / 10.0 / 9.4 / 10.7 / 11.0 us (i32) and 6.5 / 5.1 us (i16).

| Set | Default | Drain (bytes, fresh, 450 GB/s) |
|---|---|---|
| llama pp512 (16 of 20: the MMQ launches did not run) | 9.2% | **7.9%** (rms_norm -27 → -7, sgemm -23 → -18, cpy -10 → -5, convert -8 → -5, k_bin_bcast -48 → -37; soft_max +9 → +16, rope f32 0 → +8) |
| llama fa (4) | 6.8% | **5.5%** |
| llama tg (10) | 8.2% | 8.2% |
| tinygrad (5) | 13.0% | 13.2% |
| Halide stencil_chain (32), nlmeans, linear blur | stencil -6..+1 | stencil -2..+6; nlmeans and blur within 1-3 points |
| held-out (160) | 7.5% | not run: the evaluation was stopped by host memory pressure |

It stays off until the held-out set and the MMQ launches are scored. Rerun with `evalsets.sh OUT BIN
AGX_STORE_DRAIN=8 AGX_STORE_DRAIN_ISSUE=1 AGX_STORE_DRAIN_GBPS=450`, one set at a time.
