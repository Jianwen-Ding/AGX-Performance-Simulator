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

## Model: store drain by bytes (default since 2026-11-25; `AGX_STORE_DRAIN=0` turns it off)

The 2026-11-11 drain experiment (stores stall the core's issue at the drain rate) fitted llama's kernels, but it was
rejected because the Halide stencil stages went to +48%. That came from the charge, not from overlap:

- Stencil stores are 64 B over 2-3 partial 64 B segments, and the stores of a thread rewrite segments the core already
  stored. The drain charged segments, about 5x the bytes.
- New: each trace records the bytes the active lanes access (`Trace::mem_bytes`, passed to the cost model in
  `MemModel::mem_bytes`). The drain charges min(segments, bytes / 64), and only segments this core has not stored
  before (`MemState::drained`). An in-place accumulation (non_local_means_sum_s1) then costs nothing extra.
- The probe's store costs with the model are 11.7 / 10.0 / 9.4 / 10.7 / 11.0 us (i32) and 6.5 / 5.1 us (i16).

| Set | Before | Byte-based drain (now the default) |
|---|---|---|
| held-out (160) | 7.5%, median 5.5, w10 119 | **6.8%, median 5, w10 125** (srad -20 → -8, backprop -18 → -9, laplacian f2 -14 → +3, f36 -24 → -11; mergeSortFirst 0 → +6) |
| llama pp512 (20) | 10.3%, median 8, w10 15 | **8.7%, median 7, w10 16** (rms_norm -27 → -7, sgemm -23 → -18, cpy -10 → -5, k_bin_bcast -48 → -37; soft_max +9 → +16, rope f32 0 → +8; MMQ unchanged) |
| llama fa (4) | 6.8% | **5.5%** |
| llama tg (10) | 8.2% | 8.2% |
| tinygrad (5) | 13.0% | 13.2% |
| core scaling grid / chains / idle / dests | 3.1 / 3.3 / 7.1 / 5.8% | 3.0 / 3.3 / 7.1 / 5.8% (idle p90 0.41 → 0.28) |
| `verify-memory-model`, `verify-dispatch`, `verify-cost-model`, `verify-jump-fetch` | pass | pass (dispatch 4.8%, cost model identical) |

### The two launches that got worse: their stores are right, the rest is too slow

The stores were replaced by `iadd r9, r9, 0` in the code (same size), as in the 2026-11-11 hardware variants, and the
model was run on the result:

| Kernel | No-store, hw | No-store, model | Store part, hw | Store part, model (drain) | Store part, model (before) |
|---|---|---|---|---|---|
| rope_neox f32 | 28.3 us | 32.3 us (+14%) | +5.2 | +3.9 | +1.3 |
| soft_max_f32 | 403.1 us | 461.2 us (+14%) | +22 | +30 | +2.5 |

The old model's rope 0% and soft_max +9% came from under-charged stores offsetting compute that is 14% too slow.
For soft_max, the 2026-11-11 cut sweep had already found the cause: its two dependent row loads cost 0.62 us per
workgroup in the model and 0.34 on hardware. An extra rule that hides store time behind other groups' DRAM loads
(`AGX_STORE_HIDE_DRAM`, tried and reverted) changed nothing.

### soft_max's slow part is not its serial loads (hardware probe, [rowload-hw.txt](rowload-hw.txt))

`probe_hot_lines.py` with SERIAL=1 NOSTORE=1 (load, wait, load, wait, add) has the shape of soft_max's two dependent
row loads. `WG=512 REGS=188` gives one 512-thread workgroup per core at a time, as in soft_max. Times in us, two
repeats:

| Workgroups | Distinct second load: hw | model | Broadcast second load: hw | model |
|---|---|---|---|---|
| 448 | 11.1 | 9.9 | 8.2-8.3 | 6.8 |
| 1792 (7 MB, SLC) | 33.2 | 32.7 | 22.5 | 20.8 |
| 7168 (29 MB, end of the SLC transition) | 127.5 / 137.6 | 155.4 | 79.8-79.9 | 74.3 |

At soft_max's occupancy with SLC-resident data the model matches. So the +14% of soft_max without its stores comes
from somewhere else. The register-address load form (LOADREG=1, soft_max's `device_load 0, i32, x, r19, r10_r11,
r5, unsigned`, verified byte for byte against its capture) makes no difference: hw 11.2 / 33.6 us against 11.1 /
33.2, model 10.1 / 33.4. What remains is its threadgroup reductions with barriers, at 16 groups per core. That
contradicts the 2026-11-11 cut sweep ("the rest matches"), so the next step is a cut-for-cut model comparison. At 128- and
512-thread workgroups with many per core, distinct loads come out +6..+9% and broadcast loads up to -29% (WG 512,
40.3 vs 28.8 us).
