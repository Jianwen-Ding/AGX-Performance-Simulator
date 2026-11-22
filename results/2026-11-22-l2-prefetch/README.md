# 2026-11-22 round 8: per-core L2 share, stream prefetch (experiments, every knob default off)

Branch `claude/accuracy-2026-11-22` (off `claude/accuracy-2026-11-12`). **The defaults are unchanged** because no knob
improved the real-program sets. Every knob added here is an experiment, and its effect is documented below.

## Hardware findings (this round, `hw/`)

- **One core can use far more than 256 KB of L2.** The round-4 stack probes at 1-4 busy cores (b2b 20, noisy) give
  A (256 KB/core) 701 us, G (512 KB) ~510-620 us and D (1 MB) ~475 us. G and D are not slower than A.
- **Sequential and gap-strided streams hide miss latency.** `probe_l2_share.py` (new option `STEP`): 2 MB per core
  with every other 4 KB chunk skipped (STEP=8, 1 MB touched) takes 369 us at 1 core and 388 us at 32 cores. That is
  185 ns per dependent trip, below the DRAM latency (~400 ns). The round-5 log gives the contiguous case:
  418 / 432 us, about 590 GB/s at 32 cores, which is more than DRAM bandwidth.
  (STEP 16/32/64 shrink the touched footprint under the wrap mask to 512/256/128 KB, so they do not test prefetch.)
- **At 16-32 busy cores the stack probes are bandwidth-like.** G and H move about 300-360 GB/s GPU-wide, and the time
  grows with the number of cores. The model instead becomes latency-bound, because every trip waits on one miss.

## Knobs (cost.rs) and results (`sim/`)

| knob | probe effect | real-program effect | status |
|---|---|---|---|
| `AGX_L2_LRU=4 AGX_L2_CORE=600000` (per-core recency window 1 MB, one core's cap 600 KB of the GPU-wide L2) | stack G 986 -> 493 us at 8 cores (hw 540), H 975 -> 480 (475); l2_share 512 KB 813 -> 347 (hw 341-360) | MMQ Q8_0 +3 -> -5% (the only MMQ launch finished) | experiment |
| `AGX_L2_EVICT_SLC=1` (a line the GPU-wide L2 lost falls through to the SLC check) | stack D 575 -> 974 at 32 cores (hw 1586) | not run | experiment, use with the above |
| `AGX_STACK_EVICT_DRAM=1` + `AGX_L2_GPU_STACK=4194304` | latency-bound saturation: G ~1060 flat at 16-32 cores, A 1535 at 24 cores (hw 1045) | - | rejected |
| `AGX_WB_DIRTY=1` (DRAM charge for written lines re-fetched from DRAM) | no change (the probes' lines are SLC hits in the model) | - | rejected |
| `AGX_PREFETCH=8` (stream prefetcher: trains on two consecutive misses, the next D lines are ready at the trigger's issue time + latency) | l2_share 2 MB@1 794 -> 424 (hw 418), 2 MB@32 897 -> 469 (432), 256 KB@32 740 -> 342 (349); GEMMs unchanged; soft_max +9 -> +7% | partial eval (60 launches): pp median 8 -> 7.5; held-out (25 launches) mean 7.0 -> 8.4% (particlefilter kernel -10 -> -23%, find_index -10 -> -20%) | experiment, not adoptable |
| `AGX_PF_REGION=8` (the first miss in a 16 KB region fetches the region) | STEP=8 2 MB: 606/803 -> 348/358 (hw 369/388) | sgemm -23 -> -31%, hgemm -11 -> -31%, MMQ fixup -6 -> -49% | rejected |

The partial full evaluations are in `eval-partial/` (base = defaults, pf8 = AGX_PREFETCH=8). They were stopped when
the session ended. Note that the default model has been 2.1x slow on `probe_l2_share` 256 KB at 32 cores (740 us vs
hw 349) since the GPU-wide L2 (round 5) was turned on.

## Next

- The prefetcher is right for streams, but it speeds up kernels that are already predicted too fast (particlefilter,
  GEMMs). Find what those kernels are missing first (store time, see round 5) and then retry `AGX_PREFETCH`.
- Run the full sets for `AGX_L2_LRU=4 AGX_L2_CORE=600000 AGX_L2_EVICT_SLC=1` (MMQ Q5_K / Q6_K have 352 KB+ stacks per
  workgroup).
- The stack probes at 16-32 cores need a bandwidth regime for lines the L2 lost, not one full-latency wait per trip.
