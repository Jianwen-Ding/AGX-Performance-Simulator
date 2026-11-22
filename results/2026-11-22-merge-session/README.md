# 2026-11-22: model experiments scored on the held-out set (defaults unchanged)

Branch `claude/merge-all-2026-11-22`. Held-out set: 159 warm launches (Rodinia, Halide apps, cfd and the others,
without Cycles; `capcheck --hw-from`, runtime profile). The tables are in `eval/`.

| setting | held-out mean abs error | within 10% | other sets | status |
|---|---|---|---|---|
| defaults | 7.6% | 119 | llama tg (10 launches) 9.4% | - |
| `AGX_RESIDENT_SHARE=1` (new: SLC-resident lines do not get the DRAM share) | 8.0% | 119 | gPyramid4 sweep 31 -> 19%, outGPyramid1 +25 -> +51% | rejected |
| `AGX_DRAM_SHARE=0` | 11.5% | 113 | gPyramid4 sweep 31 -> 12% | rejected |
| `AGX_FETCH_OVERLAP=0.6` (default 0.3) | 7.9% | 115 | llama tg 9.4 -> 7.4% | rejected |
| `AGX_FETCH_COLD_NS=60` (default 70; fits the beyond-i-cache probe, results/2026-11-22-fetch-overlap) | 7.6% | 118 | likelihood +6 -> +1%, myocyte (2.8 MB straight-line code) 0 -> -12% | not adopted |
| `AGX_FETCH_OVERLAP=1.0` | not run | | llama tg 10.7% | - |
| `AGX_L2_LRU=4 AGX_L2_CORE=600000 AGX_L2_EVICT_SLC=1` (round 8) | not finished | | | still open |

The full-set runs of the round-8 L2 knobs and of fetch overlap 1.0 were stopped when the host ran low on memory
(three capcheck sets in parallel at J=2-3, two simulation threads each, alongside the llama pp512 MMQ kernels). Run them
one set at a time.

## Fetch overlap (llama token generation)

The Q8_0 mat-vec kernel (-19%) is fetch-bound: 79 cold code lines give 9.4 us of fetch against 4.9 us of core work
(`AGX_DEBUG_TIME=1`). `scripts/fetch_overlap_need.py CAPDIR HWFILE` solves, per launch, for the overlap factor that
would reproduce the hardware time. Single-wave kernels need 0.7-1.0: Q8_0 0.97, rope 0.72 / 1.08, rms_norm 0.76.
Multi-wave mat-vec kernels need 0.43-1.15. The tiny kernels (quantize_q8_1, k_set_rows) need a negative factor, so
their fixed costs are too high. A single global factor does not help: 0.6 improves tg and worsens the held-out set.
The factor probably depends on the structure, for example on whether all groups of a wave stall on the same cold line.
A hardware probe that varies the number of groups running the same cold straight-line code would settle it.
