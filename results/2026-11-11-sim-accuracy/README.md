# 2026-11-11 simulator accuracy, round 5

Branch `claude/accuracy-2026-11-11` (based on `claude/sim-accuracy-2026-11-10` 9bd534d, which is not merged into main
yet). Before = 9bd534d defaults (the 11-10 round-3/4 eval `fin1`), after = this branch's defaults (`eval/after`).

## Adopted (defaults)

1. **Far-shared lines in the footprint estimate** (`dispatch.rs`, `far`; `AGX_FAR_SHARE=0` off): lines that the
   workgroup half the grid away along a dimension also reads count once per that dimension's extent. llama.cpp
   soft_max (mask read by all 14 heads): footprint 28 -> 15 MiB, loads no longer at DRAM: +17 -> +9%.
2. **GPU-wide 6.5 MB L2 on by default** (`AGX_L2_GPU`, round-4 experiment) with **per-wave busy cores for placed
   grids** (`AGX_PLACED_BUSY`, now default on) and **placed grids' waves simulated unchained** (`AGX_PLACED_CHAIN=1`
   restores chaining): the second-round workgroups of llama.cpp's MMQ stream-k grids (4 cores busy) no longer pay 34
   cores' L2 pressure. MMQ Q5_K -22 -> -12%, Q5_1 -7 -> +1%, Q6_K -5 -> +2%; heartwall +5 -> +6%.

## Error per set (|sim / hw - 1|; `eval/table-*.md` per launch)

| set | launches | median before -> after | p90 | mean | within 10% |
|---|---|---|---|---|---|
| llama.cpp pp512 (-fa 0) | 20 | 8 -> 8% | 27 -> 27% | 11.6 -> 10.3% | 12 -> 13 |
| llama.cpp tg | 10 | 9 -> 9% | 19 -> 19% | 9.2 -> 9.2% | 5 -> 5 |
| llama.cpp flash attention | 4 | 9 -> 9% | 14 -> 14% | 8.7 -> 8.7% | 2 -> 2 |
| tinygrad | 5 | 13 -> 14% | 21 -> 21% | 13.3 -> 13.6% | 2 -> 2 |
| held-out (Rodinia + Halide, no Cycles) | 160 | 5 -> 5% | 18 -> 18% | 7.7 -> 7.6% | 112 -> 114 |
| - Rodinia | 46 | 8 -> 8% | 20 -> 20% | 9.2 -> 8.7% | 33 -> 34 |
| - Halide | 114 | 5 -> 5% | 18 -> 18% | 7.0 -> 7.1% | 86 -> 86 |

(tinygrad r_32_16 -13 -> -14% from the far-share footprint; the Halide mean +0.1 from laplacian f31 +15 -> +18,
blurz 0 -> +6 vs outGPyramid3_o +18 -> -8, both from the L2 capacity.)

## Hardware findings (playground novel-results/2026-11-11-store-time-not-overlapped)

- rms_norm pp (hw/500, 505): truncated copies match the model up to the store (12.6 vs 12.1 us at 14 workgroups per
  core); the store adds +5.0 us = all of its 1.9 MB at 365 GB/s; the fences after it cost nothing.
- No-store copies (hw/510, 520): the store time is fully added in rms_norm, cpy_scalar, convert, agx_sgemm_tile64,
  ~1/2 in soft_max, ~1/4 in unary_gated / rope (DRAM streaming) and in Halide stencil_chain stage_5.
- soft_max cut sweep (hw/515): its two dependent row loads cost ~0.34 us per workgroup (model 0.62 after the
  footprint fix).

## Rejected experiments (kept, default off)

| knob | effect |
|---|---|
| AGX_STORE_DRAIN=8 AGX_STORE_DRAIN_ISSUE=1 (store lines stall the core's issue at its share of 365 GB/s, DRAM lines excepted) | pp median 8 -> 4.5%, p90 27 -> 11% (rms_norm -27 -> -3%) but held-out mean 7.6 -> 17.2% (stencil_chain stages -2 -> +48%, nlmeans sum_s1 +74%) — `eval/e2-far-st` |
| AGX_STORE_ADD=1 (time += min(core, store)) | rms_norm ok, unary_gated +5 -> +35%, soft_max +17 -> +27% |
| AGX_STORE_DRAIN with _BLOCK / _PIPE / _SLOT | stores still overlap in the model (rms_norm -27 -> -13%, cpy_scalar unchanged) |
| AGX_SEQ_PREFETCH=K (sequential lines get L2 latency) | probe_l2_share 512 KB 813 -> 326 us (hw 341-361), soft_max +1%, but hgemm -11 -> -31%, sgemm -23 -> -34%, MMQ fixup -6 -> -52% |
| AGX_CHAIN_SHARERS=1 (chained workgroups each with their own busy cores) | Q6_K 6101 (+10%), Q5_K 3222 (-13%); unchaining was better |
| AGX_L2_GPU=8 MB (chained, round-4 rerun) | Q5_K -10%, Q6_K +21% |

## Files

`eval/before`, `eval/after`, `eval/e1-far` (far only), `eval/e2-far-st` (+ store stall), `eval/e3-far-l2` (= after);
`hw/` GPU job scripts and logs; `evalsets.sh`, `runsub.sh`, `storetest.sh`, `mkvar.py`, `dis.py` (round tools, see
/scratch/agx-sim-notes.md); `rmscut-sim.txt` (model at the rms_norm cuts).
