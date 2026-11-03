# Launch overhead, the instruction cache and refill, on 37 captured launches

Status: model comparison against stored warm hardware times (`/scratch/perf-goal/gbcap-check.txt`, `/scratch/perf-goal/sim/capcheck-warm.txt`:
29 Geekbench and 8 llama launches, one M2 Max), `scripts/eval_captures.py`, no new GPU runs. The six Blender launches of
`2026-11-03-real-launch-prediction` are not in this set, and their captures are gone: recapture them.

| Model | Median error | Mean error | Within 25% | Signed median |
| --- | --- | --- | --- | --- |
| fetch cost on every launch, 5.6 us launch | 18% | 28% | 22 | +18% |
| no fetch cost, 5.6 us launch | 17% | 21% | 25 | +11% |
| no fetch cost, `--launch-profile zluda` (3.0 us) | 10% | 15% | 30 | -3% |
| same, `AGX_REFILL=1` | 10% | 14% | 33 | -7% |

## Findings

1. **Launch overhead through ZLUDA is about 3.0 us, not 5.6.** Fitted over these launches (0.5 to 5.6 us tried). An older `--launch-profile zluda` was in
   the old evaluation scripts but not in this code; it is back. It is fitted to these launches, so they do not test it independently.
2. **The fetch cost of `2026-11-03-real-launch-prediction` is wrong for small code.** It made this set worse: `mul_mat_vec_q` +8% to +54%, `raw_process` +11% to +59%.
   This code is at most 12.8 KB and repeats, so it stays cached. `AGX_ICACHE_BYTES` drops the charge when the touched code fits. It is off by default,
   because the capacity is not known: free at under 13 KB (from this set), not free from 64 KB (probe). **To do:** `SIZES="4096 8192 16384 24576 32768 49152 65536"
   scripts/probe_fetch_scaling.py`, with power, then set the default, and check the Blender launches keep their fetch cost.
3. **Waves and refill seem to bracket the hardware.** Waves wait for each batch's slowest workgroup, refill (`AGX_REFILL=1`) starts one as soon as registers free. Refill
   corrects `blur_gray` +38% to +10% and `image_rotate` +35% to -5%, but overshoots `convolve_horizontal` -7% to -22% and `initialize_dye` +6% to -19%. So the hardware probably
   refills with a cost not modelled. **Unexplained:** several image kernels land on exactly 970.2 us under refill. Explain that before trusting it.
4. **Occupancy gains are overpredicted.** One `intersect_closest` launch, 16512 threads: 128 registers, block 384: 76.4 us; 80, 384: 76.4; 128, 192: 64.0; 80, 192: 53.7 (-30%). Renders got
   about +1-4% from block size alone. One launch against a whole render, a 128-register binary, and no hardware time for it: a lead, not a measurement.

## Not modelled

Spills from a lower cap unless the kernel is compiled at that cap; the whole-render mix of kernels and launch sizes. **Test to do:** capture launches compiled at caps 128, 80, 56
and compare with the measured curve (samples/min, classroom / junkshop / monster: 128: 204/247/290, 96: 217/276/325, 80: 233/283/349, 72: 234/265/355, 64: 234/264/354,
56: 226/240/333, 48: 201/227/293). Does the model find the peak near 72-80 and the fall at 48?
