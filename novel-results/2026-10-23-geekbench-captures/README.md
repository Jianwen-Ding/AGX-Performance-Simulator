# The model on Geekbench 7's GPU (CUDA) kernels, and threadgroup-load costs in compiled code

Status: measured on one device (driver-reported M2 Max G14C B1, AC power), Geekbench 7.1.0 Linux ARM Preview running
its CUDA GPU benchmark through ZLUDA's AGX driver (`zluda_agx`, ZLUDA branch `claude/geekbench-agx`), 2026-10-23.

## Method: real launches with their real memory

`ZLUDA_AGX_CAPTURE=DIR` (zluda_agx) saves a launch as it is about to run: the code, every region its arguments and
parameter block reach with the contents the launch finds, the uniforms, grid, threadgroup memory and register
allocation. One warm launch of each of 88 kernels was captured (`ZLUDA_AGX_CAPTURE_SKIP=20`).

Hardware time: the captured launch itself ran after the GPU idled for the capture, at a low clock (a captured
`convolve_vertical`: 3185 us alone), so each capture is run again with `agx_run_hw --back-to-back 30` (the launch
submitted 30 times without waiting; median firmware time of the last 15: 1571 us, the same as Geekbench's own warm
launches and `agx_bench_kernel` through the driver). Replays with zero-filled memory are meaningless for data-dependent
kernels (ray traversal, Hough voting, k-NN: -90 to -99%), and the median of other launches with the same arguments is
not the same work either (the path tracer reuses buffers across bounces).

Prediction: `agx_dispatch` on the same capture (`--map` of every region, the captured uniforms, `--regs`,
`--tg-bytes`). `capcheck.py CAPTURES PROFILE --hw` runs both; `capcheck-before.txt` / `capcheck-after.txt` are the
results before and after the wave-sampling change below. Three captures fault on replay (`fast_corners`,
`generate_rays`, `trace_bvh`: they follow pointers stored in buffers, which the capture does not follow).

## Results (85 launches)

| | median abs error | within 10% | within 25% |
| --- | --- | --- | --- |
| first and last wave simulated, the rest = the first | 16% | 32 | 50 |
| 8 waves (first, last, middle, 1, 3, 7, 15, ...), interpolated | 16% | 28 | 54 |

**Fixed: wave sampling.** Kernels whose threads past a data-dependent count exit at once have their work in the first
waves of the grid; extrapolating the first wave to all 34 waves predicted `sort_keypoints` at 17.7 ms against 0.69 ms
(+2472%), `nonmax_suppression` +973%, `compute_keypoints` +695%, `hough_scores` +102%. With dense sampling after the
first wave: -19%, +25%, -14%, +43%. The stored dispatches (`make verify-dispatch`) are unchanged. Uniform image kernels
moved by a few percent either way (`convolve_vertical` +4% -> +13%, `blur_gray` +68% -> +80%: the interior waves
simulate slower than the first).

**Open gaps** (after the change):

* Large image kernels predicted too slow: `blur_gray` +80%, `convolve_vertical_wide` +81%, `gemm_non_block_size` +69%,
  `blend` +59%, `image_rotate` +39% (thousands of workgroups per core, streaming 8-bit image reads).
* Small, short kernels predicted too fast: `gemm_non_block_size_bias` -58%, `bilinear_grayscale` -56%,
  `resize_bilinear` -55%, `depthwise_convolution_2d` -48%, `global_average_pooling_2d` -46%, `velocity_advection` -39%
  (one or two waves; a fixed per-workgroup or per-wave cost seems missing).
* Texture sampling: `miss` (path tracer; one `tex.2d` of a hardware-described texture) -90%.
* Global atomics on a histogram: `hough_scores` +43%, `build_histograms` +48%.
* No instruction class explains the errors (`classmix.py`: correlation of each class's dynamic share with log(hw/sim)
  within +-0.30 over the 85 launches). The error follows the number of waves instead (correlation -0.39 with log of the
  busiest core's waves): median hw/sim 1.19 for launches under 30 us, 1.15 for 30-100 us, 1.04 for 0.3-1 ms, 0.88
  above 1 ms. A fixed cost per launch or wave seems missing for short dispatches, and consecutive waves overlap on the
  hardware (workgroups start as slots free) where the model adds them end to end.

## Threadgroup loads in compiled code (gemm_block_size_bias)

`gemm_block_size_bias` (Super Resolution; tiled SGEMM, 32x4 threads, 8 KiB of threadgroup memory, 98 GPRs) as
Geekbench launches it (M=2304, N=64, K=576, grid 2x72): 229.6 us firmware time (Geekbench: 235 us). Ablations of its
`.asm` (`gemm_variants.py`; edited, reassembled with llvm-mc, run with `agx_bench_kernel`; times in us):

| variant | hw | sim |
| --- | --- | --- |
| base | 229.6 | 170.5 |
| no `fmadd32` in the inner loop | 213.6 (-7%) | 92.7 (-46%) |
| no address `iadd`s | 219.7 (-4%) | 154.4 (-9%) |
| `xy` threadgroup loads as `x` (half the bytes) | 226.0 (-2%) | 158.7 (-7%) |
| no threadgroup loads in the inner loop | 119.5 (-48%) | 82.7 (-51%) |
| no inner loop | 76.4 (-67%) | 39.8 (-77%) |
| no `xy` loads (8 of 10) | 140.5 (-39%) | 183.8 (+8%) |
| no `x` loads (2 of 10) | 205.6 (-10%) | 164.2 (-4%) |
| half of the `xy` loads | 181.0 (-21%) | 177.3 (+4%) |
| `wait 0` before the first use of the loads | 229.5 | |
| instruction bit 15 set on the threadgroup loads (Mesa's form) | 229.5 | |

On the hardware the inner loop is bound by the **number** of threadgroup load instructions (11-12 us each of the 10
per iteration, `x` or `xy`, broadcast or per-lane alike); the model makes it FMA-bound and the loads nearly free.
Loads without a `wait` are interlocked by the hardware (results correct).

Probe kernels (`tgprobe.py`, `tgprobe-matrix.txt`: PTX through zoc, 4 loads per iteration with consumers and address
arithmetic, 4-32 SIMD groups per core) are 52-114 core cycles per load per group, where `mem_probes.py`'s loop (8
loads from fixed bases, one `wait`) gives 31.4 (reproduced today). Hand-edited variants of the probe loop (4 groups per
core, 512 iterations of 8 loads):

| loop body | 4 groups | 32 groups |
| --- | --- | --- |
| 8 loads back to back | 101.1 us | 199.0 us |
| 8 loads, then 8 independent `iadd` | 105.0 (+4%) | 215.5 (+8%) |
| load, `iadd`, load, `iadd`, ... | 152.7 (+51%) | 307.5 (+55%) |
| load, 3 x `iadd`, load, ... | 164.7 (+63%) | 341.7 (+72%) |

**ALU instructions between threadgroup loads cost 50-70%; the same instructions after a run of loads are nearly free.**
The model has no such interaction (its predictions on the probe matrix are -41% to +244% off and barely depend on the
number of groups). Also measured: `xyzw` threadgroup loads at addresses that are only 4-byte aligned read the right
words (hardware and functional simulator).

More probe loops (512 iterations of 8 `x` threadgroup loads, whole-dispatch times through the driver, slope over 4096 /
8192 / 16384 iterations): broadcast or per-lane addresses, immediate indices 0-7 or 0, one base register or eight, a
`wait 0` after the 8 loads or none: all the same, **23 ns per load per group at 4 groups per core and 49 ns at 32**
(0.46 loads per cycle per core at 1398 MHz). `mem_probes.py --kind tgload` (in-shader timestamps, the difference of
32- and 128-step loops) gives 31.4 cycles per step at both 4 and 32 groups, i.e. 1.02 loads per cycle at 32: the two
methods disagree by 2.2x at 32 groups and have not been reconciled (the next step is one kernel under both harnesses).
Setting instruction bit 15 (set in Mesa's 8-byte form) on the 6-byte short form makes the load fault.

Reconciled: `mem_probes.py`'s own tgload kernel, timed by the firmware instead of in-shader timestamps, also takes
2.4x longer at 32 groups per core than at 4 (1,945 -> 4,630 us) while its in-shader per-step cycles stay 31.4: each
group progresses at the same rate, but not all 32 at once. The same 32 groups per core as 1 workgroup of 1024
threads, 2 of 512, 4 of 256 and 8 of 128 take 207, 145, 203 and 207 us; 16 groups per core (1 x 512 or 2 x 256) take
106 and 102 us, the same as 4 groups (101 us): up to 16 groups no slowdown at all.

`tgcheck.py` (47 cases: these loops, the GEMM ablations and the PTX probe matrix) gives the current model a median
error of 41% (4 within 10%). The model charged the threadgroup pipelines' occupancy to the access's result latency only,
so back-to-back loads whose results nothing reads cost only their issue slot. `AGX_TG_GATE=1` (experimental, off by
default) makes threadgroup loads and stores wait for the group's previous access and the core's pipeline, and
`AGX_TG_CORE_LOAD` sets the core's reference cycles per load: with 1.5, tgcheck's median error is 31% (21 of 47 within
25%; low-occupancy loops whose loads feed fadds stay too slow, +55 to +195%), but the stored in-shader set (tg.json)
fails at 29.5%: a model must have groups that each keep ~31 cycles per access while the core holds more of them than
progress at once. To do: a threadgroup-load issue model with that occupancy limit and the run-length effect, then
the open gaps above.

## Launch-time specialization (ZLUDA 6197520)

zluda_agx compiles kernels again with their scalar arguments and block size as constants once a launch repeats
(`docs/src/agx.md`). Geekbench GPU time (`fastsum.py`, 10th percentile per launch configuration): 35.75 -> 29.92 s.
`speccheck.py` runs the generic and the specialized code of each captured launch on the capture's memory
(`capvariant.py`): 59 captures give bit-identical outputs (`speccheck.txt`); per launch, gemm -43..-56%, im2col
(`build_patches_padding`) -70..-80%, depthwise convolution -66..-71%, `color_look_up_tetrahedral` +10% (index
arithmetic: two `imadd` become `bfi`/`or` sequences). Findings on the way:

- A kernel run with fewer registers allocated than its code uses faults (`Bad address`), also when only the upper part
  of a vector load's destination (`xyzw` into `r22_r23_r24_r25` with 24 allocated) is past the allocation. A capture's
  `regs.txt` is in 16-bit halves of the code that ran; `capvariant.py CODE.bin:HALVES` raises it for other code.
- A capture's uniforms are laid out for the code that ran: a module without textures has no texture heap uniform, so
  its arguments start at `u0` rather than `u2` (speccheck skips those).
- `div.approx` with constant operands folded to the exact quotient and gave other bits than the hardware reciprocal
  (`dye_advection`); it now always uses the hardware's (ZLUDA `llvm.agx.rcp.f32`).
- Immediate offsets on `xyzw` device loads (4, 16, 20, 32 elements) read the same words as register offsets.
- Geekbench chooses block sizes by timing launches: if those run generic code and later ones specialized, it may pick a
  slower configuration (run 19: `build_patches_padding` 1268 instead of 596 ms); specializations in the compilation
  cache are now used from the first launch.

## Simulator on specialized code; ALU throughput against occupancy

`speccheck.py --sim` also times both codes in `agx_dispatch`. On the 7 run-13 captures it agrees with the hardware on
which specializations are faster (6 of 7) but finds smaller gains: the specialized code is predicted well
(`build_patches_padding` 11.7 vs 12.5 us) and the generic code too fast (21.7 vs 61.8 us), whose difference is mostly
32-bit division by kernel arguments (a 32-instruction dependent sequence of `icmpsel`, `imadd`, `convert`, `rcp`).

`aluocc.py` times a loop (`probes/dv.ptx`: one such division and an add per iteration) with 256 and 2048 iterations on
the GPU (`agx_run_hw --back-to-back`, firmware times) and in the simulator, for grids that put 2 to 72 SIMD groups on
the busiest core (cycles per iteration on that core, 1398 MHz):

| grid x block | groups (busiest core) | division loop hw | sim | `xs.ptx` (16 `bfeil` + 16 `xor`) hw | sim |
|---|---|---|---|---|---|
| 38 x 32 | 2 | 108.1 | 104.5 (-3%) | 113.0 | 117.1 (+4%) |
| 38 x 64 | 4 | 113.1 | 104.6 (-8%) | 114.0 | 117.1 (+3%) |
| 19 x 128 | 4 | 113.2 | 104.6 (-8%) | 114.1 | 117.1 (+3%) |
| 38 x 128 | 8 | 231.4 | 123.0 (-47%) | 196.0 | 147.1 (-25%) |
| 152 x 32 | 5 | 122.5 | 104.6 (-15%) | 127.2 | 132.5 (+4%) |
| 304 x 32 | 9 | 231.3 | 142.0 (-39%) | 195.9 | 178.7 (-9%) |
| 76 x 128 | 12 | 231.3 | 167.7 (-27%) | 196.0 | 205.4 (+5%) |
| 152 x 128 | 20 | 327.6 | 273.6 (-16%) | 321.3 | 339.4 (+6%) |
| 152 x 256 | 40 | 662.1 | 546.1 (-18%) | 649.1 | 678.5 (+5%) |
| 304 x 256 | 72 | 1238.8 | 984.1 (-21%) | 1046.5 | 1217.7 (+16%) |

Two gaps: (1) 38 x 128 and 304 x 32 (8 and 9 groups on the busiest core by the cluster split) take as long as 76 x 128
(12 groups) with both loops: the busiest core holds more than the split gives, or two groups that share a scheduler
interleave worse than modeled. (2) The division loop is 16-21% slower than modeled from 20 groups on, while the
shift/xor loop matches. Per instruction at 40 groups per core (`probes/k_*.ptx`, 16 of one instruction per
iteration in 8 independent chains): `icmpsel` hw 259.9 vs sim 338.1 cycles (+30%: the hardware issues ~2.5 per
cycle per core), `imadd` 646.4 vs 672.3, `convert` 642.3 vs 672.3, `rcp` 972.5 vs 1024.1 (all within 5%); at 4
groups `mul.hi.u32` (a 32 x 32 -> 64 `imadd` and an `iadd`) 112.0 vs 77.6 (-31%: its latency is longer than
modeled) and `rcp` 109.9 vs 99.4. Summing the measured per-instruction costs of the division loop gives ~17.9
cycles per group per iteration against 16.5 measured: the hardware overlaps the units' work less than the model.

Workgroup placement (`probes/m.ptx`: each workgroup records `%smid`, the core index, while spinning;
`wgmap-placement.txt`): up to 32 workgroups of 128 threads each get their own core; 38 use 34 cores (4 cores of
cluster 3 take two); the busiest core then has 2, 2, 2, 2, 3, 3, 4, 5, 7 and 8 workgroups for 39, 48, 57, 64, 76,
96, 114, 152, 228 and 304 workgroups, which the simulator's cluster split (10, 9, 9, 10 cores) reproduces except
304 (9). Core indices are 0-9, 16-25, 32-41 and 48-57; workgroups arrive in runs of about 8 per cluster.

## Compiler regression check on captures

`speccheck.py --vs-captured` runs the code the current compiler generates for each captured launch against the code
that ran in the capture's run (an older compiler) on the same memory. After the threadgroup address offsets and the
16-byte threadgroup unrolling (llvm-project f4c3a027740c): 89 of 95 captures give the same output
(`compiler-regression-tgaddr.txt`; the others: `hough_scores` races plain stores to the same bins, `hough_lines` and
`nonmax_suppression` append through atomics, three stale captures fault with either code).

## Hoisted all-pairs loads (llvm-project bdadb1eb5b56), simulator

`particle` with the loads of its unrolled copies issued together: the current generic code takes 110 us on the run-13
capture against 153 us for the code that ran then (same output); the simulator predicts 179 against 197 us (-9%
instead of -28%) and overestimates both (2 waves of 54 one-group workgroups on the busiest core; wave overlap is an
open gap). The other run-13 captures time the same with either code on both (`speccheck.py --vs-captured --sim`).
Measurement note: `capvariant.py` now runs a warm-up first, since the first code timed after the idle time of
compiling ran at a low clock (+50..100%).

## Taken forward jumps in cached loops (cost.rs, 2026-10-23)

`k_ifk*.ptx`: 8 if-blocks per iteration whose condition is always false (a taken `jmp_exec_none` over 1-32 stores).
Hardware: 387 cycles per iteration at 4 groups per core, 580-595 at 40, whatever the size of the skipped body. The
simulator gave 393 and 3,591 (+518%): the 33-cycle scheduler hold of a taken forward jump was fitted to 40 KB
straight-line probes (playground 2026-10-20-fp32-ieee), which are instruction-fetch-bound. It now applies only to
kernels with more than 16 KB of code (`FETCH_BOUND_CODE_BYTES`, a guess between the two measured cases); smaller
kernels get `branch_forward_sched_cached_real` (0; `AGX_BR_FWD_CACHED` to experiment). k_ifk1 at 40 groups: 951
(+64%; the rest is the 3-unit `if`/`pop_exec` hold, which lower values fit here but make particle worse).

`gridsweep.py` (a capture with fewer workgroups) on `particle`'s current code, 200 back-to-back runs: hardware 49 /
50 / 70 / 90 / 113-126 us at 1140 / 1520 / 2052 / 3078 / 4096 workgroups; the simulator before 56 / 67 / 94 / 135 /
179 us (+13..+50%), after 39 / 57 / 77 / 103 us at 1520..4096 (-14..-20%). verify-dispatch: median error 0.054 ->
0.048, p90 0.135 -> 0.092, max 0.686 -> 0.215; verify-cost-model output identical; mem_check and tgcheck unchanged.
Open: at 1-2 workgroups per core the simulator is ~44% fast on particle (latency of its loads past L1).
