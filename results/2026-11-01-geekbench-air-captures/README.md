# Geekbench 7 GPU kernels on the AIR route: simulator vs hardware (2026-11-01)

The 29 kernels with the most GPU time in a Geekbench 7 GPU (CUDA) run on zluda_agx's AIR route (ZLUDA
claude/perf-air-agx), except the ray tracer's (trace_bvh*, miss, hit_surface, generate_rays, resolve_radiance,
accumulate_*: replays of BVH kernels with wrong data have hung the GPU), each captured after its 20th launch. Every
capture ran on the simulator first (none hit the step limit), then `scripts/capcheck.py --hw --hw-repeat 3` (the
median of 3 runs of 1000 back to back) against `agx_dispatch --launch-profile zluda`. No GPU faults.

| Model | Median abs. error | Within 10% | Within 25% |
| --- | --- | --- | --- |
| zluda profile, store floor launch + stores (`capcheck-hw3.txt`) | 18% | 9 / 29 | 23 / 29 |
| store floor overlapping the launch beyond the stores' time | 9% | 17 / 29 | 25 / 29 |
| and later waves warm when the data fits the system-level cache (`capcheck-final.txt`) | **8%** | 17 / 29 | 25 / 29 |

- The large image kernels write a lot over thousands of workgroups per core: their launch and their stores overlap
  (gradient: 967 us of workgroup launch, ~500 us of stores, 1036 us measured), so the store floor adds the launch only
  up to the stores' time; the store microbenchmarks (1.5-24 MB) stay within +4% to +14%.
- Later waves warm without the system-level-cache condition gave 6% (21 / 29 within 10%, 27 / 29 within 25%:
  image_rotate +35% -> -5%, convolve_2d +18% -> +1%) but made the direct-DRM calibration's DRAM-bound kernels too
  fast (verify-dispatch p90 9% -> 16%, max 74%): with the condition, the calibration is unchanged. The image kernels
  that would gain are classed as not resident by the footprint estimate (single-wave segments scaled to all cores,
  which counts the rows neighbours share several times). Capping the estimate at the captured data does not change
  that: their mapped allocations are 48-189 MB, above the 19 MiB the model takes for the system-level cache. Their
  data does not stay resident; what warms their later waves is reuse across waves (a stencil's next rows were read by
  the wave before). Next: warm only the lines a wave shares with the previous one (simulate wave 0 first, then the
  others against its lines).
- gemm_block_size_bias (+43%) examined: 18 workgroups (one per core, 4 SIMD groups, 8 KB threadgroup memory, 188
  register halves), a long K loop; per group 4363 instructions (1536 fmadd32, 1203 iadd, 960 threadgroup_load, 97
  device_load). Hardware ~71k cycles of execution, the model ~104k. Not the threadgroup load cost: AGX_TG_GROUP_LOAD 31
  -> 8 leaves 77.5 us, AGX_TG_GATE 73.3 us. The model's timeline (`AGX_ISSUE_PROFILE=1 AGX_ISSUE_PROFILE_MIN=4000`):
  each of the inner loop's 16 fmadd32 waits ~409 ticks (~41 reference units: tg_load_issue 21 + tg_load_fast 19.4)
  for the threadgroup load it uses, the 10 loads of an iteration serialized per group; the loop takes over half of
  the group's time. **Correction**: a first reading took this for threadgroup-load cost and measured `tg2.ptx` /
  `tg10.ptx`, but the compiler hoists their loads out of the loop (they ran once): what they time is 10 (or 2)
  independent FMA accumulator chains, 21.7 / 25.4 / 26.5 us with 1 / 2 / 4 groups on the GPU (~5.2 cycles per FMA
  for one group) against 84.4 us in the model, which charges each FMA ~21 units. A global tg_load_issue of 4 broke
  three calibrated checks and was reverted; so was a streaming-load variant, which fitted the GEMM (54.2 us) for the
  wrong reason. The likely cause for both is the result-cache rule (only the last 8 written registers fast, older
  ones waiting a ~17-cycle writeback): to check against the 2026-10-11 result-cache measurements next.
  The compiled tg10 loop is 10 fmadd32 (each into its own accumulator, operands a register loaded before the loop and
  a uniform) + iadd + while_icmp + jmp_exec_any: ~52 cycles per iteration on the GPU for one group (~4 per
  instruction), the model ~220 units, all charged to the FMAs. The result-cache measurements (10 rotating fadd32:
  1.63 cycles per instruction) used 72-instruction loop bodies, so a short loop's control and taken jump cost more
  here; next: the same loop with 10 / 20 / 40 FMAs per iteration (and 1 / 2 / 4 groups) on both, to separate the
  loop's cost from the FMAs'.
  Done (`fma*.ptx`, the multiplicands from an ALU add before the loop): 21.1 / 15.4 / 12.8 us on the GPU, 22.4 / 14.7 /
  12.1 in the model: FMA chains and loop control are modelled well. The difference in tg10 is that its multiplicands
  come from threadgroup loads never followed by a `wait`: the model's load deferral (ALU work in flight defers a
  load's completion) kept re-deferring them at every ALU instruction for the rest of the program, as only a `wait`
  settles them. `AGX_TG_SETTLE_ON_READ=1` settles a threadgroup load once an instruction reads its result: tg10 84.4
  -> 23.0 us (GPU 26.5), tg2 17.0 (17.8), gemm_block_size_bias +43% -> -2%, depthwise_convolution_2d +24% -> +1%,
  calibration checks unchanged. Not the default: llama.cpp's shared-memory reductions (soft_max, mul_mat_vec_q/_f)
  then come out 13-27% fast and gemm_non_block_size_bias -26%, an undercount the stale deferral had offset (all 40
  captures: mean |error| 10.5% -> 12.1%, median 8% both). In llama.cpp's mul_mat_vec_q<Q5_1> (hw 26.3 us, model 25.7
  default / 19.3 settled) the settled time is all in the reduction after the barrier (by instruction kind, ticks
  before issue, default -> settled: fadd32 35177 -> 10601, simd_shuffle_xor 14581 -> 1739, device_store 6141 -> 336,
  if_icmp 5856 -> 416; the main loop's imadd / asr / bfi unchanged): the partial sums exchanged through threadgroup
  memory and reduced with shuffles cost the GPU something the stale deferral stood in for. Next: a microbenchmark of
  that sequence (threadgroup store, barrier, threadgroup load, shuffle reduction) with 2 and 4 groups.
  (`AGX_ISSUE_PROFILE_SKIP` skips the cost model's own calibration calls, `AGX_ISSUE_PROFILE_ROWS` lists more rows.)
- Open: gemm_block_size_bias +43% (several workgroups on one core modelled as overlapping less than they do, the
  known gap); blur_gray, image_rotate, hough_scores +27-38% (hundreds of waves, each simulated with its own latency,
  which the hardware overlaps); particle -22%; depthwise_convolution_2d +24%.
