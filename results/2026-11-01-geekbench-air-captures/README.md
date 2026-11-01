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
  memory and reduced with shuffles cost the GPU something the stale deferral stood in for. Measured (`red.ptx`: 200
  rounds of FMA, threadgroup store, barrier, another group's partial loaded, 5 butterfly shuffles + adds, barrier;
  38 workgroups): 2 groups 56.1 us, 4 groups 57.7 us on the GPU; the model 72.2 / 72.7 by default (+29%), 60.9 / 61.7
  with AGX_TG_SETTLE_ON_READ (+8%). So the reduction sequence itself is better with the settle rule; what makes the
  llama.cpp kernels fast under it is elsewhere (their device loads and waits, or the stores at their end), which the
  stale deferral's overcharge offset. The rule stays opt-in until that is found.
  Located further: llama.cpp's mul_mat_vec_q<Q5_1> replayed with 1 / 4 / 12 workgroups of 2 groups per core (grid x 1216
  / 4864 / 14336): GPU 18.1 / 16.6 / 25.75 us; the model 17.35 / 17.35 / 25.72 by default and 17.35 / 17.35 / 19.31
  with the settle rule. The two agree up to 5 workgroups per core and part at 12 (two waves of 12 SIMD groups): the
  GPU's time grows 1.55x from 5 to 12 workgroups per core, the settled model's only 1.11x. What the stale deferral
  supplied is contention between many resident groups; next, a model of that (shared issue or register-file
  bandwidth at high occupancy) measured on its own.
  Correction from a finer sweep (6 / 7 / 8 / 10 / 12 workgroups per core): GPU 16.9-18.4 / 19.1 / 19.6 / 23.5 / 26.3
  us; default 17.35 / 21.2 / 21.9 / 23.8 / 25.7; settled 17.35 / 17.35 / 17.35 / 17.35 / 19.3 (the same with later
  waves cold). The settled model is flat because its execution time falls under the instruction-fetch floor (138
  code lines x 104 ns + launch + writes = 17.35 us) until 12 workgroups per core: it is not occupancy contention but
  the settled model's execution of this kernel being too short in every wave, hidden by the floor at low
  occupancy. That disagrees with red.ptx, where the settle rule is the more accurate one; the difference between
  the two (mul_mat_vec_q's device loads and waits around the threadgroup exchange) is the open question.
  (`AGX_ISSUE_PROFILE_SKIP` skips the cost model's own calibration calls, `AGX_ISSUE_PROFILE_ROWS` lists more rows.)
- Open: gemm_block_size_bias +43% (several workgroups on one core modelled as overlapping less than they do, the
  known gap); blur_gray, image_rotate, hough_scores +27-38% (hundreds of waves, each simulated with its own latency,
  which the hardware overlaps); particle -22%; depthwise_convolution_2d +24%.

Rechecked under the zluda profile with partial warm and chained waves (results/2026-11-01-fft-shared-captures):
AGX_TG_SETTLE_ON_READ=1 still trades gemm_block_size_bias (+36 -> -10%) against llama.cpp's reductions
(mul_mat_vec_q<Q5_1> -7 -> -33%, mul_mat_vec_f +17 -> -15%, soft_max -5 -> -19%) and gemm_non_block_size_bias
(-10 -> -33%): 40 captures mean 10.2 -> 11.7%. Still opt-in; the reductions' missing cost (high-occupancy
contention, see above) is what to model before it can be the default.

What the reductions' missing cost is (`red/`, 456 workgroups = 12 per core, GPU / model default / model settled, us):
- red.ptx (store, barrier, load, 5 shuffles): 64 threads 59.8 / 72.8 / 61.5; 128 threads 83.8 / 75.9 / 65.1. The
  GPU takes 1.40x for twice the SIMD-groups, the settled model 1.06x.
- red_tg.ptx (no shuffles): 64 threads 53.0 / 57.4 / 57.4, 128 threads 61.0 / 59.1 / 59.1: modelled.
- red_shfl.ptx (no threadgroup memory or barriers): 64 threads 32.0 / 25.3, 128 threads 57.4 / 44.8 (model -21%);
  one 32-thread group per core 15.04 / 13.92 (-7%).
- Integer chains (mad.ptx, shrbfi.ptx, 8 per thread): 2 and 13 workgroups per core within -4..+11%.
So the gap is shuffle-and-add chains with many SIMD-groups resident: in the model the kernel is issue-bound with all
52 groups resident (fadd32 and simd_shuffle_xor waits); scaling simd_shuffle_xor's per-core rate down to 0.4x does
not move it (25.3 -> 26.4 us), so it is not that unit's cap but the core's issue across many groups of dependent
shuffle / add chains. Next: dependent and independent chains of fadd32 alone and of shuffles alone at 1, 4, 16 and
48 SIMD-groups per core, to calibrate issue under occupancy; then the settle rule can be the default.

Calibration data (`red/fadd.ptx`, `red/shfl.ptx`: one dependent chain per 32-thread group, 16 per round, 200
rounds; 1 / 4 / 16 / 48 groups per core; GPU / model zluda us):
- fadd32 chain: 12.33 / 12.69, 12.5 / 12.71, 17.12 / 18.21, 39.21 / 42.14 (+2..+7%): modelled.
- simd_shuffle_xor chain: 18.96 / 19.26, 24.25 / 19.26, 38.29 / 28.45, 83.17 / 72.73 (+2, -21, -26, -13%).
Shuffles per core per us after the 2.3 us launch, GPU: 193, 584 (3.0x), 1422 (7.4x), 1899 (9.8x) with 1 / 4 / 16 / 48
groups: a gradual saturation (each shuffle waits longer as the unit fills, from 4 groups on), where the model is
linear up to its cap (4 groups no slower than 1, ~1960 per us already at 16). Scaling the shuffle rate cannot fit
both ends (0.6x: 16 groups 36.9 us but 48 groups 95.6; 0.25x: 4 groups 27.0 but 16 groups 79.5). So the model
needs a load-dependent shuffle latency (a queue at a core-wide shuffle unit) rather than a lower cap; with it the
reductions' cost, and then the settle rule, can follow.

Tried: `AGX_SHFL_QUEUE` (cost.rs), each shuffle's latency + q cycles per other SIMD-group's shuffle in flight on the
core. q = 0.6 fits the dependent chains within 6% (20.1 / 23.2 / 38.0 / 83.8 us against 18.96 / 24.25 / 38.29 /
83.17), but the mixed reduction kernels hardly move (their adds keep fewer shuffles in flight; red_shfl 25.3 -> 24.4
us, GPU 32.0), the 48 captures are unchanged (mean 10.2%; with the settle rule 11.6%), and verify-cost-model's
instruction-bench fit worsens (p90 0.169 -> 0.278) as does the core-scaling grid (0.277 -> 0.332): their many
independent chains per group are already limited by the unit cap. Kept off (opt-in). The extra latency should follow
the unit's recent load, not a count of shuffles in flight; and the reductions' missing cost is not only the shuffles'.

