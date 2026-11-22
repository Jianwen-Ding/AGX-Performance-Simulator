# Fragment varying cost (2026-11-14)

Probe: `apple-documentation-playground/tests/gles_raster.c` with `RASTER_VARY=N` (N vec4 varyings summed by the fragment shader) and
`RASTER_FLAT=1`, `RASTER_FLUSH=1 RASTER_KEEP=1`, 2048x2048, one opaque layer, 256-iteration dependent ALU loop (so the shader is
latency bound), 20 frames, best of 5. `hw_sweep_vary.txt` is the measurement (ms per frame). `hw_sweep_vary_alu8.txt`: the same with an
8-iteration loop, where fixed per-frame cost (~0.1 ms) hides every difference.

| varyings | smooth (iterproj) ms | flat (ldcf) ms |
|---|---|---|
| 1 | 1.131 | 1.191 |
| 4 | 1.141 | 1.207 |
| 8 | 1.148 | 1.230 |
| 16 | 1.167 | 1.275 |
| 24 | 1.183 | 1.322 |

Slope: smooth 2.3 us per vec4 varying (4.19 Mpixel: 0.54 ps per pixel and varying), flat 5.7 us. Interpolating is nearly free next to a
latency-bound ALU chain.

Fragment launch / issue (2026-11-14, later): the ALU-depth and size sweeps (`hw_sweep_size_alu.txt`) show hardware time linear in loop iterations at
2048x2048 (4.14 us per iteration, ~0.08 ms fixed) and the parallelism-limited floor ~0.32 ms below 1024x1024. The compute issue model gives 3.30 us per
iteration, so `agx_dispatch --frag` applies `frag::FRAGMENT_ISSUE_FACTOR` = 1.25 to the simulated core cycles (empirical, fitted on this shader family; cause unknown). With it
the eight cases of `scripts/frag_cost_check.py` are 4-19% low (without: 24-35% low); the varying-count increments become smooth +74 vs +52 us, flat +112 vs +131 us.

Simulator (`scripts/frag_cost_check.py`: agx_dispatch over the captured fragment `main` binaries `frag_<N>_<flat>.bin`, 4096 workgroups of
1024 threads, loop bounds u11 / u13 / u15 = 256, `--no-mem`): iter / iterproj / ldcf were generic device loads with an
unmeasured memory latency; they now issue in the ALU pipe (class fmadd32), ld_tile / st_tile like an fadd32. Result: smooth 1 -> 24 varyings +59 us
simulated vs +52 us measured; flat +90 vs +131 us. Every ALU class for ldcf gives the same +90 us, so the missing flat time is not
instruction issue (candidates: per-coefficient fetch in the rasterizer, 4x more instructions competing for the coefficient read port).
The absolute level is 24% low (855 us simulated vs 1131 us measured for 1 varying): fragment launch / tile setup is not modelled.
Tile loads and stores: blending 8 layers of a trivial shader costs 5 ps per pixel and layer (0.055 -> 0.092 ms at 1024x1024), i.e. nothing next to the shader.

## Second shader family and the coverage / depth ops (2026-11-14, later)
`hw_sweep_variants.txt` (2048x2048, `RASTER_DISCARD=1` adds `if (a < -1e30) discard;`, `RASTER_DEPTH=1` adds `gl_FragDepth = v.x * 0.5` and a depth buffer):
base 0.094 / 0.426 / 1.491 ms at 16 / 64 / 256 loop iterations; discard 0.071 / 0.423 / 1.493; depth write 0.078 / 0.442 / 1.550. A never-taken discard (two fcmpsel, or,
wait_pix, sample_mask) is within noise, a depth write costs ~4% of the 256-iteration shader. `sample_mask` and `zs_emit` had no cost class (`agx_cost` refused the
program); they now issue like an fadd32. Simulated discard +0.2% vs measured +0.1%; the depth write's `zs_emit` sits in the driver epilogue, outside the captured main, so only the
depth arithmetic is simulated (+0.6% vs +4% measured).

**Limitation.** This base shader (`a = a*b + 0.5; b = b*a - 0.25`, two dependent fmadds per iteration, `frag_dep2fma.bin`) is predicted at 1073 us (with the 1.25 factor) against 1491 us
measured (-28%), while the 1-varying family above is -6%: both come out at the same simulated time (the compute issue model sees the same five instructions per
iteration) but the hardware runs this one 32% slower. The single `FRAGMENT_ISSUE_FACTOR` therefore does not generalise: expect fragment ALU-loop predictions within -30% .. +5%.

## Root cause of the fragment "factor": the compute loop model (2026-11-14, last)
The same ALU loops as compute kernels (`apple-documentation-playground/tests/gles_compute_alu.c`, 2048x2048, the store guarded by a never-true compare so no
bandwidth is measured; `hw_compute_loop_variants.txt`) take the same time as the fragment versions (family B: 1.45 ms compute vs 1.49 ms fragment at 256 iterations;
family A 1.09 vs ~1.13). So the fragment stage is not special: the **compute issue model underpredicts these tight loops** (no `--frag`: A 862 us, B 864 us at 256
iterations; measured 1091 / 1449 us, -21% / -40%). Hardware time per iteration (us, 2048x2048): 1 fmadd 4.09, 2 fmadds (dependent or independent) 5.51, 4 fmadds 5.50;
the model gives 2.60 / 3.29 / 4.69. So per-iteration time on hardware is nearly independent of the fmadd count between 2 and 4 (loop control / issue structure dominates),
which the model does not reproduce (it adds ~0.7 us per fmadd on top of a lower base). `frag::FRAGMENT_ISSUE_FACTOR` is a stop-gap for fragment programs; the real fix is in
`cost.rs` (taken-backward-jump and exec-mask loop costs at high occupancy), to be calibrated with compute loop microbenchmarks of 1..8 ALU instructions per iteration.

## Vertex outputs and coverage (2026-11-14, last)
`hw_sweep_vertex.txt` (`RASTER_VARY=N`, 301,088 triangles = 903,264 vertices, 1024x1024): `st_var` is a throughput-limited path (vertex output store). With
`ST_VAR_CORE_CYCLES` = 48 per st_var per group the simulated times are -4 / -9 / -36 / +5% for 1 / 4 / 12 / 24 varyings (the 12 and 24 points saturate on hardware).
MSAA 4x: Mesa's standard sample positions (0.375,0.125) (0.875,0.375) (0.125,0.625) (0.625,0.875) reproduce the hardware's covered-sample count on all 1507 touched pixels and
the exact per-pixel `gl_SampleMaskIn` (sr124); this check found that the simulator evaluated sr60 / sr124 once per instruction instead of per lane (fixed).

## Loop-model investigation (2026-11-14, final)
Hardware measurements (compute, `gles_compute_alu`, 4.19 Mthreads unless noted; files `hw_compute_*.txt`):
- **Straight-line ALU** (fully unrolled, 32 / 64 / 96 independent fmadds per thread): 0.224 core cycles per group-instruction = **4.47 per core-cycle**, i.e. the four schedulers at peak.
- **The same ALU instructions inside a loop**: per group-iteration 1.0 + 0.6 x n core cycles for n independent fmadd chains (n = 1 / 2 / 3 / 4 / 6 / 8: 1.60 / 2.2 / 2.8 / 3.47 / 4.6 / 5.6), whatever the operand form (`a*a-0.25`, `a*b+0.5`, `a*b+c`, an fadd with an immediate: all ~3.4 at n = 4). Dependent chains of 1..4 cost the same as 1 (3.95 / 3.97 / 4.10 us per iteration), 8 dependent cost more.
- **Occupancy** (workgroups of one SIMD group, 38 per row, k groups per core, 4 independent fmadds + loop control, `hw_compute_loop_occupancy_ipc.txt`): per group-iteration core cycles 9.9 / 5.0 / 4.15 / 3.98 / 4.04 / 4.04 / 4.29 / 4.72 at k = 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 (IPC saturates at ~1.75 of 4).
  The simulator gives 10.1 / 5.06 / 3.37 / 2.53 / 2.12 / 2.01 / 2.05 / 2.26: **hardware and model agree to 1-2% up to 8 groups per core (latency-bound), then the model keeps scaling while hardware saturates: ratio 1.23 at 12, 1.58 at 16, 1.9-2.1 from 24 groups.**
So the compute model's loop throughput is ~2x too high once more than about 8 groups per core are resident, although straight-line ALU throughput is exact. The shortfall is per loop iteration (control: `while_icmp` + taken backward `jmp_exec_any` + the loop-carried chains), not per instruction. A constant scheduler hold per taken backward jump does not fit all three loop shapes (+0.61 / +0.90 / +0.33 core cycles per group-iteration for 1 / 2 / 4 fmadds), and the existing validated loop kernels (`make verify-cost-model`) match without one, so the fix needs the validated loop set re-recorded at >= 16 groups per core. Not done; `frag::FRAGMENT_ISSUE_FACTOR` stays a stop-gap.

## Compute blit replay and image_write_block cost (2026-11-14, final)
`scripts/frag_blit_check.py` replays Mesa's compute blit program (explicit-coordinate st_tile u8norm, threadgroup barrier, texture_sample, image_write_block) for five glBlitFramebuffer cases (nearest and linear, down- and upscaling, odd sizes): **0 differing bytes out of 4096 .. 24000 each**. This fixed the simulator's `image_write_block`: lane (0, 0) of each 32 x 32 workgroup stores the whole tile block, clipped to the image (it was one pixel per lane). Timing (`hw_blit_time.txt`): 4096 -> 3072 takes 145 us, 2048 -> 1536 61 us; the simulator's `agx_dispatch` gives 111 / 32 us, the same 11-12 ps per pixel slope (11.2 vs 11.9) with ~30 us of per-command overhead on hardware that a 1-command timing includes: the generic store / texture classes already fit the block store, so it keeps them.
