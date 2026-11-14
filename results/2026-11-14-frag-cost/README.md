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
