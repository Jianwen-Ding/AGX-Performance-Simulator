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

Simulator (`scripts/frag_cost_check.py`: agx_dispatch over the captured fragment `main` binaries `frag_<N>_<flat>.bin`, 4096 workgroups of
1024 threads, loop bounds u11 / u13 / u15 = 256, `--no-mem`): iter / iterproj / ldcf were generic device loads with an
unmeasured memory latency; they now issue in the ALU pipe (class fmadd32), ld_tile / st_tile like an fadd32. Result: smooth 1 -> 24 varyings +59 us
simulated vs +52 us measured; flat +90 vs +131 us. Every ALU class for ldcf gives the same +90 us, so the missing flat time is not
instruction issue (candidates: per-coefficient fetch in the rasterizer, 4x more instructions competing for the coefficient read port).
The absolute level is 24% low (855 us simulated vs 1131 us measured for 1 varying): fragment launch / tile setup is not modelled.
Tile loads and stores: blending 8 layers of a trivial shader costs 5 ps per pixel and layer (0.055 -> 0.092 ms at 1024x1024), i.e. nothing next to the shader.
