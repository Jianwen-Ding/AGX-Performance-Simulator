# Render (raster) pass model, decoded through direct ioctls (2026-11-21)

The simulator now runs a whole `drm_asahi_cmd_render` itself (`rust/src/raster.rs`, `agx_render`): VDM control stream, PPP state,
vertex shading, primitive assembly, clipping, viewport, culling, rasterization, coefficient setup, depth / stencil, hidden-surface
removal, fragment / background / end-of-tile programs, Z/S load and store. It also predicts the pass's vertex and fragment times.
Every rule below was checked against the GPU by **replaying a captured pass through direct ioctls**, with no Mesa in the loop.
Most were checked a second time by replaying mutated passes (patched state words, vertex data, or our own assembled shader code).

No proprietary software is involved. Passes come from Mesa's open-source GLES driver, the field layouts come from Mesa's open genxml
(`cmdbuf.xml`), and the probe shaders are assembled with the in-repo assembler.

## Method

| Step | Tool |
|---|---|
| Capture: run a GLES scene under an `LD_PRELOAD` shim that copies the SUBMIT command buffer and every bound buffer (read only) | `tests/gles_scene.c`, `tests/render_trace.c` |
| Decode: `drm_asahi_cmd_render` (240 B), VDM blocks, PPP records, USC words, CF bindings, PBE / texture / ZLS | `scripts/render_decode.py` (genxml-driven, pack + unpack) |
| Replay: re-create every buffer at its VA, submit the same command buffer with our timestamps, dump the framebuffer | `tests/drm_render_replay.c`, `scripts/render_replay.py` |
| Read back pixels: switch the end-of-tile PBE (and the partial-render reload texture) to an uncompressed linear layout | `scripts/render_probes.py` `Pass.linear_pbe_patch` |
| Probe: patch fields or replace the fragment shader with our own code (lane ids, `ldcf` coefficient readout, `iter` values) | `scripts/render_probes.py` |
| Simulate the identical inputs (the replay plan), then compare pixels and times | `rust/target/release/agx_render PLAN`, `scripts/render_check.py` |
| Time at the top clock: clock-keeper warm-up, then a burst of 15 back-to-back submits; median of reps 3-14 | `scripts/clock_keeper.py`, `render_probes.run_warm` |

## Decoded hardware behaviour (all HW-checked)

| Item | Finding | Evidence |
|---|---|---|
| Fragment SIMD group | an 8x4-pixel block of 2x2 quads, four across; lane = 4·quad + 2·(y&1) + (x&1) | lane-id shader readback |
| sr0 / sr1 in fragment programs | the 32x32 tile coordinates | same probe |
| sr53 in fragment programs | groups of a tile numbered by 16x16 quadrant (TL, BL, TR, BR), row-major inside | same probe |
| Triangle merging | tiles crossed by a shared edge get 34 groups instead of 32: edge blocks of the two triangles mostly share a group | same probe |
| Vertex snapping | 1/256 pixel (8 sub-pixel bits); planes are fitted to the snapped vertices | ldcf readback: 0-1 ulp snapped, 600-2900 ulp unsnapped; 1/16 and 1/1024 worse |
| Coefficient registers | A and B in f32; C = the plane's value at the **tile origin**, so the iterator works in tile-local coordinates. A constant plane's A and B read as the smallest denormal | ldcf readback, 12 tile/register pairs: 9 bit-exact, 3 within 1 ulp |
| Varying interpolation (`iter`) | exact A·x + B·y + C, rounded **toward zero** to f32 (nearest-even matched 52%) | iter readback with the HW's own coefficients: 1498 / 1498 |
| `iterproj` | within ±2 ulp of exact numerator / denominator (an approximate reciprocal; exact form open) | iterproj / iter / W readback, 1401 pixels |
| Top-left rule and facing | top edge runs +x and left edges run −y in the positive-area orientation; with `Front face CCW`, positive area is the back face | 8 cull scenes, 5 shared-edge scenes, fan of 12 |
| Guard band | vertices far off screen are rasterized exactly; only the near plane (window z ≥ 0) is clipped | clip scenes |
| PBE linear stride | row pitch = ((field >> 4) + 1) · 16: rows are whole 16-byte units, Mesa writes pitch − 16 | 33-px target stored 128-byte rows |
| End-of-tile `image_write_block` | `q0 = 0` ignores the coordinate operand and each lane stores its own pixel (`q0 = 1` is the blit's explicit-coordinate form) | EOT replay byte-exact |
| HSR switch | **Fragment control 2's pass type**: Opaque = deferred HSR, only the last covering fragment per pixel is shaded; Translucent (blend) / Punch through (discard, gl_FragDepth) = shaded in order. Mesa leaves the first Fragment control at Opaque | flipping the field by ioctl: blend forced Opaque 691 → 96 µs and the image becomes one layer; opaque forced Translucent 92 → 304 µs, same image |
| Depth surface layout | GPU tiling = 64x64-element tiles (4 B; 128x64 for 2 B, 128x128 for 1 B), Morton order inside, x in the even bits | 12-replay bit sweep of a Z load |
| Depth compression | 16x16-pixel blocks, one metadata byte per 8x4 sub-block: code 3 = constant, value in word (block·256 + sub-block); all-zero metadata loads as 1.0 | Z-load replays with patched data, metadata and `Z Load Compress` |
| Z / S load | `ZLS Control` Z Load / S Load with a masked clear (`glDepthMask(GL_FALSE)` keeps the old depth, and Mesa sets Z Load) | depth_nowrite, stencil_invert_mask |
| Partial renders | passes with ~1500+ triangles reload the target through the partial-background program's texture | replays break until that texture gets the same linear layout |

## Functional accuracy (simulator vs hardware replay, RGBA8 bytes)

| Set | Scenes | Byte-exact scenes | Within 1 LSB | Pixels exact | Replay = Mesa GL readback |
|---|---|---|---|---|---|
| suite1 (features) | 58 | 52 | 58 | 374,751 / 374,807 | 58 |
| perf1 (timing sweeps) | 43 | 43 | 43 | 17,388,544 / 17,388,544 | 43 |
| perf2 (vertex / binning sweeps) | 17 (+1 probe limit) | 15 | 15 | 3,473,406 / 3,473,408 | 17 |
| held1 (held out) | 22 | 19 | 21 | 5,030,547 / 5,030,568 | 22 |
| held2 (held out, all random) | 20 | 18 | 19 | 2,737,761 / 2,737,764 | 20 |

Remaining differences:
- 1-LSB pixels, from ±1 ulp of plane setup or `iterproj`.
- 1-2 pixels in dense random depth-tested scenes: depth near-ties under the same ±1 ulp.

## Timing accuracy (top clock, |predicted / measured − 1|)

| Set | Vertex median / p90 | Fragment median / p90 |
|---|---|---|
| Fit set, leave-one-out (suite1 + perf1 + perf2, 118) | 2.6% / 13.9% | 3.1% / 14.3% |
| held1 (22; examined once before the last model change) | 1.9% / 7.2% | 7.0% / 12.9% |
| **held2 (20, fresh, drawn at random)** | **2.2% / 24%** | **4.8% / 31%** |

The formula is in `raster::timing`. Fragment shading time uses the calibrated core model (`cost::simulate_trace`), which needs a
×1.21 factor plus a tail term: hardware / core model is 2.2× at 144 tiles and 1.35× at 2304. Measured structure:
- **HSR:** 8 opaque layers cost the same as 1 (30.7 µs), with or without a depth test.
- **Blending:** each blended layer adds its full shading cost.
- **Tile cost:** ≈10.6 ns per processed tile, overlapping with shading (max, not sum).
- **Vertex stage:** ≈6.8 µs fixed, saturating in vertices, ≈42 ns per draw.

Known misses:
- Large instanced triangles in the vertex stage (≈35% low).
- Thousands of small, heavily overlapping triangles in the fragment stage (≈50% low).
- Z/S load / store traffic (one stencil case 30% low).

## Freeze safety

- **GPU side:**
  - Every replay waits at most 2 s per submit, with a 20 s process timeout.
  - Every case runs under `gpu_safety.Tripwire`, which counts the kernel log's "Attempting recovery" lines before and after. Any recovery stops the suite.
  - One incident happened and was fixed. SIGKILL of the clock keeper while its commands were in flight tore its VM down under the GPU: gCDM_CS unmapped faults, then about 1000 recoveries a second for 3 minutes (18:13:33–18:16:28 UTC). The GPU recovered: the s1 replay was exact afterwards. The keeper now stops gracefully, using a stop file and letting the running submit finish.
  - One unexplained single IAP fault (unmapped read at 0x3039f45ac0) did not reproduce.
  - All later runs: 0 recoveries.
- **Simulator side:**
  - `agx_render` runs under a 120 s timeout and an 8 GB address-space limit.
  - The longest case takes 7 s; all 160 cases take 65 s in total.
  - A per-primitive tile-state scan made 10,000-triangle passes take 92 s. Replaced by direct lookups, they take 0.8 s.

## Reproduce

```sh
make verify-render-model                                  # offline: 20 stored replays (fixtures/), pixels + timing
python3 -I scripts/render_scene.py DIR [--perf|--perf2|--held|--held2 --seed N]
python3 -I scripts/render_check.py suite DIR/suite.json --out CASES.jsonl   # hardware: capture, replay, simulate, compare
python3 -I scripts/render_summary.py CASES.jsonl ; python3 -I scripts/render_fit.py check CASES.jsonl
python3 -I scripts/render_fit.py nl TRAIN.jsonl... held=HELD.jsonl                 # refit the timing model
```
