# Task: the render-pass model (`rust/src/raster.rs`, `agx_render`) — status and open items

Started 2026-11-21, following render-01 / render-02. Goal: a full model of the AGX raster path, decoded with direct ioctls.
Findings, accuracy and method: [results/2026-11-21-render-ioctl](../results/2026-11-21-render-ioctl/README.md).

## Done
- `agx_render PLAN.txt` simulates a `drm_asahi_cmd_render` end to end: VDM, PPP, vertex stage, primitive assembly (lists,
  strips, fans, indexed, restart, instancing, base vertex), near clipping, viewport, culling, a 1/256-snapped top-left
  rasterizer, tile-relative coefficient planes with RTZ interpolation, depth / stencil (all functions and ops), scissor,
  region clip, deferred HSR (Fragment control 2 pass type), late z, Z/S load / store (64x64 Morton tiles, constant-block
  compression), and the background / fragment / end-of-tile programs. On hardware replays, 160 scenes: 99.99% of pixels
  byte-exact, every scene within 1 LSB except 1-2 depth-tie pixels in dense random scenes.
- Timing: `raster::timing`, built on the calibrated core model. Held-out median errors 2.2% (vertex) and 4.8% (fragment).
- Harness: capture shim, genxml decoder, direct-ioctl replay (single, waited, or train), mutation probes, scene generator,
  suites, fitter, offline fixtures (`make verify-render-model`), clock keeper, GPU-recovery tripwire.

## Open
1. **Plane-setup arithmetic** (±1 ulp of C / A; per-tile A varies by up to 2 ulp): the hardware recomputes the plane per tile in
   f32. Edge-function forms with RNE / RTZ match 50-64% of words (`/scratch/render/v2/tools/cf_search.py`). This is the source
   of the last 1-LSB pixels.
2. **`iterproj` reciprocal**: within ±2 ulp of exact; none of division RNE / RTZ or reciprocal × multiply matches exactly.
3. DONE (v2): dense-pass fragment cost = tile-list block fetch chain (`raster::isp_ns`). Still open: a 6000-triangle overlapping
   pass is 38% high (block / visible terms over-count large overlapping triangles).
4. DONE (v2): the instanced misses were tiny-instance cost (vertices per instance), not binning; modelled in `timing`.
4b. **Use the tile -> core map in timing**: `raster::tile_core_map` / `shade_loads` (exported as `core_load_max`,
   `tile_load_max`) explain few-tile and 36-vs-64-tile shading times: shading is about t_tile x (1.3 x tiles on the busiest core
   + 0.48). A first unbounded Nelder-Mead refit collapsed; next try a bounded fit or a linear solve of the shading coefficients.
   The map itself is exact on 25/30 measured sizes; the in-cluster visiting order is not decoded.
5. **Partial renders**: bimodal per process (v2: detector via partial_bg -> bg; about 5x when they happen). Still open: when they trigger and what they cost (seen at ~1500 triangles). Count them through the vertex timestamp
   or a probe of the partial-background program, then model the extra bg / eot passes.
6. **Compressed stencil loads** (codes 0x60, 0x8c, 0xac, ...), compressed z16, and other depth codes than 0 / 3 are not decoded:
   they load as the clear value, with a warning in the log.
7. **MSAA**: the rasterizer handles sample positions (ppp_multisamplectl), but no MSAA scene has been replayed. Blocker: the
   linear readback needs the multisampled PBE layout.
8. Points / lines are assembled but not rasterized.
9. ZLS traffic is not in the timing model.

## Safety rules learned (keep)
- Never SIGKILL a process with GPU commands in flight: it faulted the GPU into about 1000 recoveries a second for 3 minutes.
  The clock keeper stops on a stop file.
- Run every GPU case under `gpu_safety.Tripwire` (kernel-log recovery counter, `/save/kernel.log`); stop on any recovery.
- Time short passes at the top clock: keeper warm-up, then back-to-back bursts. Waited reps run 1.5-3x slower.
