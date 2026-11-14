# Task: model the render (raster) path in the simulator

Started 2026-11-14 at the user's request. The simulator (`rust/src`) models compute dispatches only. Mesa's asahi driver
runs GLES/Vulkan on this machine (open source; no NVIDIA material involved), so render workloads can be measured on the
GPU. Mesa sources for the hardware structures: `agx-notes/mesa/src/asahi/lib/` (`agx_tilebuffer.c`, `agx_ppp.h`, `agx_usc.h`,
`agx_bg_eot.c`, `agx_nir_lower_tilebuffer.c`, `agx_nir_lower_msaa.c`), `src/asahi/genxml`.

## Phases
1. **Probe + first findings (done):** `apple-documentation-playground/tests/gles_raster.c`,
   `novel-results/2026-11-14-render-hsr/` (hidden-surface removal confirmed; fragment ALU rate).
2. **Capture the vertex / fragment binaries** of the probe with `scripts/agx_capture.py`; run `agx_cost` on them and compare
   the fragment ALU sweep (per-pixel cost at 64/256/1024 iterations) with the measured 1.44 ps per pixel-iteration. This
   tests whether the existing core model carries over to fragment SIMD groups (32 px, 2x2 quads) unchanged.
3. **Measure the fixed structure:** tile size and count (sweep target size, triangle size), bin / vertex cost (many small
   triangles), tile load/store and clear cost (RGBA8/RGBA16F/D32, MSAA 2/4), blend as tile-memory read-modify-write,
   what disables HSR (discard, depth write, alpha-to-coverage, sample mask), quad overshading on thin triangles.
4. **Add `rust/src/raster.rs`:** `agx_render` input = a pass description (target, tile size, triangles with covered pixel
   counts / overdraw order, fragment + vertex binaries, flags); output = predicted ms. Reuse `cost.rs` for shader time per
   SIMD group and `dispatch.rs` wave/occupancy logic with the fragment-specific limits (tile memory, imageblock).
5. **Calibrate and verify** against the stored sweeps (`make verify-render-model`), then real captures (glmark2-style scenes).

## Rules
One heavy GPU job at a time; probes need a flush/read per frame. Record failed paths in /scratch/render/NOTES.md.
