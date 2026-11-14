# Task: fragment-stage instructions in the functional simulator and cost model

Goal (2026-11-14): decode and model the raster-path instructions the compute work never exercised. All 26 open descriptors
in the playground's isa-coverage that are fragment-only. The disassembler already decodes them; `agx_run` did not execute them
(they fell through to applegpu's emulation, which has no semantics for them).

| Instruction | Status |
|---|---|
| `iter`, `iterproj` | **Executed** (`fsem::frag_iter`, `frag.rs`, `agx_run --frag FILE`): channel k = plane(I+k) at the lane's pixel; `iterproj` divides by plane(J) (the 1/w plane). Checked against a hardware render: 32 lanes x 4 channels within 5e-7 (`scripts/frag_iter_check.py`, results/2026-11-14-frag-iter). fp16 destination (16-bit register tuple) rounds RNE. |
| `ldcf` | **Executed**: the (A, B, C) triple of each channel's coefficient register (flat varyings: Mesa compiles flat to `ldcf` + ALU). Not yet compared to hardware. |
| `wait_pix`, `signal_pix` | decoded; costed as barriers; functionally a no-op (single pixel order). Open: cost (raster-order-group wait), what the `i`/`j` immediates select. |
| `ld_tile`, `st_tile`, `image_write_block` | decoded (format, mask, pixel offset, explicit coordinates); no tile-buffer memory model. Mesa source: asahi/lib/agx_tilebuffer.c, agx_nir_lower_tilebuffer.c, agx_bg_eot.c. Needed: tile memory in `frag.rs` (per-pixel, per-sample, render-target layouts), then a hardware comparison through a GLES render with a known output. |
| `zs_emit`, `sample_mask` | decoded; no semantics. Depth/stencil and coverage outputs: a hardware probe that writes gl_FragDepth / gl_SampleMask and reads the target back. |
| `st_var`, `st_var_final`, `no_var` | vertex-stage varying stores (`st_vary`); `no_var` ends the SIMD group (already costed as control). Needs a vertex-stage dispatch harness. |
| `iter` forms with sample / centroid | centroid = pixel center for covered pixels (assumption); `sample` uses `samplepos` from the setup file. The MSAA sample positions and centroid movement are open (probe: gles_varying with samples 4 and a triangle edge). |

Unverified assumptions (see the header of rust/src/frag.rs): lane -> pixel layout, the f32 evaluation order, the coefficients' sign and
unit conventions of the rasterizer setup (the simulator takes them as input, as it takes uniforms).

Cost model: `iter`, `iterproj`, `ldcf`, `ld/st_tile`, `image_write_block`, `st_var*` sit in the generic load/store classes of cost.rs with
no measurement. Next: time fragment shaders with N varyings / N tile accesses through gles_raster-style probes and add measured classes.

Method that worked: capture the shader Mesa compiles for a probe (`agx_capture.py`), run the same bytes in the simulator with a setup
derived from the probe, compare the readback.
