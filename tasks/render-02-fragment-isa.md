# Task: fragment-stage instructions in the functional simulator and cost model

Goal (2026-11-14): decode and model the raster-path instructions the compute work never exercised. All 26 open descriptors
in the playground's isa-coverage that are fragment-only. The disassembler already decodes them; `agx_run` did not execute them
(they fell through to applegpu's emulation, which has no semantics for them).

| Instruction | Status |
|---|---|
| `iter`, `iterproj` | **Executed** (`fsem::frag_iter`, `frag.rs`, `agx_run --frag FILE`): channel k = plane(I+k) at the lane's pixel; `iterproj` divides by plane(J) (the 1/w plane). Checked against a hardware render: 32 lanes x 4 channels within 5e-7 (`scripts/frag_iter_check.py`, results/2026-11-14-frag-iter). fp16 destination (16-bit register tuple) rounds RNE. |
| `ldcf` | **Executed**: the (A, B, C) triple of each channel's coefficient register (flat varyings: Mesa compiles flat to `ldcf` + ALU). Not yet compared to hardware. |
| `wait_pix`, `signal_pix` | **Executed** as functional no-ops (pixel order is not simulated; they stay costed as barriers). Mesa emits `wait_pix 1, 0` before `sample_mask` / `zs_emit` and masks 0x8 / 0xC around tile loads / stores. Open: cost, what `i` / `j` select. |
| `st_tile`, `ld_tile` | **Executed** (`fsem::frag_tile`, `frag::TileState`): byte-addressed tile memory per (pixel, sample), components compacted by the mask, formats i8 / i16 / i32 / f16 / u8norm / s8norm / u16norm / s16norm, writes only to live samples. **Checked on hardware**: `iterproj` + `st_tile u8norm` equals an RGBA8 render byte for byte (including clamped channels); `st_tile` -> `ld_tile` round trip. **rgb10a2 and srgba8 checked on hardware, byte exact** (32 pixels each, clamping and the sRGB curve included); rg11b10f and rgb9e5 (and their .rtz forms) are implemented from the formats' definitions but unverified (Mesa stores rg11b10f through a software-packed `i32` tile, so the hardware format is not exercised by GLES). Explicit coordinates (the 16-bit register pair C, tile-local; checked with a program only) and multi-sample tiles (`samples N`, `coverage X Y MASK` in the setup file: initial live mask and sr60 / sr124) are executed; no hardware comparison of per-sample coverage. |
| `sample_mask` | **Executed** (`fsem::frag_sample_mask`): TARGET samples in LIVE stay live and count as tested, the rest are killed (Mesa's documented semantics). **Checked on hardware**: Mesa's discard sequence over a 64x64 target, 1446 / 1446 triangle pixels agree on drawn vs discarded. Open: multi-sample coverage (the probe compiles a different epilogue with MSAA). |
| `zs_emit` | **Executed** (`fsem::frag_zs_emit`): records depth (f32) / stencil (u16) of the live samples in the mask; the depth/stencil test outcome is not modelled (always passes). Not compared to hardware (the instruction sits in the driver's epilogue, which `agx_capture` does not split out). |
| `st_var`, `st_var_final` | **Executed** (`fsem::frag_st_var`): recorded per (group, lane, slot) in `TileState::uvs` (dumped by `--tile-dump`). Not run through a vertex harness yet. `no_var` stays a control op (ends the SIMD group). |
| `image_write_block` | **Executed** (`func::image_write_block`): each lane copies its pixel's tile bytes (offset register R, tile format F, one component per image channel) into the PBE image of texture state T at the explicit coordinates C, or at its own pixel; no conversion (tile format size must equal the image element size). Covered by a unit test through the real tex_state / PBE path (`st_tile_then_image_write_block`); no hardware comparison yet (the colour-resolve programs run inside the driver). |
| `iter` forms with sample / centroid | centroid = pixel center for covered pixels (assumption); `sample` uses `samplepos` from the setup file. The MSAA sample positions and centroid movement are open (probe: gles_varying with samples 4 and a triangle edge). |

Unverified assumptions (see the header of rust/src/frag.rs): lane -> pixel layout, the f32 evaluation order, the coefficients' sign and
unit conventions of the rasterizer setup (the simulator takes them as input, as it takes uniforms).

Cost model: `iter`, `iterproj`, `ldcf`, `ld/st_tile`, `image_write_block`, `st_var*` sit in the generic load/store classes of cost.rs with
no measurement. Next: time fragment shaders with N varyings / N tile accesses through gles_raster-style probes and add measured classes.

`agx_run --tile-dump FILE` writes the final tile memory, live / tested masks, depth / stencil and vertex outputs as text. Checks: `scripts/frag_iter_check.py`, `scripts/frag_tile_check.py` (data in results/2026-11-14-frag-iter).

Method that worked: capture the shader Mesa compiles for a probe (`agx_capture.py`), run the same bytes in the simulator with a setup
derived from the probe, compare the readback.
