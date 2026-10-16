# Task safe-05: texture and image instructions through Mesa

Read [safe-probes.md](safe-probes.md) first.

**Goal:** semantics of `texture_sample`, `texture_load`, `image_write` and the texture-state forms, for CUDA texture
and surface objects.

These are **not** safe to hand-encode: they need valid descriptor state, and they are listed "open: needs
texture/sampler/image state" in the playground's `docs/isa-coverage.md`. They are safe when Mesa compiles the shader
and the Vulkan driver builds the state, which is how `texfetch`, `texfilter`, `imgerode` and `imgtonemap` already run.
On the GPU those four match their references; the simulator reports them as UNSIM.

## Steps

1. **Encodings.**
   - Write small GLSL compute shaders (`spirv/shaders/`), one feature each:
     - `texelFetch` on 1D / 2D / 3D / array;
     - `texture` with nearest and linear filtering;
     - every address mode;
     - LOD and gradient forms;
     - normalised and unnormalised coordinates;
     - the formats CUDA uses (r8 / r16 / r32 unorm, snorm, uint, sint and float, 1-4 channels);
     - `imageLoad` / `imageStore`.
   - Export each shader's AGX binary (`vk_run` `export` / `props`) and decode it with `agx_asm`. List each
     instruction's fields as Mesa sets them.
2. **Semantics on the GPU.**
   - Run each shader with `vk_run` over textures holding every texel value of a format (exhaustive for 8 / 16-bit
     formats).
   - Probe coordinates at and between texel centres, at the edges and out of range.
   - Questions:
     - Filtering precision (the `texfilter` probe found 10 sub-texel bits).
     - Rounding of unorm / snorm / sRGB conversion.
     - Border colours.
     - Out-of-range integer coordinates.
3. **Simulator.**
   - Implement `texture_load` / `texture_sample` / `image_write` in `rust/src/func.rs`, with a texture-descriptor
     model taken from Mesa's descriptor layout.
   - Turn the 5 UNSIM workloads into PASS.
4. **CUDA mapping.** A table of `tex1D` / `tex2D` / `tex3D` / `tex2DLayered`, `tld4` and `surf*` against AGX
   instructions and descriptor fields, noting unsupported modes.

## Safety notes

- Never edit texture-state words by hand.
- Only change GLSL and the Vulkan sampler / image create-info that Mesa validates.
- A shader the driver rejects is a host-side error, not a GPU hang.
