# Shader capture examples (scripts/agx_capture.py)

Produced on the M2 Max (G14C B1): GLES through Mesa 25.1.9, Vulkan through Mesa 25.3.6. See
[docs/shader-capture.md](../../docs/shader-capture.md).

| file | command |
| --- | --- |
| `listing-gles-private.txt`, `capture-gles-private.json` | `make capture-demo` (`gles_compute private 64`, with `--ir`) |
| `listing-gles-private-loop1024.txt` | same cache, `--loop-trips 1024` (the private-array fill loop really runs 1024 times) |
| `listing-gles-index.txt` | `agx_capture.py --ir -- gles_compute index 64 /dev/dri/renderD128` |
| `listing-vulkan-index.txt` | `agx_capture.py --blob exported.bin`, where `exported.bin` is the `vkGetShaderBinaryDataEXT` output of `vulkan_compute export index.spv 64 ...` and `vulkan-index.comp` is the shader |

The listings were regenerated after the review fixes (IR matching on stage, workgroup size,
final-IR instruction count, stack size and shaderdb stats; `stack_adjust` class; loop trip count
warning). The GLES runs were repeated with `--ir`, which now uses `AGX_MESA_DEBUG=shaders,shaderdb`.
Cycle counts come from the revised `scripts/agx_cost.py` (real cycles; see
novel-results/2026-10-11-cost-model): listings were regenerated offline from the saved caches and IR dumps
(`--cache`, `--ir-dump`, `--blob`) after the model revision. Treat them as model output, not measurements;
store costs use the repeated-address store measurement and are likely pessimistic for streaming stores.
