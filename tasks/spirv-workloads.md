# SPIR-V workloads on AGX: shared procedure

Eight realistic compute kernels, written in GLSL (`spirv/shaders/*.comp`) and compiled to SPIR-V:

| Workload | Kernel | Stresses |
|---|---|---|
| `reduce` | Two-stage float + uint sum (subgroupAdd, shared memory) | Subgroup reductions, barriers, memory bandwidth |
| `scan` | Three-pass exclusive prefix sum | subgroupExclusiveAdd, multi-dispatch, read-modify-write |
| `sgemm` | 64×64-tiled fp32 GEMM, 4×4 register blocking | FMA throughput, shared-memory reuse, register pressure |
| `conv2d` | 5×5 convolution with a halo tile | Shared-memory reuse, clamped addressing, FMA |
| `histogram` | 256-bin byte histogram | Threadgroup and device atomics under contention |
| `nbody` | All-pairs gravity step | rsqrt / transcendental pipe, shared-memory broadcast |
| `bitonic` | Bitonic sort of (key, value) pairs; 6 (small) / 66 (large) dispatches | Data-dependent swaps, barriers, dispatch overhead |
| `sha256` | One SHA-256 block per thread | Integer ALU (rotates, adds, bit ops), register arrays |

Each workload has two sizes:

- **small:** for the simulator.
- **large:** for timing. 7–470 ms per run on lavapipe (CPU).

The four task files group them:

- `spirv-01-reduce-scan.md`
- `spirv-02-sgemm-conv2d.md`
- `spirv-03-histogram-bitonic.md`
- `spirv-04-nbody-sha256.md`

Every workload already passes its numpy reference on Mesa lavapipe (CPU Vulkan), at both sizes. A wrong result on the
M2 is therefore a GPU, driver or runner finding, not a shader bug.

## Tools

| Command | What it does |
|---|---|
| `scripts/spirv_task.py build` | glslangValidator + spirv-val for every shader. Also builds `spirv/build/vk_run` from `tests/vk_run.c`. |
| `scripts/spirv_task.py run W\|all --size small\|large` | Runs the workload with `vk_run` and writes `spirv/out/<W>-<size>/`. See below. |
| `scripts/spirv_task.py extract shader.bin` | Splits the exported MesaVkShaderBin into `main.bin`, `preamble.bin` and their listings. Uses the Rust decoder; applegpu is not needed. |
| `scripts/spirv_task.py sim W --code-dir DIR --hardware spirv/out/W-small` | Runs the driver's own preamble and main on the functional simulator (`rust/target/release/agx_run`). Carries buffers across dispatches, checks the result against the same reference, and diffs it word for word against the hardware output. |
| `scripts/spirv_task.py predict W --code DIR/main.bin` | Whole-dispatch timing from the cost model (`agx_dispatch`). |

`run` writes these files into `spirv/out/<W>-<size>/`:

- `result.json`: functional verdict, GPU time per dispatch (median over repeats), GFLOP/s or GB/s;
- `shader.bin`: the driver's compiled binary, via VK_EXT_shader_object;
- `props.json`: pipeline statistics and IR, if the driver supports VK_KHR_pipeline_executable_properties;
- the spec, inputs and read-back outputs.

## Procedure on the M2 (Asahi Linux)

1. **Install the tools and build.**

   ```sh
   dnf install glslang spirv-tools vulkan-headers vulkan-loader-devel
   make rust spirv-build
   ```

2. **Run and save the outputs.** Run all workloads at both sizes and keep each `result.json`:

   ```sh
   python3 -B scripts/spirv_task.py run all --size small --repeat 5
   python3 -B scripts/spirv_task.py run all --size large --repeat 20
   ```

   - `vk_run`'s first JSON line records the device (`"device"`); it must be the AGX, not llvmpipe.
   - Choose another device with `VK_RUN_DEVICE=<index>`.

3. **Extract the compiled code**, for each workload (small size):

   ```sh
   python3 -B scripts/spirv_task.py extract spirv/out/sgemm-small/shader.bin --out spirv/out/sgemm-small/code
   ```

   - Check `code/main.txt` against the driver's own disassembly where one is available: `AGX_MESA_DEBUG=shaders` / `ASAHI_MESA_DEBUG` (the variable depends on the Mesa version), or `props.json`.
   - If `segments.json` says `layout_as_mesa_25_3_6: false`, confirm the code region by hand.
   - Any undecodable instruction in main or preamble is a decoder gap. Report it with its bytes.

4. **Pin down the root-table layout once.** This is a one-time investigation; record what you learn in `spirv/root_layout.json`.

   - The preamble loads the root table (u0_u1) and descriptor memory into uniforms. `spirv/root_layout.example.json` has the starting point, taken from the stored control shader.
   - The real layout comes from Mesa's `hk_root_descriptor_table` (`src/asahi/vulkan/hk_cmd_buffer.h` for your Mesa version) and `hk_descriptor_set` storage-buffer descriptors.
   - Verify it with a one-binding probe shader: write `gl_GlobalInvocationID`, push constants and `gl_NumWorkGroups` to a buffer. `sim` must then match the hardware word for word.

5. **Simulate functionally:**

   ```sh
   python3 -B scripts/spirv_task.py sim sgemm --code-dir spirv/out/sgemm-small/code --hardware spirv/out/sgemm-small
   ```

   - Integer workloads (`scan`, `histogram`, `bitonic`, `sha256`, and reduce's uint half) must match the hardware word for word.
   - Float workloads should match too, since the simulator implements the hardware's float semantics. Every differing word is a simulator bug, unless it is in an fp32 transcendental (rcp/rsqrt/log2/exp2/sin), which is task #5's known gap.
   - When `agx_run` stops on an unimplemented instruction or format, that is the finding: report it, and implement it in `rust/src/func.rs` / `fsem.rs` if the semantics are clear.

6. **Predict performance and compare:**

   ```sh
   python3 -B scripts/spirv_task.py predict sgemm --code spirv/out/sgemm-small/code/main.bin --size large \
       --regs <GPRs from props.json or the binary> --tg-bytes <shared bytes>
   ```

   Compare `predicted_us` with `result.json` → `dispatch_ns_median`. Use the large size, so that launch overhead is small.

7. **Write the report** in `results/<date>-spirv-<group>/README.md`. For each workload, include:
   - hardware vs reference;
   - simulator vs hardware: words differing, and why;
   - measured vs predicted time per dispatch, with the ratio;
   - throughput vs the M2 peaks used elsewhere in this repository.

   Commit the `result.json`, `segments.json` and listings, `sim.json` and the report. Don't commit the large input and output buffers.

## Rules

- No shader edits to "make it pass". A shader change needs a reason, such as a real bug found against lavapipe, and must be re-verified on lavapipe:

  ```sh
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json python3 -B scripts/spirv_task.py run all --size small
  ```

- Timing runs follow the usual GPU-safety practice. These are ordinary Vulkan dispatches through the driver, so they should not hang the GPU. If one does, stop and report it.
