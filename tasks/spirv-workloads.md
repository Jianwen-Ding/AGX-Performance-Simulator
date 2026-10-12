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
| `scripts/spirv_task.py extract shader.bin --out DIR` | Splits the exported MesaVkShaderBin with the driver's own `agx_shader_info`: `main.bin`, `preamble.bin`, `rodata.bin` (constants the driver loads into uniforms), their listings, and `info.json` (program offsets, workgroup size, threadgroup bytes, registers, root-table layout). Uses the Rust decoder; applegpu is not needed. |
| `scripts/spirv_task.py sim W --code-dir DIR --hardware spirv/out/W-small` | Runs the driver's own preamble and main on the functional simulator (`rust/target/release/agx_run`). Carries buffers across dispatches, checks the result against the same reference, and diffs it word for word against the hardware output. |
| `scripts/spirv_task.py predict W --code-dir DIR --size large` | Whole-dispatch timing of every dispatch (`agx_dispatch`), with the SIMD groups' paths from the functional simulator over the real inputs. |
| `scripts/spirv_task.py compile all --out DIR --mesa SRC` | Compiles the workloads with Mesa's AGX compiler and keeps only the binaries (off the M2: through the no-op drm-shim, `make spirv-compile`). |

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

   - `segments.json` says `"layout": "agx_shader_info"` when the blob had a known layout (Mesa 25.2-26.0 checked;
     25.3.6 has the same 72-byte header and code offset 0x8bc as 25.2). `"scan"` means it fell back to guessing the
     programs: report that, with the blob.
   - Compare `code/main.txt` with the offline build of the same shader, `spirv/compiled/mesa-26.0.3/sgemm/main.txt`
     (and 25.2.8). Differences are compiler-version changes; note any instruction the M2's compiler emits that the
     offline ones never do.
   - Any undecodable instruction in main or preamble is a decoder gap. Report it with its bytes.

4. **Check the root-table layout once.** `extract` picks it from the preamble (`info.json` → `root_layout`):
   `spirv/root_layout.mesa-26.0.json` when the preamble reads the set address from the preloaded `u2_u3` (Mesa >= 25.3),
   `spirv/root_layout.mesa-25.2.json` otherwise. Both come from `offsetof()` on built drivers; 25.3.6 is expected to
   match 26.0 (the stored control shader reads base_group from words 20..22 and push word 2 from word 124).

   - Verify it with a one-binding probe shader: write `gl_GlobalInvocationID`, push constants and `gl_NumWorkGroups`
     to a buffer. `sim` must then match the hardware word for word. If it does not, write the corrected layout to
     `spirv/root_layout.mesa-25.3.json` and pass it with `--layout`.

5. **Simulate functionally:**

   ```sh
   python3 -B scripts/spirv_task.py sim sgemm --code-dir spirv/out/sgemm-small/code --hardware spirv/out/sgemm-small
   ```

   - Offline, all eight workloads already pass their references with Mesa 25.2.8 and 26.0.3 code (`make spirv-sim`).
   - Integer workloads (`scan`, `histogram`, `bitonic`, `sha256`, and reduce's uint half) must match the hardware word
     for word.
   - Float workloads only need to be close: within the workload's reference tolerance. Last-bit float differences
     (fp32 transcendentals, summation order) are not worth chasing.
   - When `agx_run` stops on an unimplemented instruction or format, that is the finding: report it, and implement it
     in `rust/src/func.rs` / `fsem.rs` if the semantics are clear.

6. **Predict performance and compare:**

   ```sh
   python3 -B scripts/spirv_task.py predict sgemm --code-dir spirv/out/sgemm-small/code --size large
   ```

   - The launch (workgroup size, registers, threadgroup bytes) comes from `info.json`; the SIMD groups' paths and
     addresses come from the functional simulator over the workload's inputs.
   - Compare `dispatch_us` with `result.json` → `dispatch_ns_median`, per dispatch. Use the large size, so that launch
     overhead is small.
   - The offline predictions from the 26.0.3 code are in `results/2026-10-12-spirv-offline/`: compare them too.

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
