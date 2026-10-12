# SPIR-V workloads on AGX: shared procedure

## Quick start on the M2 (Asahi Linux)

```sh
make asahi-verify            # = scripts/asahi_verify.sh: preflight, build, record every workload on the GPU, verify
git add spirv/expected results/*-asahi-verify && git commit -m 'Asahi verification: SPIR-V workloads recorded on the M2'
```

None of these tests counts as verified yet. Until the M2 has run them, `make spirv-verify` reports every workload as
**PENDING**. Passing the numpy reference (lavapipe, or the simulator on shim-compiled code) only shows that the shader
and the reference agree; that is a sanity check, not hardware verification.

`asahi-verify` does the following:

1. Runs each workload on the GPU and stores what it produced in `spirv/expected/<workload>/` (format:
   `spirv/expected/README.md`). That includes the driver's own compiled code from that machine.
2. Runs the simulator on that exact code and compares it with the GPU word for word. Each workload is one of:

   | Status | Meaning |
   |---|---|
   | PASS | Identical to the GPU |
   | CLOSE | Differs, but both are within the reference tolerance |
   | FAIL | A difference to investigate |
   | STALE | The recording no longer matches the workload's inputs |

3. Also compares the large-size GPU time with the cost model's prediction.

After the recordings are committed, `make spirv-verify` repeats the comparison offline on any machine.
The rest of this file is the detailed procedure, for investigating results.

## Workloads

Seventy-nine realistic compute kernels, written in GLSL (`spirv/shaders/*.comp`) and compiled to SPIR-V:

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
| `fp16` | Half-precision fma, multiply-subtract, fp32→fp16 conversion | fp16 ALU, 16-bit loads and stores |
| `int64` | splitmix64 hash, signed 64-bit product and shift | 64-bit multiply lowering, carries across halves |
| `bitops` | bitfieldExtract/Insert, findLSB/MSB, bitCount, bitfieldReverse, rotates, mulExtended, uaddCarry | Bit-field ALU, 32×32→64 products |
| `intdiv` | Unsigned and signed division / modulo by variables and constants, int↔float conversions | Division lowering, convert |
| `compact` | Stable stream compaction, 3 passes | subgroupBallot, ballot bit counts, broadcast, a serial single-group pass |
| `spmv` | CSR sparse matrix × vector, one thread per row | Irregular gathers, data-dependent trip counts |
| `transpose` | 32×33 padded shared-memory tile transpose, edge tiles | 2-D workgroups, threadgroup memory, bounds |
| `mandelbrot` | Q13 fixed-point iteration counts | Divergent loops with per-lane trip counts (exact integers) |
| `softmax` | Row softmax, shared-memory max and sum trees | exp, barriers in loops, reductions |
| `rgba8` | unpack/packUnorm4x8 tone curve, luma through 8-bit storage | unorm conversions, byte stores |
| `atomics` | Every device atomic with returns, contended and distinct words; threadgroup counter and max | Atomic return values, contention, shared atomics |
| `geom3d` | 4×4×2 workgroups over a ragged 3-D grid, every built-in id written | 3-D ids, partial workgroups |
| `spill` | 96-word private array with data-dependent indices | Register spills: `stack_store` / `stack_load` |
| `control` | 10-way switch, nested break / continue, early returns, a called function | Branch-heavy integer control flow |
| `shuffle` | subgroupShuffle / Xor / Up / Down / Broadcast | Cross-lane shuffles (reference follows the reported subgroup size) |
| `floatops` | min / max / clamp / abs / sign / floor / ceil / fract / round / trunc / step / compares on special values | Exact float semantics (±0, ±inf, ties) |
| `radix_sort` | 4-bit LSD radix sort of (key, value) pairs, 8 passes × 3 dispatches, stable | Histograms, a single-group scan, stable scatter |
| `fft` | 1024-point complex radix-2 FFT per workgroup in shared memory | sin / cos twiddles (`sin_pt`), unrolled butterflies |
| `crc32` | CRC-32 of 64-byte messages through a shared-memory table | Data-dependent threadgroup loads, byte extraction |
| `kmeans` | Nearest-centroid assignment (k centroids in shared memory) | Float distance argmin, shared-memory broadcast |
| `int8dot` | int8 × int8 → int32 matrix product through 8-bit storage | Sign-extending byte loads, integer multiply-add |
| `stencil3d` | 7-point Jacobi step on a 3-D grid, 8×8×4 workgroups | 3-D addressing, clamped neighbours |
| `int16` | int16 / uint16 wrapping arithmetic, shifts, min / max, widening | 16-bit integer ALU and storage |
| `trig` | sin, cos, asin, acos, atan, atan2, exp, exp2, log, log2, sqrt, inversesqrt | Transcendental lowerings, against Vulkan's precision bounds |
| `raytrace` | Ray tracer: 3 spheres, checkered ground, hard shadows, one mirror bounce, RGBA8 image (1920×1080 large) | Divergent shading, sqrt / normalize, packUnorm; writes a viewable `.ppm` |
| `mlp` | Two-layer perceptron inference (128 → 256 ReLU → 16 softmax), one workgroup per input | Matrix-vector FMA chains, shared-memory activations, exp |
| `sobel` | Sobel edge detection on an 8-bit image (3840×2160 large), shared-memory halo tile | Byte loads / stores, integer stencil (exact); writes a viewable `.pgm` |
| `life` | Conway's Game of Life on a toroidal grid, one dispatch per generation (16 / 64) | Byte stencil, ping-pong buffers; exact; writes a `.pgm` |
| `bfs` | Level-synchronous breadth-first search on a random graph (1 Mi vertices large), 24 / 32 dispatches | atomicMin, atomic counters under divergence (Mesa aggregates them with SR58 / SR59); exact |
| `conv_layer` | CNN layer: 3×3 convolution, 16 → 32 channels, bias, ReLU (224×224 large), 3-D dispatch | FMA chains, zero-padding branches; error bound |
| `hist_eq` | Histogram equalisation of an 8-bit image: histogram, CDF / lookup table, remap (3 dispatches) | Shared and device atomics, integer division, byte remap; exact; writes a `.pgm` |
| `aes` | AES-128 encryption, one block per thread, S-box in shared memory (16 MiB large) | Byte table lookups, xor / shift chains; exact (reference checked against FIPS-197) |
| `hash_table` | Open-addressing hash table: insert with atomicCompSwap, then lookups (1 Mi keys large) | Compare-and-swap, probing loops; exact lookups |
| `word_count` | Words, lines and word-length histogram of a 16 MiB text | Byte loads, data-dependent loops, shared atomics; exact |
| `dct8` | JPEG forward DCT + quantisation of 8×8 blocks of a 4K image | cos, shared-memory block, round; ±1 only at .5 ties |
| `attention` | Scaled dot-product attention (64 Ki queries × 256 keys, d = 64 large) | Dot products, two shared-memory reduction trees, exp |
| `resize` | Bilinear RGBA8 upscaling (1280×720 → 3840×2160 large) | Fractional coordinates, floor, mix, unorm pack; ±1 level; writes a `.ppm` |
| `median3` | 3×3 median filter (19-exchange min/max network) on a noisy 4K image | Integer min / max, shared halo tile; exact; writes a `.pgm` |
| `poisson` | Jacobi iterations of a 2-D Poisson solve (1024², 200 dispatches large) | Many small dispatches, ping-pong buffers, boundary branches |
| `lbm` | Lattice-Boltzmann fluid flow (D2Q9 BGK, periodic 512², 200 steps large) | 9-direction gathers, division, many dispatches; mass conserved |
| `base64` | Base64 encoding of 12 MiB (both '=' padding cases) | Byte loads / stores, select chains; exact |
| `md5` | MD5 of 256 Ki single-block messages | Integer rotates, branches per round group; exact |
| `gemv` | Dense matrix-vector product, 8192² fp32, one workgroup per row | Memory bandwidth, shared-memory tree reduction |
| `levenshtein` | Edit distance of string pairs, a dynamic program per thread | Dynamically indexed private array (register selects) |
| `blur` | Separable 9-tap Gaussian blur, 4K fp32, two passes | Shared-memory tiles, fp32 FMA, multi-dispatch |
| `yuv420` | RGBA8 → NV12 colour conversion at 4K | 8-bit stores, integer multiply-add, arithmetic shifts |
| `montecarlo` | Monte Carlo π from a PCG hash | Integer hashing, subgroup reduction, elected atomics |
| `xxhash` | xxHash32 of 1036-byte records | 32-bit multiply-rotate rounds, four independent chains |
| `bloom` | Bloom filter insert and query | `atomicOr`, integer modulo, two dependent dispatches |
| `collatz` | 3n + 1 trajectories in uint64 | 64-bit integer arithmetic, data-dependent loop lengths |
| `tridiag` | Batched tridiagonal solves (Thomas algorithm) | fp32 division, private arrays on the stack |
| `merge` | Merge-path merge of sorted arrays | Binary searches with data-dependent addresses |
| `ccl` | Connected-component labelling (308 dispatches) | `atomicMin` in place, long dispatch chains |
| `pagerank` | PageRank power iteration on a CSR graph | Irregular gathers, fp32 accumulation, ping-pong buffers |
| `skinning` | Linear blend skinning of mesh vertices | mat4 arithmetic, byte-packed indices, vec4 loads and stores |
| `strsearch` | Substring search over 64 MiB | Unaligned byte loads, `atomicAdd` / `atomicMin` |
| `quantize` | Per-row int8 quantization (ML) | `roundEven`, float-to-int, int8 stores, shared atomics |
| `rule110` | Rule 110 cellular automaton, 1000 dispatches | Bit-packed shifts across words, launch overhead |
| `voxel` | Point-cloud voxelization | Scattered `atomicAdd` with heavy contention in dense voxels |
| `nms` | Non-maximum suppression of detection boxes | Shared-memory tiles, triangular (uneven) work per workgroup |
| `sat` | Summed-area table (integral image) | Workgroup scans with carry, strided column pass |
| `sssp` | Bellman-Ford shortest paths, 78 dispatches | `atomicMin` relaxation over CSR edges |
| `morton` | Morton codes for BVH / octree building | 32 and 64-bit bit interleaving |
| `utf8` | UTF-8 validation and decoding | Byte classification, data-dependent lookahead, atomics |
| `fir` | 63-tap audio FIR filter | Shared-memory sliding window, fp32 FMA chains |
| `composite` | Alpha compositing (Porter-Duff over) | 8-bit channel arithmetic, exact division by 255 |
| `nqueens` | N-queens solution counting | Bitmask search, explicit private stack, heavy divergence |
| `lzmatch` | LZ77 match finding (compression) | Nested data-dependent byte loops |
| `bitpack` | 7-bit value packing into a bit stream | `atomicOr` across word boundaries |
| `perlin` | Procedural terrain from Perlin noise | Table lookups, `floor`, polynomial fades, fp32 |
| `adler32` | Adler-32 checksums (zlib) | Byte sums in long serial chains, integer modulo |

Each workload has two sizes:

- **small:** for the simulator.
- **large:** for timing. 3–470 ms per run on lavapipe (CPU).

The eight task files group them:

- `spirv-01-reduce-scan.md`
- `spirv-02-sgemm-conv2d.md`
- `spirv-03-histogram-bitonic.md`
- `spirv-04-nbody-sha256.md`
- `spirv-05-isa-coverage.md` (integer, fp16, 64-bit, 8-bit and subgroup features the first eight do not reach)
- `spirv-06-semantics.md` (atomics, 3-D geometry, spills, control flow, shuffles, exact float operations)
- `spirv-07-algorithms.md` (radix sort, FFT, CRC-32, k-means, int8 GEMM, 3-D stencil, int16, transcendentals)
- `spirv-08-applications.md` (a ray tracer, neural-network inference, edge detection: end-to-end results a person can look at)

Every workload passes its numpy reference on Mesa lavapipe (CPU Vulkan), at both sizes. That only shows the shaders and
references are right; a wrong result on the M2 is therefore a GPU, driver or runner finding, not a shader bug.

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

   - Offline, all seventy-nine workloads pass their numpy references with shim-compiled Mesa 25.2.8 and 26.0.3 code
     (`make spirv-sim`). That is a sanity check; verification is `make spirv-verify` against the M2 recordings.
   - Integer workloads (`scan`, `histogram`, `bitonic`, `sha256`, `int64`, `bitops`, `intdiv`, `compact`, `transpose`,
     `mandelbrot`, `atomics`, `geom3d`, `spill`, `control`, `shuffle`, `floatops`, `radix_sort`, `crc32`, `int8dot`, `int16`, `sobel`,
     `life`, `bfs`, `hist_eq`, `aes`, `hash_table`, `word_count`, `median3`, `base64`, `md5`, and reduce's uint half) must match the hardware word for word.
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
