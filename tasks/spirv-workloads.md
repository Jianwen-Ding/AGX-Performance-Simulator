# SPIR-V workloads on AGX: shared procedure

## Quick start on the M2 (Asahi Linux)

```sh
make asahi-verify            # = scripts/asahi_verify.sh: preflight, build, record every workload on the GPU, verify
git add spirv/expected results/*-asahi-verify && git commit -m 'Asahi verification: SPIR-V workloads recorded on the M2'
```

After the run, read `tasks/m2-run-checklist.md` first: it lists the questions the recording settles and the change
attached to each outcome.

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
   | UNSIM | Texture / storage-image / indirect-dispatch workloads only: the GPU's result matches the reference; the simulator does not model that path |
   | NODEV | The device's driver does not offer a feature the shader needs (`vk_run` `require`); nothing was run |

3. Also compares the large-size GPU time with the cost model's prediction. The last line of `verify.txt` sums this
   up as `"timing"`:
   - the median and p90 error factor (predicted / measured, either direction);
   - how many workloads fall within ±10%, ±25% and 2×;
   - the five most under-predicted and the five most over-predicted workloads.
   Workloads with 2 to 16 dispatches also list `per_dispatch_us` (measured, predicted) per dispatch.

   That one line is the model's accuracy on real kernels. `nms` is expected near the top of the under-predicted
   list (see `spirv-08-applications.md`).

4. Runs `spirv_task.py probes`, which gives `probes.txt` / `probes.json` and the report's Probes section. It reads
   the recordings of the timing probes and lists measured against predicted per dispatch, together with what each probe
   measures:
   - `barriercost`: µs per barrier round;
   - `atomcontend`: hot-counter over spread-counter time;
   - `smembank` / `coalesce`: time per stride relative to stride 1;
   - `aluprobe`: dependent over 8-chain ratio for fp32, fp16 and integer;
   - `rule110`: µs per dispatch.

   It also lists the device measurements (`texfilter` sub-texel bits, `imgtonemap` rounding, `mplitmus` relaxed stale
   reads, `f16pack` subnormal results) and the features the driver lacks (NODEV workloads). These are the numbers to
   bring back for the next model revision.

After the recordings are committed, `make spirv-verify` repeats the comparison offline on any machine.
The rest of this file is the detailed procedure, for investigating results.

## Workloads

One hundred and sixty-five realistic compute kernels, written in GLSL (`spirv/shaders/*.comp`) and compiled to SPIR-V:

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
| `sieve` | Segmented prime sieve | Shared-memory `atomicOr`, popcount, nested strided loops |
| `gcd` | Binary GCD | `findLSB`, data-dependent loop lengths |
| `dither` | Ordered (Bayer) dithering to 1 bit | Constant-table lookups, bit packing |
| `advect` | Semi-Lagrangian fluid advection, 40 dispatches | Clamped gather, bilinear fp32 interpolation |
| `knn` | k-nearest neighbours (k = 8) | Shared tiles, per-thread insertion sort in registers |
| `csvparse` | CSV integer field parsing with validation | Character classification, decimal accumulation |
| `pip` | Point-in-polygon (winding number) | 64-bit integer cross products, long uniform loops |
| `lift53` | JPEG 2000 5/3 integer wavelet | Shared-memory lifting steps, arithmetic shifts |
| `chacha20` | ChaCha20 stream cipher | 32-bit add / xor / rotate rounds, 16 live state words |
| `modexp` | Modular exponentiation | 64-bit multiply and software 64-bit modulo |
| `floyd` | Floyd-Warshall APSP, 512 dispatches | Long dependent dispatch chain, in-place min-plus |
| `f16pack` | fp32 ↔ fp16 conversion | `packHalf2x16` rounding, fp16 subnormals and infinities |
| `raster` | Software triangle rasterizer | Integer edge functions, nested pixel loops, contended `atomicMin` |
| `canny` | Canny edge non-maximum suppression | Integer gradients, sector logic, two dependent dispatches |
| `secded` | ECC encode / correct / detect | Bit extraction, popcount parity, syndrome branches |
| `bicubic` | Bicubic image upscaling | 4 × 4 clamped gathers, cubic weights, fp32 |
| `dfa` | Aho-Corasick keyword search | Table-driven state machine, shared-memory byte lookups |
| `b64decode` | Base64 decoding with validation | Character classification, partial byte stores |
| `cordic` | Fixed-point CORDIC sin / cos | Arithmetic shifts, sign-dependent add / subtract |
| `rsenc` | Reed-Solomon RS(255,223) encoder | GF(2^8) log / exp lookups, 32-register LFSR |
| `texfetch` | `texelFetch` from an r32f texture | The texture unit (unfiltered), bit-exact; hardware-only (UNSIM) |
| `texfilter` | Hardware bilinear filtering of an RGBA8 texture | The texture unit's filtering and sub-texel precision; hardware-only (UNSIM) |
| `imgerode` | Morphological erosion on r32ui storage images, 4 ping-pong dispatches | `imageLoad` / `imageStore` (`texture_load` / `image_write`), exact; hardware-only (UNSIM) |
| `imgtonemap` | In-place Reinhard tone mapping of an RGBA8 storage image | unorm load / store conversion; hardware-only (UNSIM) |
| `argmax64` | Per-bucket argmax by 64-bit packed-key `atomicMax` | `GL_EXT_shader_atomic_int64` on buffers; requires `buffer_int64_atomics` |
| `minmax64` | Min / max / xor with shared then global 64-bit atomics | 64-bit atomics on shared memory and buffers |
| `dsum` | fp64 dot-product tree reduction | `shaderFloat64`; requires `float64` (NODEV on the offline Asahi driver) |
| `quadmip` | 2 × 2 mipmap downsampling with quad operations | `quad_shuffle` / `quad_shuffle_xor`, subgroup ids |
| `clustered` | Clustered subgroup add / max over 2, 4, 8 lanes | `simd_shuffle` trees and `quad_iadd` |
| `specblur` | Box blur with radius and workgroup size as specialization constants | `vk_run` `specialize`; constants folded by the compiler (simulated) |
| `cullindirect` | Frustum culling → dispatch arguments → `vkCmdDispatchIndirect` | GPU-driven launch; hardware-only (UNSIM) |
| `bigsort` | Bitonic sort of 8192 keys per workgroup | The limits: 1024 invocations, all 32 KiB of shared memory |
| `pushmat` | Vertex transform by four mat4 push constants | A full 256-byte push-constant block |
| `mplitmus` | Message-passing litmus test, release / acquire and relaxed | Vulkan memory model atomics (`device_store` / `device_load` coherence variants, `memory_barrier`) |
| `splat` | Bilinear point splatting with fp32 `atomicAdd` | `VK_EXT_shader_atomic_float`; requires `buffer_float32_atomic_add` (NODEV on the offline Asahi driver) |
| `splatfix` | The same splat with 16.16 fixed-point integer atomics | The portable technique where float atomics are missing |
| `smembank` | Shared-memory stride sweep (1 .. 33), one dispatch per stride | Bank conflicts; a timing probe for the cost model |
| `coalesce` | Device-memory stride sweep (1 .. 64), one dispatch per stride | Coalescing / cache lines; a timing probe for the memory model |
| `atomcontend` | atomicAdd into 1 .. 2^20 counters, one dispatch each | Atomic contention; a timing probe |
| `barriercost` | 0 .. 256 rounds of shared exchange + barriers, one dispatch each | Barrier cost; a timing probe |
| `aluprobe` | fp32 / fp16 fma and integer multiply-add chains, dependent and 8-wide | ALU latency and throughput; a timing probe |
| `ubopalette` | Vertex transform by a dynamically indexed mat4 palette in a 4 KiB uniform buffer | Uniform buffers (`vk_run` `ubuffer`) |
| `uboparams` | Colour correction with all parameters from a small uniform buffer | Uniform buffers read uniformly |
| `bindless` | Lookups through a descriptor array of 8 storage buffers, non-uniform index per lane | `nonuniformEXT` descriptor indexing (simulated: the set is laid out as honeykrisp does) |
| `bindlessu` | The same array indexed per workgroup (dynamically uniform) | Core dynamic descriptor indexing |
| `rle` | Chunked run-length encode (count, serial offsets, write runs) then decode by binary search, 4 passes | Ballot-based run detection, a serial single-thread scan, data-dependent binary search |
| `huffdec` | Canonical Huffman decoding, one 256-symbol block per invocation, 12-bit lookup table in shared memory | Unaligned bit-stream peeks, data-dependent shared-memory lookups, serial per-thread dependency |
| `lookback` | Single-pass prefix sum with decoupled look-back across workgroups | Atomic tile counter, release / acquire flags between workgroups (Vulkan memory model), spin-waiting on earlier workgroups; requires `vulkan_memory_model` |
| `hll` | HyperLogLog cardinality sketch: 64-bit hash per item, per-workgroup shared-memory sketch merged into 4096 global registers | Shared-memory and device `atomicMax`, `findMSB`, integer multiply-heavy hashing |
| `tricount` | Triangle counting: one edge per invocation, merge-intersection of two sorted forward-neighbour lists | Irregular, data-dependent loop lengths (load imbalance within a SIMD group), dependent CSR loads, sparse atomics |
| `sha3` | SHA3-256 (Keccak-f[1600]) of independent 64-byte messages | 64-bit xor / and-not / rotate on 32-bit hardware, a 25-lane state in registers, a 6000-instruction unrolled body |
| `stereo` | Stereo depth: 5×5 census transform, then per pixel the best of 64 disparities by windowed Hamming cost | `bitCount`, 3200 independent loads per pixel (1600 census pairs) from L1-resident rows, an argmin loop |
| `viterbi` | Viterbi decoding of the K = 7 rate-1/2 convolutional code, one frame per workgroup, one state per invocation | Shared-memory add-compare-select with a barrier per step, shared `atomicOr`, a single-thread traceback; its bit test compiles to a masked `bfeil` with an unmasked shift, whose hardware behaviour is still open (applications note 37) |
| `viterbifix` | The same decoder with the traceback bit test written as `bitfieldExtract` with a masked offset | The workaround: correct whichever way that question goes |
| `shiftfuse` | Ten bit-manipulation idioms with variable, explicitly masked shifts, one output column each | Which shift / mask / or fusions (`bfeil`, `bfi`, `extr`) keep the shift mask; per-column mismatch counts |
| `bc1enc` | BC1 (DXT1) texture compression of 4×4 blocks: bounding-box endpoints in RGB565, nearest of 4 palette entries per texel | Integer divides by 255 and 3, a 16-texel block held in registers (140 GPRs, low occupancy), unrolled distance search |
| `lz4dec` | LZ4 block decompression, one 4 KiB block per invocation, safe decoding | Byte-serial parsing with data-dependent lengths, overlapping back-reference copies, device read-modify-write per byte |
| `nttmul` | Negacyclic polynomial multiplication mod 7681 (degree 256) by forward / inverse NTT in shared memory | Integer multiply and `%` by a constant (compiler reciprocal multiply), 8 barrier-separated butterfly stages, strided shared-memory access |
| `topk` | k-th smallest of 1024 keys per workgroup by 4-pass MSB-first radix select | Shared `atomicAdd` histograms with heavy collisions (duplicate-rich keys), `subgroupInclusiveAdd` scans, uniform control flow across 13 barriers |
| `cellsort` | Particle binning by counting sort: count, one-workgroup scan, scatter by returned atomic | Device `atomicAdd` with and without a returned value, hot cells (same-address contention), a 1024-invocation workgroup scan |
| `boruvka` | Minimum spanning tree by Borůvka rounds (best edge by `atomicMin`, hooking, pointer jumping), 120 dispatches | Iterative graph algorithm whose per-dispatch work shrinks round by round; device `atomicMin`; pointer chasing |
| `biquad` | Fixed-point audio filter bank: 4 cascaded Q14 biquads (8th-order low-pass), 64-bit accumulator, 16-bit saturation | 32×32→64 integer multiply-add (`imadd` to a register pair), a long serial recurrence per channel, coalesced interleaved channels |
| `edt` | Exact squared Euclidean distance transform: row scans, then a per-pixel column minimum of g² + Δy² | A dispatch of only 8 workgroups with long serial scans (latency-bound, most of the GPU idle), then a coalesced O(H) inner loop per pixel |
| `mt19937` | Mersenne Twister streams, one per workgroup, 624-word state in shared memory, twist in three parallel phases | A 623-step serial seeding loop on one invocation, 4 barriers per twist with partly idle workgroups, xor-shift tempering |
| `bitap` | Approximate DNA pattern matching (Wu-Manber bitap, ≤ 2 edits), one 1024-position chunk per invocation | Long serial chains of 32-bit shift / and / or, 2-bit unpacking of packed bases (the masked-`bfeil` form, note 37) |
| `cholesky` | Batched Cholesky factorisation of 16×16 SPD matrices, one workgroup per matrix in shared memory | fp32 `sqrt` and division (`rsqrt_special`, `rcp`), 48 barriers per workgroup, shrinking active region |
| `rsshash` | Packet processing: IPv4 header checksum (RFC 1071) and Toeplitz RSS hashes over the TCP 4-tuple and the address pair | Byte swapping, 96 + 64 data-dependent bit tests per packet with a 64-bit-window key extraction, 16-bit ones'-complement folding |
| `poly1305` | Poly1305 MACs (RFC 8439) of random-length messages, 26-bit limbs (donna-32) | 25 32×32→64 multiply-adds per 16-byte block, carry chains, partial-block masking, loop counts that differ per lane |
| `crc32c` | CRC-32C of a 16 MiB buffer: per-chunk table CRCs, then a one-workgroup tree merge with GF(2) shift matrices | 4096-step dependent chains of shared-memory table lookups at low occupancy, then a 12-level barrier-separated tree |
| `siphash` | SipHash-2-4 of 0-255-byte messages, one per invocation | 64-bit add / rotate / xor on 32-bit ALUs (rotations by 32 are free half swaps), loop counts that differ per lane |
| `blake2s` | BLAKE2s-256 (RFC 7693) of 0-255-byte messages, written with loops over the G mix and SIGMA-indexed message words | Dynamic local-array indexing: the state and message live in scratch memory (416 B, 256 GPRs) |
| `blake2sfast` | The same hash with every G call written out (constant indices, generated) | Everything in registers (about 90 GPRs, no scratch): with `blake2s`, the scratch-memory penalty for identical work |
| `q4gemv` | 4-bit quantised GEMV in llama.cpp's Q4_0 block format (LLM token generation) | 8-bit storage-buffer loads (18-byte unaligned blocks), fp16 scale rebuilt from bytes, nibble unpacking, `subgroupAdd` row reduction; requires `storage_buffer_8bit` |
| `attndec` | Single-query attention decode over an fp16 KV cache, one workgroup per head | 16-bit storage loads, `exp`, workgroup max / sum reductions, 16 KiB of scores in shared memory, few workgroups (one per head); requires `storage_buffer_16bit` |
| `attndecsplit` | The same decode split along the sequence (flash-decoding): one workgroup per (head, 256-position chunk), then a merge pass rescaling partial softmaxes | 16× more workgroups for the same work; with `attndec`, what filling the GPU buys |
| `rmsrope` | Fused RMSNorm and rotary position embedding of 2048 tokens' 4096-element hidden vectors | `subgroupAdd` reduction, `inversesqrt`, `pow`, `sin` / `cos` of angles up to 4095 rad (outside the range where Vulkan specifies their precision) |
| `q8dot` | int8 GEMV with packed dot products (`OpSDotKHR` on 4×8-bit words; llama.cpp's Q8 integer path), per-block fp32 scales | `VK_KHR_shader_integer_dot_product`, which AGX lowers to byte sign-extension (`bfi` / `asr`) and `imadd`; requires `integer_dot_product` |
| `isp` | Camera ISP: 12-bit RGGB raw → black level, white balance, bilinear demosaic, 3×3 colour matrix, gamma LUT → RGBA8 (12 MP) | 16-bit storage loads, 13 overlapping neighbour reads per pixel, CFA-position branching, a 4096-entry shared-memory table; all integer, bit-exact |
| `contour` | Marching-squares isolines of an integer height map, saddles resolved by the cell centre, 16.16 fixed-point endpoints | Branchy 16-case classification, 64-bit integer division, sparse output appended with `atomicAdd`, per-workgroup shared-memory histogram |
| `hough` | Hough line transform: every edge point votes for 180 angles with Q14 fixed-point trig, `atomicAdd` into a θ × ρ accumulator | 6.8M device atomics spread over 1M counters, with hot bins at the line peaks |
| `spmvsg` | CSR SpMV, vector kernel: one subgroup per row, `subgroupAdd` row sums, power-law row lengths (up to 2047 nonzeros) | Lanes striding over a row's gathers, subgroup reduction; with `spmv` (one invocation per row), scalar vs vector SpMV |
| `varint` | Protobuf sint32 serialisation: zigzag + LEB128 varints packed back to back (count, scan, scatter) | 8-bit storage-buffer stores at unaligned byte offsets, a shared-memory scan per workgroup, a serial one-invocation offsets pass; requires `storage_buffer_8bit` |
| `bvhrays` | Closest-hit ray traversal of a median-split BVH over a triangle mesh, short stack in shared memory, Möller-Trumbore tests | Pointer-chasing node fetches, per-lane divergent traversal depth and order, `1/x` and cross products, a 12 KiB shared-memory stack |
| `adamw` | One AdamW optimiser step over 4M fp32 parameters, updated in place with both moments | Pure streaming (28 bytes per element), `sqrt` / division, buffers both read and written back |
| `stream` | STREAM (Copy, Scale, Add, Triad) plus a pure read and a pure fill over 64 MiB fp32 arrays, one dispatch each | Memory bandwidth by access kind; `probes` reports measured and modelled GB/s per kind |
| `motionest` | Full-search block motion estimation (an H.264 / HEVC encoder's costliest stage): 16×16 blocks, ±16 pixel window in shared memory, best SAD with a unique tie order | Two shared-memory loads per absolute difference, 1089 candidates over 256 invocations (the fifth round has 65 lanes busy), tree min-reduction |

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
| `scripts/spirv_task.py predict W --code-dir DIR --size large --stateful` | The same, but each dispatch timed from the buffers the previous dispatches left (the functional simulator advances them in between) instead of from the initial buffers. Sequential and slower; for multi-pass algorithms whose later dispatches depend on earlier results. |
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

   - Offline, all ninety-nine buffer workloads pass their numpy references with shim-compiled Mesa 25.2.8 and 26.0.3 code
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
