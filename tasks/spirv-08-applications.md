# Task SPIR-V 8: applications (ray tracer, neural networks, image processing, cellular automaton, graph search,
# encryption, hashing, text, compression)

**Workloads:** `raytrace`, `mlp`, `sobel`, `life`, `bfs`, `conv_layer`, `hist_eq`, `aes`, `hash_table`, `word_count`, `dct8`, `attention`, `resize`, `median3`, `poisson`, `lbm`, `base64`, `md5`, `gemv`, `levenshtein`, `blur`, `yuv420`, `montecarlo`, `xxhash`, `bloom`, `collatz`, `tridiag`, `merge`, `ccl`, `pagerank`, `skinning`, `strsearch`, `quantize`, `rule110`, `voxel`, `nms`, `sat`, `sssp`, `morton`, `utf8`, `fir`, `composite`, `nqueens`, `lzmatch`, `bitpack`, `perlin`, `adler32`, `sieve`, `gcd`, `dither`, `advect`, `knn`, `csvparse`, `pip`, `lift53`, `chacha20`, `modexp`, `floyd`, `f16pack`, `raster`, `canny`, `secded`, `bicubic`, `dfa`, `b64decode`, `cordic`, `rsenc`. These are recorded and verified by `make asahi-verify` with the others;
they are **PENDING** until then. The detailed procedure is in `tasks/spirv-workloads.md`.

## Why these

The other workloads test instructions and algorithms. These three are small but complete applications: the kind of
thing someone would actually run on the GPU. Each produces a result a person can check by eye, as well as by the
reference.

| Workload | What it computes | Output |
|---|---|---|
| `raytrace` | Three spheres on a checkered ground plane, a directional light with hard shadows, Lambert shading, one mirror bounce (the blue sphere and the floor reflect) | RGBA8 image: `image0.ppm` from `run`, `sim_image0.ppm` from `sim` |
| `mlp` | A 128 → 256 (ReLU) → 16 (softmax) perceptron over a batch, one workgroup per input | Class probabilities; the check also requires the same predicted class as float64 |
| `sobel` | Sobel gradient magnitude, min(\|gx\| + \|gy\|, 255), on an 8-bit 4K image | 8-bit image, `image1.pgm`; integer-exact |
| `life` | Game of Life, 64 generations on a 2048² torus (one dispatch each) | Final grid as `image0.pgm`; exact |
| `bfs` | Breadth-first search from vertex 0 over a 1 Mi-vertex random graph, one dispatch per level | Distances and the number of vertices found per level; exact |
| `conv_layer` | A 3×3 convolution layer, 16 → 32 channels, ReLU, on 224×224 | Feature maps; within a summation error bound |
| `hist_eq` | Histogram equalisation of a low-contrast 4K image | Equalised image `image2.pgm` (contrast stretched to 0..255); exact |
| `aes` | AES-128 (ECB) of 1 Mi blocks with a random key; the Python reference must reproduce FIPS-197 appendix C.1 first | Ciphertext; exact |
| `hash_table` | Build a 2 Mi-slot open-addressing table from 1 Mi keys (atomicCompSwap), then 1 Mi lookups (about half hits) | Lookup results, table occupancy; exact |
| `word_count` | Word count, line count and word-length histogram of 16 MiB of generated text | 18 counters; exact |
| `dct8` | JPEG luminance DCT + quantisation of a 4K image (the first stage of a JPEG encoder) | Quantised coefficients; exact except ±1 at .5 ties |
| `attention` | The core of a transformer layer: softmax(Q Kᵀ / 8) V for 64 Ki queries over 256 keys | Attention outputs; relative error < 2e-5 |
| `resize` | Bilinear 3× upscale of a 720p RGBA image to 4K | `image1.ppm`; within 1 level per channel |
| `median3` | Removing salt-and-pepper noise (10% of pixels) from a 4K image with a 3×3 median | `image1.pgm`; exact (830 K noisy pixels in, 643 out on lavapipe) |
| `poisson` | 200 Jacobi sweeps of a 1024² Poisson solve, one dispatch each | Solution field; within 1e-5 of the same iteration in float64 |
| `lbm` | A decaying shear wave in a lattice-Boltzmann fluid (512², 200 steps) | Distributions within 2e-6 of float64; mass drift < 1e-5 |
| `base64` | Base64-encode 12 MiB of bytes | Text identical to Python's `base64` |
| `md5` | MD5 of 256 Ki messages | Digests identical to `hashlib` |
| `gemv` | y = A x for an 8192 × 8192 fp32 matrix (256 MiB read) | Within the summation error bound; GB/s against the memory bandwidth |
| `levenshtein` | Edit distance of 262144 string pairs (up to 32 bytes), one dynamic program per thread | Exact; Mesa keeps the 33-entry row in registers, indexed through `icmpsel` chains (329 of 421 instructions) |
| `blur` | Separable 9-tap Gaussian blur of a 3840 × 2160 fp32 image, row pass then column pass | Within 40 ulp of the image maximum (float64 reference) |
| `yuv420` | RGBA8 → NV12 (BT.601 limited range, integer coefficients) at 3840 × 2160 | Byte-exact luma and chroma planes; `sim_image1.pgm` / `image1.pgm` show the luma |
| `montecarlo` | π from 2^28 PCG-hashed points with 15-bit integer coordinates | Exact per-thread counts and total (subgroup sum, one atomic per subgroup) |
| `xxhash` | xxHash32 of 131072 records of 1036 bytes (deduplication) | Exact; the numpy reference matches the xxHash library |
| `bloom` | Bloom filter of 2^21 keys (10 bits per key, 6 hashes): insert with `atomicOr`, then query | Exact bit array and answers; no false negatives, ~0.8% false positives |
| `collatz` | 3n + 1 trajectories of 2^20 starts below 2^31 in uint64: steps and peak | Exact; per-lane trip counts from 20 to ~800 steps (divergence) |
| `tridiag` | 65536 tridiagonal 64 × 64 systems, Thomas algorithm per thread | Within 64 ulp of float64 `numpy.linalg.solve`; private arrays go to the stack |
| `merge` | Stable merge of two sorted int32 arrays (8.4M + 6.3M) by merge path, one binary search per output | Exact values and source indices (ties take A first) |
| `ccl` | Connected components of a 512 × 512 binary image by label propagation with `atomicMin`, 308 dispatches | Exact labels (smallest pixel index per component); the dispatch count covers the slowest (Jacobi) order |
| `pagerank` | 20 PageRank iterations on a 2^20-page graph with heavy-tailed out-degrees (CSR, pull form) | Within 1e-4 relative of float64; rank sum 1 |
| `skinning` | Linear blend skinning of 2^20 vertices by 4 of 64 bone matrices: positions and normals | Within 2^-18 of float64 |
| `strsearch` | Find an 8-byte pattern in 64 MiB of DNA-like text: per-position flags, `atomicAdd` count, `atomicMin` first | Exact flags, count and first position |
| `quantize` | Per-row int8 quantization of 32768 × 256 fp32 activations (round half to even), with per-row Σ\|q\| in shared memory | Exact; row 0 holds ties (0.5 → 0, 1.5 → 2, 2.5 → 2) |
| `rule110` | Rule 110 automaton on an 8192-cell ring, 1000 dependent dispatches building the space-time history | Exact every row; `sim_image0.pgm` / `image0.pgm` show the history |
| `voxel` | Bin 2^22 clustered points into a 64³ grid: counts and 16.16 fixed-point coordinate sums with `atomicAdd` | Exact under any atomic order |
| `nms` | Fast non-maximum suppression of 8192 score-sorted detection boxes (IoU > 0.5), earlier boxes streamed through shared memory | Exact keep flags and count (integer IoU test: 2 · intersection > union) |
| `sat` | Summed-area table of a 3840 × 2160 8-bit image: row scans, then column scans in place, shared-memory Hillis-Steele with carry | Exact; a box sum from four lookups is checked too |
| `sssp` | Bellman-Ford shortest paths on a 512 × 512 road grid with highways, `atomicMin` relaxation, 78 dispatches | Exact distances (the dispatch count covers synchronous rounds; in-place updates only converge faster) |
| `morton` | 32-bit and 64-bit Morton (Z-order) codes of 2^22 points | Exact (mask-and-shift bit spreading, 64-bit integers) |
| `utf8` | Validate and decode 16 MiB of UTF-8 (1- to 4-byte characters, half the text with planted bad bytes) | Exact code points, U+FFFD for every invalid sequence; the valid half also matches Python's decoder |
| `fir` | 63-tap windowed-sinc low-pass FIR over 8M audio samples through a shared-memory window | Within the dot-product error bound; the stop-band tone drops ~100 dB |
| `composite` | Premultiplied RGBA8 "over" onto a 4K background with exact /255 rounding | Exact; `sim_image2.ppm` / `image2.ppm` show the composite |
| `nqueens` | Count all 13-queens solutions: 2197 three-row prefixes, bitmask DFS with an explicit stack per thread | Exact per prefix and 73712 in total; Mesa keeps the stack in registers (180 `icmpsel`) |
| `lzmatch` | LZ77 longest-match search (64-byte window, up to 32-byte matches) at every position of 1 MiB of repetitive prose | Exact length and nearest distance per position |
| `bitpack` | Pack 2^24 7-bit values into a dense bit stream with `atomicOr` (straddling values split over two words) | Exact words, and the stream unpacks back to the values |
| `perlin` | Four octaves of 2-D improved Perlin noise over 2048 × 2048: fp32 heights and an 8-bit height map | Within 1e-5 of float64, image within one level; `sim_image2.pgm` / `image2.pgm` show the terrain |
| `adler32` | Adler-32 (zlib / PNG) of 16384 blocks of 4 KiB, deferred modulo as in zlib | Exact against `zlib.adler32` |
| `sieve` | Segmented sieve of Eratosthenes up to 2^26: 4096-number segments in shared memory, base primes marked with shared `atomicOr` | Exact composite bits and π(2^26) = 3957809 |
| `gcd` | Binary (Stein) GCD of 2^22 pairs of 32-bit integers sharing random factors, zeros and powers of two included | Exact against `numpy.gcd` |
| `dither` | 8 × 8 Bayer ordered dithering of a 4096 × 4096 grayscale image to 1 bit per pixel | Exact packed bits; `image1.pgm` shows the dithered picture |
| `advect` | 40 steps of semi-Lagrangian dye advection in a 1024² swirl (bilinear back-trace sampling) | Within a per-step bound of fp32 position rounding times the dye slope (≤ 1 per cell) |
| `knn` | 8 nearest of 4096 integer reference points for 65536 queries, shared-memory tiles, insertion-sorted per thread | Exact indices and squared distances (ties to the lower index; a duplicated point forces one) |
| `csvparse` | Parse 4M fixed-width CSV integer fields, flagging malformed ones (letters, stray signs, decimals, blanks) | Exact values and validity (reference: the regular expression ` *-?[0-9]+`) |
| `pip` | Winding number of 2^20 points against a 500-vertex concave polygon with 64-bit cross products | Exact, vertices themselves included |
| `lift53` | JPEG 2000 lossless LeGall 5/3 forward wavelet of 4096 rows of 2048 samples by integer lifting | Exact, and the inverse transform reconstructs the input |
| `chacha20` | ChaCha20 (RFC 8439) encryption of 16 MiB, one 64-byte keystream block per thread | Exact; the numpy reference is cross-checked with the `cryptography` package where installed |
| `modexp` | 2^20 modular exponentiations base^exp mod m with 32-bit operands and 64-bit products | Exact against Python `pow` (mod 1, x^0 and a Fermat identity included); Mesa lowers the 64-bit `%` to ~1800 instructions |
| `floyd` | Floyd-Warshall all-pairs shortest paths on a 512-vertex sparse graph, 512 dependent dispatches | Exact distances, unreachable pairs stay at the sentinel |
| `f16pack` | `packHalf2x16` of 2^23 floats from fp16-subnormal to overflow magnitudes, then `unpackHalf2x16` in a second dispatch | Bit-exact against IEEE round-to-nearest-even, fp16 subnormals and infinities included |
| `raster` | Software rasterization of 131072 small triangles at 1920 × 1080: integer edge functions over each bounding box, `atomicMin` depth / id resolve | Exact frame buffer (degenerate triangles cover only their box, as scanned) |
| `canny` | Canny edge thinning of a 4K image: integer Sobel, direction sector from exact tan 22.5° comparisons, non-maximum suppression, threshold (2 dispatches) | Exact edge map; `image2.pgm` shows the edges |
| `secded` | Extended Hamming(8,4) SECDED encode of 2M words, then decode after injected 0 / 1 / 2-bit errors per byte (2 dispatches) | Exact; single errors corrected, every double error flagged |
| `bicubic` | 2× Catmull-Rom bicubic upscaling of a 1920 × 1080 fp32 image to 4K | Within 64 ulp of the image maximum (float64 reference) |
| `dfa` | Aho-Corasick multi-keyword search (6 keywords, 24-state DFA in shared memory) over 2^18 log lines of 128 bytes | Exact per-line counts, overlapping matches included |
| `b64decode` | Base64 decoding of 8 MiB of text with corrupted characters, padding at the end | Exact bytes and per-quantum validity flags (against Python's `base64`) |
| `cordic` | Q2.30 fixed-point CORDIC sine / cosine of 2^22 angles over [-π, π], 30 shift-and-add steps | Exact against the same integer steps in numpy; within 2e-8 of `sin` / `cos` |
| `rsenc` | RS(255, 223) Reed-Solomon parity of 32768 codewords over GF(2^8), log / antilog tables in shared memory | Exact parity; sampled codewords have zero syndromes |

## Offline sanity checks (not hardware verification)

- **lavapipe:** all pass at both sizes.
- **Shim-compiled Mesa 25.2.8 and 26.0.3 code on the functional simulator:** 134 of 134.
  - `bfs` found a simulator bug. Mesa aggregates `atomicAdd(found[l], 1)` under divergence: one lane, chosen with SR58
    (`active_thread_index_in_simdgroup`), adds SR59, the number of active lanes. The simulator modelled both from the
    launch shape (lane index, 32), which is only right when every lane is active, so the counts came out as multiples
    of 32. They are now the rank among, and the count of, the active lanes. The recorded suite only has full-exec SR
    reads, so it could not catch this.
  - **On the M2, confirm the active-lane meaning of SR58 / SR59.** `bfs` PASSing word for word confirms it.
- **Cost model:** predictions for the large sizes.
- **`raytrace`:** checked against an independent float64 numpy version of the same scene. Pixels on silhouettes and
  shadow edges may flip between float implementations. The check allows 1% of pixels to differ by more than 2 levels:
  lavapipe is at 0.09% at 1920×1080, and the simulator at 0.4% at 96×64. The far ground is a uniform grey on purpose:
  a distant checkerboard aliases, and any rounding difference flips its squares.

## On the M2

1. After `make asahi-verify`, look at the images: the GPU's render and the simulator's render of the recorded code
   should be indistinguishable. Convert them to PNG with any image tool.
2. `sobel` must be word-for-word identical (PASS).
3. `raytrace` and `mlp` use sqrt / exp / rcp, so expect CLOSE. Record the number of differing pixels and their
   largest difference.
4. Compare the large-size times with `predict`, and report frames per second for the 1080p ray tracer, and inferences
   per second for `mlp`.
5. `life`, `bfs` and `hist_eq` are exact and multi-dispatch. They must PASS, and they show whether results carry
   correctly across dispatches. For `life` and `bfs`, compare the time per dispatch (64 and 32 small dispatches) with
   the launch overhead.
6. `levenshtein`, `yuv420` and `montecarlo` are exact and must PASS. `levenshtein` is the first workload where the
   compiler turns a dynamically indexed private array into register selects; compare its time with `predict`, since
   the select chains make it ALU-bound. `blur` is CLOSE or PASS; report megapixels per second for both passes.
7. `xxhash`, `bloom` and `collatz` are exact and must PASS. `tridiag` divides, so expect CLOSE; report its largest
   difference. Mesa 25.2.8 compiles it to 1304 instructions and 26.0.3 to 2266 (about 900 extra `xor`), and the
   simulator executes twice as many with 26.0.3: note which Mesa the M2 runs and compare the time with `predict`.
8. `merge` and `ccl` are exact and must PASS. `ccl` converges whatever order the GPU's in-place updates land in,
   because each update can only lower a label faster than the Jacobi order that sets the dispatch count. Its
   prediction runs every dispatch from the initial labels. `pagerank` and `skinning` are fp32 sums, so expect CLOSE
   or PASS. `pagerank` is irregular gathers; compare its time with `predict` as a memory-latency case.
9. `strsearch`, `quantize`, `rule110` and `voxel` are exact and must PASS. `quantize` confirms `roundEven` and the
   fp32 → int conversion. `rule110` is 1000 dispatches of 128 threads each: its time per dispatch is a direct
   measure of launch overhead (the model predicts 6.4 µs, of which 5.6 µs is launch).
10. `nms`, `sat`, `sssp` and `morton` are exact and must PASS.
    - **`nms` is a known under-prediction.** Workgroup k compares its boxes with all earlier ones, so its work grows
      with k. Its 32 workgroups fit one per core, and the dispatch takes as long as the last, heaviest workgroup.
      The model simulates workgroup 0 as the representative, the lightest, so it predicts 78 µs.
    - Expect the M2 to take far longer, roughly 32 × the work of workgroup 0 on its busiest core.
    - The measured time decides how the model should pick representative workgroups when work is uneven.
11. `utf8`, `composite` and `nqueens` are exact and must PASS. `fir` is fp32 dot products, so expect CLOSE or PASS.
    `nqueens` is all divergent integer control flow: compare its time with `predict`.
12. `lzmatch`, `bitpack` and `adler32` are exact and must PASS; `perlin` is fp32, so expect CLOSE or PASS. Look at
    the `perlin` height map next to the simulator's.
13. `sieve`, `gcd` and `dither` are exact and must PASS; `advect` is fp32 interpolation over 40 dispatches, so expect
    CLOSE. Its bound is derived (positions rounded to fp32, dye slope at most 1 per cell, non-expansive sampling).
14. `knn`, `csvparse`, `pip` and `lift53` are exact and must PASS.
15. `chacha20`, `modexp`, `floyd` and `f16pack` are exact and must PASS.
    - `f16pack` is the first workload that checks fp16 subnormal *results* of a conversion on the GPU. A FAIL there
      means the hardware flushes them; record which values differ.
    - Its round trip is split into two dispatches on purpose. In one shader, lavapipe's compiler folded
      `unpackHalf2x16(packHalf2x16(v))` back to `v` for one component, which is not a correct round trip.
16. `raster`, `canny` and `secded` are exact and must PASS; `bicubic` is fp32 (CLOSE or PASS).
17. `dfa`, `b64decode`, `cordic` and `rsenc` are exact and must PASS.
