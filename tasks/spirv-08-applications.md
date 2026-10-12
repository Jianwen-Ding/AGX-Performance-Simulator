# Task SPIR-V 8: applications (ray tracer, neural networks, image processing, cellular automaton, graph search,
# encryption, hashing, text, compression)

**Workloads:** `raytrace`, `mlp`, `sobel`, `life`, `bfs`, `conv_layer`, `hist_eq`, `aes`, `hash_table`, `word_count`, `dct8`, `attention`, `resize`, `median3`, `poisson`, `lbm`, `base64`, `md5`, `gemv`, `levenshtein`, `blur`, `yuv420`, `montecarlo`, `xxhash`, `bloom`, `collatz`, `tridiag`, `merge`, `ccl`, `pagerank`, `skinning`, `strsearch`, `quantize`, `rule110`, `voxel`, `nms`, `sat`, `sssp`, `morton`, `utf8`, `fir`, `composite`, `nqueens`, `lzmatch`, `bitpack`, `perlin`, `adler32`, `sieve`, `gcd`, `dither`, `advect`, `knn`, `csvparse`, `pip`, `lift53`, `chacha20`, `modexp`, `floyd`, `f16pack`, `raster`, `canny`, `secded`, `bicubic`, `dfa`, `b64decode`, `cordic`, `rsenc`, `texfetch`, `texfilter`, `imgerode`, `imgtonemap`, `argmax64`, `minmax64`, `dsum`, `quadmip`, `clustered`, `specblur`, `cullindirect`, `bigsort`, `pushmat`, `mplitmus`, `splat`, `splatfix`, `smembank`, `coalesce`, `atomcontend`, `barriercost`, `aluprobe`, `ubopalette`, `uboparams`, `bindless`, `bindlessu`, `rle`, `huffdec`, `lookback`, `hll`, `tricount`, `sha3`, `stereo`, `viterbi`, `viterbifix`, `shiftfuse`. These are recorded and verified by `make asahi-verify` with the others;
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
| `texfetch` | 4M `texelFetch` reads of a 2048² r32f texture through the texture unit (`texture_sample` / `texture_load` in the AGX code) | Bit-exact, signed zero and extreme values included |
| `texfilter` | 4M bilinear samples of a 1024² RGBA8 texture (ramps, checker, noise) with clamp-to-edge borders | Within 2/255 of the Vulkan definition at the sub-texel precision that fits best; that precision is reported |
| `imgerode` | 3 × 3 grey-level erosion of a 2048² r32ui storage image, ping-pong between two images over 4 dispatches | Exact |
| `imgtonemap` | Exposure + Reinhard tone mapping in place on a 4K RGBA8 storage image | Within one level of round-to-nearest (Vulkan's allowance); alpha untouched; the fraction rounding to nearest is reported |
| `argmax64` | Argmax of 4M float scores into 4096 buckets via 64-bit `atomicMax` on (order-preserving float bits, index) keys | Exact, ties to the largest index |
| `minmax64` | Min, max and xor of 4M uint64 values, shared-memory 64-bit atomics per workgroup, then global | Exact |
| `dsum` | fp64 dot products of 4M doubles with a wide dynamic range, a fixed 256-wide tree per workgroup | Bit-exact against the same tree in float64 |
| `quadmip` | 2 × 2 mipmap of a 4096² image, one quad per block via quad swaps; diagonal swap and quad broadcast written per lane | Exact; pixels follow the subgroup ids, so any subgroup layout is valid |
| `clustered` | Clustered add (2, 4, 8 lanes) and max (8 lanes) over 4M values | Exact, grouped by the ids the device reports |
| `specblur` | 15-tap horizontal box filter of a 4K 8-bit image; radius 7 and workgroup 128 from specialization constants | Exact |
| `cullindirect` | Cull 2^20 spheres against a 6-plane frustum, write dispatch arguments from the visible count, then an indirect dispatch assigns levels of detail | Exact visible set, count, arguments and per-object LOD |
| `bigsort` | 512 workgroups each sort 8192 keys in 32 KiB of shared memory with 1024 invocations (bitonic, barrier per stage) | Exact |
| `pushmat` | 4M vertices through four mat4 (256 bytes of push constants), integer-valued so fp32 is exact | Bit-exact |
| `mplitmus` | 4096 writer / reader workgroup pairs race on data + flag: release / acquire atomics, and relaxed for comparison, with no waiting | Zero stale reads after an acquire; relaxed stale reads are reported, not judged |
| `splat` | Splat 4M weighted points bilinearly into a 512² grid with fp32 `atomicAdd` (half the points in a dense blob) | Within a per-cell summation bound (any addition order) |
| `splatfix` | The same points with weights in 16.16 fixed point and `atomicAdd` on uint | Exact; the fixed-point grid is within half a unit per term of the exact sums |
| `smembank` | 64 shared-memory reads per lane at stride 1, 2, 4, 8, 16, 32, 33, one dispatch each | Exact sums; per-stride times |
| `coalesce` | 16 device reads per lane at stride 1 .. 64 words over 64 MiB, one dispatch each | Exact sums; per-stride times |
| `atomcontend` | 4M atomicAdds per dispatch into 1, 32, 1024, 32768 and 2^20 counters | Exact counts; per-dispatch times |
| `barriercost` | 0, 16, 64, 256 rounds of neighbour exchange with two barriers each | Exact; per-dispatch times |
| `aluprobe` | 6 dispatches × 524288 invocations × 256 steps: fp32 fma, fp16 fma, imad, each as one dependent chain and as 8 independent chains | Bit-exact (integer-valued floats, wrapping integers); per-kind times |
| `ubopalette` | 4M vertices × one of 64 mat4 from a std140 UBO (integer-valued, exact in fp32) | Bit-exact |
| `uboparams` | 4M RGBA pixels through gains, offsets, a 3 × 3 mix and clamps read from a UBO | Exact |
| `bindless` | 4M lookups, each lane picking one of 8 storage-buffer tables (1-8 K entries) by `nonuniformEXT` index | Exact |
| `bindlessu` | 4M lookups, workgroup g reading table g mod 8 | Exact |
| `rle` | 4M symbols (16-symbol alphabet, mean run 12) encoded per 256-element tile, then decoded | Exact runs and round trip; per-pass times |
| `huffdec` | 16384 blocks × 256 symbols (64-symbol Zipf alphabet, about 4.6 bits per symbol) | Exact |
| `lookback` | 4M uint32 (4096 tiles of 1024) scanned in one dispatch | Exact (mod 2^32), no spin timeouts, every tile claimed; mean look-back steps reported |
| `hll` | 16M items (about 5.3M distinct) into 4096 registers | Exact registers; the cardinality estimate (within 0.2% offline) is reported |
| `tricount` | 65536 vertices, about 1.05M forward edges with skewed degrees (up to 1359), 101544 triangles | Exact per-edge counts and total |
| `sha3` | 262144 messages of 64 bytes | Digests equal Python's `hashlib.sha3_256` (all at small size; 12021 sampled at large) |
| `stereo` | 1024 × 512 synthetic layered scene (disparities 4-59), 64 disparities | Exact disparity map; fraction within 1 px of the true disparity reported (0.90 offline) |
| `viterbi` | 8192 frames of 256 steps through a 3% bit-flip channel | Exact decoded bits (residual bit error rate about 1e-4 offline). Expected to FAIL on the M2: see note 37 |
| `viterbifix` | The same frames | Exact decoded bits |
| `shiftfuse` | 1M random (x, y, s) triples, 10 idioms | Exact per column; expected on the M2: 3 columns fail (note 38) |

## Offline sanity checks (not hardware verification)

- **lavapipe:** all pass at both sizes.
- **Shim-compiled Mesa 25.2.8 and 26.0.3 code on the functional simulator:** 138 of 138 buffer workloads (plus quadmip and clustered).
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
18. **`texfetch` and `texfilter` are the first workloads that use the texture unit.**
    - `vk_run` binds them as combined image samplers, uploaded through a staging buffer.
    - The simulator does not execute `texture_sample`, so verification can only compare the GPU with the reference.
      Their status is UNSIM when that matches, and FAIL when it does not. `sim` and `predict` report them as
      unsupported.
    - Record `texfilter`'s `best_subtexel_bits` (lavapipe: 8). It is the M2 texture unit's interpolation precision,
      and the first measured fact for a future texture model.
19. `imgerode` and `imgtonemap` use storage images (`vk_run`'s `image` directive). The image is reloaded from a
    staging buffer at the start of every run, and copied back after the dispatches. The AGX code uses `texture_load`
    and `image_write`; like the textures, these are UNSIM in verification. Record `imgtonemap`'s
    `round_to_nearest_fraction` (lavapipe: 0.992); it shows how the M2 rounds unorm stores.
20. **64-bit atomics: `argmax64` and `minmax64`.**
    - Offline, Mesa 25.2.8 and 26.0.3's Asahi driver do not advertise `shaderBufferInt64Atomics` or
      `shaderSharedInt64Atomics` (lavapipe does). Forcing the shaders through the compiler hits
      `no 64-bit atomics yet` in instruction packing.
    - The workloads therefore `require` those features, and expect NODEV on the M2: recorded as
      `unsupported_on_device` and not run.
    - If a newer Mesa on the M2 advertises them, they run and must PASS. That is the first check of 64-bit atomics on
      AGX.
21. More device features:
    - **`dsum`** requires fp64. The offline Asahi driver reports `float64: 0`, so expect NODEV unless the M2's Mesa
      adds it.
    - **`quadmip` and `clustered`** need quad and clustered subgroup operations, which the driver offers (subgroup op
      mask 1791). They must PASS on the M2.
    - Their references group lanes by the subgroup ids the device writes out. The 8-lane subgroups of lavapipe and
      the 32-lane subgroups of AGX are both checked correctly.
    - `vk_run`'s device line now lists these features and the subgroup op mask for every recording.
22. **`specblur`** checks that specialization constants reach the driver: the recorded AGX code has them folded
    in, and the simulator runs that code.
    **`cullindirect`** is the first launch whose size the GPU itself computes (`vkCmdDispatchIndirect`). Expect PASS;
    it is UNSIM in verification.
23. **Device limits and the memory model.**
    - `bigsort` and `pushmat` `require` the limits they use, and the offline Asahi driver offers them: 32 KiB shared
      memory, 1024 invocations, 256-byte push constants.
    - `mplitmus` is a race by design: on lavapipe every reader saw its flag, and on the M2 workgroups run truly in
      parallel.
    - It must show `acquire_stale = 0`. Record `relaxed_stale_reads`: a non-zero count would be the M2 visibly
      reordering relaxed accesses.
    - Mesa compiles its atomic loads / stores to the `.TODO` encodings of `device_load` / `device_store`, between
      `memory_barrier` sequences. The simulator now executes those as the plain accesses (their extra bits are
      coherence controls), which is exact for its single sequential memory.
24. **Float atomics.**
    - The offline Asahi driver does not offer `VK_EXT_shader_atomic_float` adds; lavapipe does. So `splat` is
      expected NODEV on the M2, unless its Mesa adds them.
    - `splatfix` is the workaround real code uses on such GPUs. It must PASS, and runs on the simulator.
    - Both read the same points (a shared seed), so their grids can be compared directly.
25. **Timing probes `smembank` and `coalesce`.** Their value is the per-dispatch time. `verify --timing` lists
    measured and predicted per stride as `per_dispatch_us`, set against the model's predictions:
    - shared memory, µs per stride 1, 2, 4, 8, 16, 32, 33: 76, 77, 77, 79, 106, 192, 76 (bank conflicts from
      stride 16; stride 33 is conflict-free);
    - device memory, µs per stride 1 .. 64: 24, 104, 197, 383, 756, 406, 230. Strides 32 and 64 wrap the 64 MiB
      buffer and reuse lines; whether the M2 shows that drop too is part of the check.
26. **Probes `atomcontend` and `barriercost`.**
    - The model predicts `barriercost` at 8.5, 16.6, 45.6 and 162 µs, about 0.27 µs per barrier pair per wave.
    - It predicts `atomcontend` flat at 11.5 µs for every table size. It has no cross-group same-address
      serialisation for return-less atomics, so the single-counter dispatch is the likely large under-prediction.
    - The M2's per-dispatch times give the slope to add.
27. **`aluprobe` checks the ALU rules on Linux.** Those rules were fitted through Metal, and Mesa unrolls each kind
    to 257 `fmadd32` / `fmadd16` / `imadd`. Predicted µs per kind:

    | fma32 dependent | fma32 ×8 | fma16 dependent | fma16 ×8 | imad dependent | imad ×8 |
    |---|---|---|---|---|---|
    | 112 | 52 | 95 | 57 | 117 | 97 |

    The ratios of dependent to independent are the latency / throughput balance to confirm.
28. **Uniform buffers.**
    - The preamble loads the descriptor addresses (`uniform_store`). The UBO *contents*, however, are loaded per
      invocation in the main code, even `uboparams`' six parameter vectors, which are the same for every invocation.
    - Compare their M2 times with `predict`. A large gap would point at that per-invocation load.
29. **Descriptor arrays.**
    - `vk_run` builds them from `buffer` lines that share a binding (element index as the fifth field). The
      offline driver offers both dynamic and non-uniform storage-buffer indexing.
    - On AGX a descriptor array is plain memory: each lookup is a `device_load` of the descriptor, then of the
      data, with no waterfall loop even for the non-uniform index.
    - The simulator's descriptor set now follows honeykrisp's layout rule: bindings in order, each `count × stride`
      after the previous ones. Both workloads simulate, and every earlier prediction is unchanged.
30. **Run-length encoding.**
    - `rle` is a real columnar/texture-compressor pattern: runs never cross a 256-element tile, so the encode
      parallelises per tile. Pass 1 (offsets over 16384 tile counts) is one thread doing a dependent
      load-add-store loop.
    - The simulator predicts about 2.09 ms, of which 1.77 ms is that serial pass, so the M2 time mostly measures
      the latency of a dependent device load and store per iteration. Pass 3's binary search (about 19 dependent
      loads per element) is predicted at only 67 µs, so compare the per-pass times rather than the total.
31. **Huffman decoding.**
    - Each symbol depends on the previous one's code length (the bit position), so a thread's 256 symbols are one
      dependent chain of a shared-memory lookup, two device loads and an add. Throughput comes only from the
      number of resident threads.
    - Predicted 257 µs for 4M symbols. A much larger M2 time would mean the shared-memory lookup latency, or the
      two unaligned word loads per peek, cost more in a dependent chain than the model assumes.
32. **Decoupled look-back scan.**
    - The only workload whose workgroups wait on each other inside one dispatch. It relies on workgroups that took
      an earlier tile id making progress while later ones spin; a bounded spin turns a missing guarantee into a
      `timeouts` count instead of a GPU hang.
    - `probes` reports `mean_lookback_steps`. The simulator runs workgroups one after another, so every look-back
      ends at the previous tile (1.0); lavapipe gave about 1.3. The M2 number shows how far ahead tiles get of the
      inclusive prefix.
    - Predicted 55 µs, which is faster than the M2's DRAM bandwidth allows for 32 MB of traffic, and does not include
      the serial chain of flag publications. A much larger M2 time is expected. Its size shows how much each
      release / acquire round trip between cores costs.
33. **HyperLogLog.**
    - The registers are order-independent maxima, so the result is exact whatever the order of the atomics.
      `estimate` / `rel_error` show that the sketch does its job end to end.
    - Predicted 345 µs. Each workgroup issues 4096 shared-memory `atomicMax` operations onto random registers
      (bank conflicts are rare), then up to 4096 device `atomicMax` operations spread over 16 KiB. A large gap on the
      M2 would most likely come from the device-atomic merge, which every workgroup does onto the same 4096 words.
34. **Triangle counting.**
    - A SIMD group runs until its longest merge finishes, so the time is set by the degree skew within groups of 32
      consecutive edges (sorted by source, so a hub's edges share groups), not by the average work.
    - Predicted 3.7 ms. The simulator follows every lane's actual path, so this tests how well the model handles
      divergent loops whose trip counts differ by orders of magnitude between groups.
35. **SHA3-256.**
    - AGX has 32-bit integer ALUs, so every 64-bit lane operation is split by the compiler: a rotation becomes
      shifts and ors on both halves. The driver unrolls all 24 rounds into about 6000 instructions with no scratch.
    - Predicted 455 µs for 262144 hashes. The M2 time checks the cost of a long, branch-free, register-heavy 32-bit
      integer stream, and whether instruction fetch keeps up with a 39 KB body.
36. **Stereo matching.**
    - Pass 1 reads 25 census pairs for each of 64 disparities per pixel (3200 loads), all from a few rows of
      the two census images, so it measures L1 hit throughput and `bitCount` rather than DRAM.
    - Predicted 3.6 ms (0.19 ms census, 3.4 ms matching). `within_1px_of_truth` shows that the output is a usable
      depth map, not just equal to the reference.
37. **Viterbi decoding and a shift-mask compiler fusion.**
    - The traceback tests bit `st & 31` of a survivor word as `(x >> (st & 31u)) & 1u`, with `st` up to 63. In
      Mesa 25.2.8 and 26.0.3 the AGX backend compiles this to `bfeil r, 0, x, st, mask 0x1`. NIR drops the `& 31`
      because `ushr` already masks its shift, and `agx_nir_algebraic.py`'s `fuse_ubfe` then turns shift-and-mask
      into `ubitfield_extract` and a `bfeil` with the unmasked shift.
    - The recorded M2 case `alu/bfeil_r14_r6_r7_r8` shows that `bfeil` takes 7 bits of its shift (shifts of 32 or
      more give 0). The functional simulator, which matches that recording, therefore fails `viterbi`: states 32-63
      read their decisions as 0, and about 30% of decoded bits are wrong. lavapipe passes.
    - So the M2 is expected to fail `viterbi` the same way, and to pass `viterbifix`, which uses `bitfieldExtract`
      with an explicitly masked offset (the compiler keeps that mask). `probes` reports both `mismatched_bits`.
    - If the M2 passes `viterbi`, the hardware's `bfeil` with an immediate mask uses fewer shift bits than the
      recorded case, and the simulator needs that distinction.
    - Predicted time (viterbifix): about 3.6 ms for 8192 frames, dominated by 256 barrier-separated steps per
      workgroup.
38. **Shift-mask fusion probe.**
    - `shiftfuse` isolates the `viterbi` miscompilation. On code from Mesa 25.2.8 and 26.0.3 the functional simulator
      fails exactly the three columns where a shift right by `s & 31` is followed by a constant low mask:
      `bit_test` (`& 1`), `byte_extract` (`& 0xFF`) and `double_and_mask` (`s & 63 & 31`, then `& 0x7FFF`). Each
      becomes `bfeil ..., s, mask K` with the unmasked shift.
    - The other seven columns pass. In the funnel shift, rotate, byte insert, shift-left-then-mask,
      arithmetic-shift-then-mask and signed `bitfieldExtract` columns the compiler keeps the `& 31`. A
      variable-width mask is not fused at all.
    - lavapipe passes all ten. On the M2, `probes` lists the failing columns. The expectation is the same three; any
      difference refines the simulator's `bfeil` shift semantics, which are fitted to the one recorded case.
    - **Root cause (Mesa 25.2.8 and 26.0.3 source).**
      - AGX shifts use more than 5 bits of the shift amount. `agx_nir_lower_algebraic_late` therefore re-adds `& 31`
        to every shift (`lower_sm5_shift`, whose comment says so), and runs late on purpose: after
        `agx_nir_fuse_algebraic_late`, "since this inserts &s for shifts so we want to do it after fusing constant
        shifts" (`agx_compile.c`).
      - In that earlier fusion pass, `fuse_ubfe` (`agx_nir_algebraic.py`) rewrites
        `iand(ushr(a, b), (1 << bits) - 1)` into `ubitfield_extract(a, b, bits)` for *any* `b`, not only constants.
        A variable shift is therefore absorbed into a `bfeil` before its `& 31` is restored.
      - The other fusions in the pass only match constant shifts: `fuse_extr` 0-31, and the shift fusions in
        `fuse_imad` 1-4.
      - A minimal fix would match only constant shifts in `fuse_ubfe` (`'#b'`), or emit
        `ubitfield_extract(a, iand(b, 31), bits)`.
      - Confirm on the M2 before reporting it: `viterbi` and the three `shiftfuse` columns failing, with `viterbifix`
        and the other columns passing.
