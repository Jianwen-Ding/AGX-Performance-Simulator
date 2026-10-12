# Task SPIR-V 8: applications (ray tracer, neural networks, image processing, cellular automaton, graph search,
# encryption, hashing, text, compression)

**Workloads:** `raytrace`, `mlp`, `sobel`, `life`, `bfs`, `conv_layer`, `hist_eq`, `aes`, `hash_table`, `word_count`, `dct8`, `attention`, `resize`, `median3`, `poisson`, `lbm`, `base64`, `md5`, `gemv`, `levenshtein`, `blur`, `yuv420`, `montecarlo`, `xxhash`, `bloom`, `collatz`, `tridiag`, `merge`, `ccl`, `pagerank`, `skinning`, `strsearch`, `quantize`, `rule110`, `voxel`, `nms`, `sat`, `sssp`, `morton`, `utf8`, `fir`, `composite`, `nqueens`, `lzmatch`, `bitpack`, `perlin`, `adler32`, `sieve`, `gcd`, `dither`, `advect`, `knn`, `csvparse`, `pip`, `lift53`, `chacha20`, `modexp`, `floyd`, `f16pack`, `raster`, `canny`, `secded`, `bicubic`, `dfa`, `b64decode`, `cordic`, `rsenc`, `texfetch`, `texfilter`, `imgerode`, `imgtonemap`, `argmax64`, `minmax64`, `dsum`, `quadmip`, `clustered`, `specblur`, `cullindirect`, `bigsort`, `pushmat`, `mplitmus`, `splat`, `splatfix`, `smembank`, `coalesce`, `atomcontend`, `barriercost`, `aluprobe`, `ubopalette`, `uboparams`, `bindless`, `bindlessu`, `rle`, `huffdec`, `lookback`, `hll`, `tricount`, `sha3`, `stereo`, `viterbi`, `viterbifix`, `shiftfuse`, `bc1enc`, `lz4dec`, `nttmul`, `topk`, `cellsort`, `boruvka`, `biquad`, `edt`, `mt19937`, `bitap`, `cholesky`, `rsshash`, `poly1305`, `crc32c`, `siphash`, `blake2s`, `blake2sfast`, `q4gemv`, `attndec`, `attndecsplit`, `rmsrope`, `q8dot`, `isp`, `contour`, `hough`, `spmvsg`, `varint`, `bvhrays`, `adamw`, `stream`, `motionest`, `jsonidx`, `cdc`, `swalign`, `swalignfit`, `swalignpad`, `jpcolor`, `postings`. These are recorded and verified by `make asahi-verify` with the others;
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
| `viterbi` | 8192 frames of 256 steps through a 3% bit-flip channel | Exact decoded bits (residual bit error rate about 1e-4 offline). Passes or fails depending on an open `bfeil` question: note 37 |
| `viterbifix` | The same frames | Exact decoded bits |
| `shiftfuse` | 1M random (x, y, s) triples, 10 idioms | Exact per column; 3 columns depend on the open `bfeil` question (notes 37-38) |
| `bc1enc` | 2048 × 2048 synthetic RGB image (gradients, flat rectangles, noise): 262144 blocks | Bit-exact blocks; decoded PSNR reported (32.2 dB offline) |
| `lz4dec` | 512 blocks of 4 KiB of request-log-like text (ratio 1.79) | Every block decodes to the original bytes; the streams also decode with the reference `lz4` library |
| `nttmul` | 8192 pairs of random polynomials mod 7681 | Every product equals the schoolbook negacyclic convolution |
| `topk` | 16384 arrays of 1024 keys (half with only 4 distinct values in one byte), random k | Exact against `np.partition` |
| `cellsort` | 4M clustered particles into 65536 cells (up to 1310 per cell) | Exact counts and starts; each cell's ids equal as a set (order inside a cell is up to atomic arbitration) |
| `boruvka` | 262144 vertices, about 1.8M edges with distinct weights, 24 rounds × 5 dispatches | The MST's edge set equals Kruskal's exactly; one component left |
| `biquad` | 4096 channels × 4096 samples of tone plus noise | Bit-exact against the same fixed-point arithmetic in numpy; the fixed-point filter is 44.8 dB SNR against scipy's floating-point `sosfilt` |
| `edt` | 768 × 512 binary image of discs and lines | Exact against `scipy.ndimage.distance_transform_edt` (squared, an independent algorithm) |
| `mt19937` | 1024 streams × 10608 outputs, stream 0 seeded 5489 | Exact against a numpy MT19937; the published known answers for seed 5489 (first output 3499211612, 10000th 4123659995) |
| `bitap` | 4M random bases with planted pattern copies (0-3 edits), pattern of 24, k = 2 | Exact match-end bitmap against Sellers' edit-distance DP (5539 ends); decodes under hypothesis B of note 37 only |
| `cholesky` | 16384 SPD matrices (B Bᵀ / n + I) | Backward error ‖A − L Lᵀ‖ / ‖A‖ ≤ 4 n 2⁻²⁴ (1.4e-7 offline), lower-triangular with a positive diagonal; difference from numpy's float64 factor reported |
| `rsshash` | 1M IPv4/TCP headers, 5% with corrupted checksums, the first 5 being Microsoft's RSS verification packets | Exact hashes and checksum verdicts; the 10 published RSS verification hashes |
| `poly1305` | 65536 messages of 0-1023 bytes with their own keys; message 0 is RFC 8439's 2.5.2 vector | Every tag equals a Python big-integer Poly1305; the RFC's published tag |
| `crc32c` | 16 MiB of random bytes (4096 chunks of 4 KiB) | Equals a plain byte-at-a-time CRC-32C of the whole buffer (no chunking, no combining); the check value crc32c("123456789") = 0xE3069283 |
| `siphash` | 262144 messages; the first 64 are the paper's vector set (key 00..0f, message 00..i-1) | Every hash equals a Python SipHash-2-4; the paper's published outputs for the empty and the 15-byte message |
| `blake2s` | 131072 messages of 0-255 bytes (message 0 is "abc") | Every digest equals Python's `hashlib.blake2s`; RFC 7693 Appendix B's BLAKE2s-256("abc") |
| `blake2sfast` | The same messages | The same checks |
| `q4gemv` | 4096 × 4096 weights quantised to Q4_0 (9.4 MB) times an fp32 vector | Within the fp32 summation bound of the exact product with the dequantised weights; the quantisation error against the fp32 weights is reported (about 8.5%) |
| `attndec` | 32 heads × 4096 cached positions × 128 dims, fp16 K / V (some keys aligned with the query) | Error ≤ 1e-4 of Σ softmax·\|v\| per output against float64 numpy on the same fp16 K / V (2.4e-6 offline) |
| `attndecsplit` | Same sizes and distribution as `attndec` | The same check (1.4e-6 offline) |
| `rmsrope` | 2048 tokens at positions 0-4095 (including 0, 1, 1000, 4095), hidden size 4096, head size 128 | Error per rotated pair ≤ 2⁻⁹ of its magnitude against float64 (3.7e-4 on lavapipe, 3.2e-4 simulated) |
| `q8dot` | 4096 × 4096 Q8_0 weights times a Q8 vector, int8 extremes (127, -128) included | Within the fp32 bound of the exact integer block dots combined with the scales; quantisation error against fp32 reported (0.8%) |
| `isp` | 4032 × 3024 synthetic sensor frame (scene, white-balance cast, noise) | Bit-exact against the same integer pipeline in numpy; in flat regions within about 0.5 display levels of the ideal floating-point development |
| `contour` | 2048 × 2048 synthetic terrain (mm), iso level 250: 113335 segments, 6506 saddle cells | The sorted segment set and the 16 case counts equal numpy's exactly (append order is up to atomic arbitration) |
| `hough` | 2048 × 2048 edge image: 12 planted lines through the interior plus noise (37622 points) | Exact accumulator against numpy; the 12 strongest peaks recover all 12 planted lines |
| `spmvsg` | 131072 × 131072, 1.75M nonzeros, Pareto row lengths (23% empty rows) | Within the fp32 summation bound of a float64 product (any summation order); empty rows exactly 0 |
| `varint` | 4M int32: 80% small magnitudes, the rest up to 2^31, both signs; 0, -1, INT_MAX, INT_MIN first | Output bytes identical to a Python protobuf-style encoder (whose output matches protobuf's documented examples) |
| `bvhrays` | 8264 triangles (terrain + boxes), 4239 BVH nodes, 65536 camera rays (70% hit) | Every ray's hit triangle and distance equal a float64 brute-force test against all triangles (no BVH), allowing fp32 ties on shared edges (1 offline) |
| `adamw` | 4M parameters, log-normal gradient scales, step 100 (zero, tiny and large gradients included) | p, m, v within 16 ulps of a float64 step on the same fp32 inputs (≤ 4.3 offline) |
| `stream` | 3 arrays of 2^24 fp32 + a fill target (64 MiB each) | Each STREAM kernel within 2 ulps of float64 applied in sequence; per-workgroup read sums within the fp32 bound; the fill exact |
| `motionest` | 1280×720 frames (3600 blocks), the second panned by (5, -3) with noise | Every block's (offset, SAD) equal to a numpy full search with the same tie order; `true_motion_found` (the fraction of blocks whose answer is the pan) is 1.0 at both sizes on lavapipe and the simulator |
| `jsonidx` | 4096 documents of 8 KiB (32 MiB) of JSON lines, strings full of escaped quotes, backslash runs and structural characters | Every chunk's mask equal to a vectorised numpy reference, itself checked against a byte-at-a-time state machine at the small size |
| `cdc` | 64 regions of 1 MiB (64 MiB): versions of six files, each with a few insertions, deletions and overwrites | Candidate masks and every region's chunk ends equal to the reference, whose gear hash is checked against a byte-at-a-time loop at the small size |
| `swalign`, `swalignfit`, `swalignpad` | A random 128-base read; 16384 random windows of 256 bases, 1 in 16 holding a copy with ~6% substitutions and an indel | Every window's best score equal to a vectorised numpy Gotoh (clamped E / F), itself checked against the textbook recursion at the small size; every planted copy scores at least half the perfect score, no random window does |
| `jpcolor` | 2^20 random points in the unit square, edges within the radius that gives ~8 neighbours (4.19M edges), distinct random priorities | Every vertex's colour equal to the reference rounds, which at the small size equal sequential greedy colouring in priority order; the colouring proper |
| `postings` | 4096 sorted lists of distinct document ids below 2^20, lengths max(16, 65536 / k^0.8) (1.43M postings), ids skewed toward popular documents; 65536 random pairs of different terms | Every query's (count, xor of id * 0x9E3779B1) equal to numpy's `intersect1d` |

## Offline sanity checks (not hardware verification)

- **lavapipe:** all pass at both sizes. Last whole-suite rerun at the small size: 157 of 157 functionally correct.
- **Shim-compiled Mesa 25.2.8 and 26.0.3 code on the functional simulator:** 138 of 138 buffer workloads (plus quadmip and clustered).
  - Last whole-suite rerun (`simall`, Mesa 26.0.3): 153 compiled workloads.
    - 144 pass.
    - 5 are hardware-only (texture, image and indirect workloads, UNSIM).
    - 4 fail by design under the default `bfeil` semantics: `viterbi`, `shiftfuse`, `lz4dec` and `bitap` (note 37).
      `cdc`, added since, is a fifth (note 70).
      All 4 pass with `AGX_BFEIL_MASKED_SHIFT32=1`.
  - The 4 workloads without compiled code need features the offline Asahi driver does not offer (NODEV).
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
37. **Viterbi decoding, and an open question about `bfeil`.**
    - The traceback tests bit `st & 31` of a survivor word as `(x >> (st & 31u)) & 1u`, with `st` up to 63.
      Mesa 25.2.8 and 26.0.3 compile this to `bfeil r, 0, x, st, mask 0x1`: NIR drops the `& 31` (`ushr` already
      takes its shift modulo 32), and `fuse_ubfe` (`agx_nir_algebraic.py`) turns shift-and-mask into
      `ubitfield_extract`, emitted as a `bfeil` with an immediate mask and the unmasked shift.
    - Whether that is correct depends on how `bfeil` *with an immediate mask* treats a shift of 32 or more, and nothing
      recorded so far shows it.
      - The one recorded case, `alu/bfeil_r14_r6_r7_r8`, is the full-width form with no mask field: the form Mesa emits
        for `ushr`. It takes 7 bits of the shift (32 or more give 0), which is why Mesa re-adds `& 31` to every `ushr`
        in `agx_nir_lower_algebraic_late`.
      - That pass runs after `agx_nir_fuse_algebraic_late` ("after fusing constant shifts"), but `fuse_ubfe` matches
        variable shifts too.
    - **Hypothesis A** (the simulator's default, extrapolated from the full-width recording): the masked form also
      takes 7 bits.
      - Then this is a Mesa bug. `viterbi` fails (states 32-63 read 0, about 30% of bits wrong), and so do the three
        `shiftfuse` columns, `lz4dec`, `bitap` and `cdc` (notes 40, 70).
      - A minimal fix would match only constant shifts in `fuse_ubfe` (`'#b'`), or emit
        `ubitfield_extract(a, iand(b, 31), bits)`.
    - **Hypothesis B**: the masked form takes its shift modulo 32.
      - Then the compiler is right and the simulator must change. Set `AGX_BFEIL_MASKED_SHIFT32=1` to simulate it:
        all of these workloads then pass, and the 721-case suite is unchanged (691 / 721 either way).
      - B is the more plausible one a priori. `lz4dec`, `bitap` and `cdc` show the fused form arising from the most
        ordinary byte / 2-bit unpacking, `(w >> ((i & 3) * 8)) & 255` (or `& 3`), and the driver is
        Vulkan-conformant.
    - No existing hardware data covers the masked form. The testsuite's only `bfeil` case is full-width, and the
      Apple-M2-Playground ALU bit-flip runs (`results/2026-10-11-alu-bitflip`) flipped bits 52, 53 and 62 of
      `bfeil r5, r6, r7, r8`, all of which decode with `m = 0`.
    - The M2 run decides. A: `viterbi`, the three `shiftfuse` columns, `lz4dec`, `bitap` and `cdc` fail, and `viterbifix` passes.
      B: everything passes, and `AGX_BFEIL_MASKED_SHIFT32` semantics should become the default.
    - It is also decided at the instruction level. `make asahi-verify` records the coverage catalog's 70 bitfield
      cases (`bfi` / `bfeil` / `extr` / `shlhi` / `shrhi` with immediate shifts 0-127 and masks 0x1 to 0x7FFFFFFF, about
      a minute) and runs `scripts/bitfield_question.py`. That script checks each recorded case under both semantics
      and prints the verdict, which goes into the report.
    - `viterbifix` writes the bit test as `bitfieldExtract` with an explicitly masked offset, which the compiler keeps,
      so it is correct under both hypotheses.
    - Predicted time (`viterbifix`): about 3.6 ms for 8192 frames, dominated by 256 barrier-separated steps per
      workgroup.
38. **Shift-mask fusion probe.**
    - `shiftfuse` maps which idioms produce the masked `bfeil` with an unmasked shift. On code from Mesa 25.2.8 and
      26.0.3 these are exactly the three columns where a shift right by `s & 31` is followed by a constant low mask:
      `bit_test` (`& 1`), `byte_extract` (`& 0xFF`) and `double_and_mask` (`s & 63 & 31`, then `& 0x7FFF`).
    - In the other seven columns the compiler keeps the `& 31` or fuses nothing: the funnel shift, rotate, byte
      insert, shift-left-then-mask, arithmetic-shift-then-mask and signed `bitfieldExtract` columns, and a
      variable-width mask.
    - The simulator fails those three columns under hypothesis A and passes all ten under B (note 37). lavapipe
      passes all ten. On the M2, `probes` lists the failing columns.
39. **BC1 texture compression.**
    - The driver keeps the 16 texels in registers: 140 GPRs, so few SIMD groups per core are resident. The M2 time
      checks the model's occupancy limit on a kernel with little memory traffic: 64 B in and 8 B out per block.
    - Predicted 0.10 ms. `psnr_db` shows that the blocks decode to a usable image.
40. **LZ4 decompression.**
    - Each invocation decodes one 4 KiB block: token nibbles, 255-continued lengths, literal runs and back-references,
      including overlapping copies. Decoding is "safe" (input bounded by the block, bad offsets stop the block), so a
      wrong decode shows up as `short_blocks`, never as stray memory accesses.
    - The streams come from a small greedy LZ4 block encoder in the harness. The reference `lz4` Python library decodes
      all of them to the original data, so this is the real format.
    - Its byte read `(comp[i >> 2] >> ((i & 3) * 8)) & 255` compiles to `bfeil ..., i * 8, mask 0xFF`: NIR folds
      `(i & 3) * 8` to `i * 8` under `ushr`'s modulo-32 shift. So it decodes only under hypothesis B of note 37: the
      simulator fails every block by default and passes with `AGX_BFEIL_MASKED_SHIFT32=1`.
    - Predicted (under B) 4.1 ms: byte-serial, with a device read-modify-write per output byte.
41. **NTT polynomial multiplication.**
    - The arithmetic of lattice post-quantum schemes (Kyber / Dilithium use the same transform with other moduli).
      Each butterfly does a 32-bit multiply and three reductions modulo a constant, which the compiler turns into
      multiply-high sequences. The stage structure puts 8 + 8 barriers per workgroup.
    - Predicted 0.25 ms for 8192 products. The M2 time checks integer multiply-high throughput and the barrier cost
      that `barriercost` measures on its own.
42. **Radix select.**
    - In the duplicate-rich arrays every key of a pass falls in a few bins, so the shared-memory `atomicAdd`
      histogram serialises on the same words. That is the contended case of `smembank` / `atomcontend`, inside a real
      algorithm.
    - Predicted 0.83 ms. Its digit extract is a masked `bfeil` with shift at most 24, so it is correct under both
      hypotheses of note 37.
43. **Particle binning.**
    - Passes 0 and 2 are device atomics on 65536 counters, but clustered particles make a few cells take more than
      a thousand updates each.
    - The model does not serialise atomics on the same address (`atomcontend` is predicted flat), so it predicts
      0.19 ms for each of those passes. The 0.26 ms one-workgroup scan in between is serial work the model does
      represent.
    - A count or scatter pass much slower than 0.19 ms on the M2 measures same-address contention inside a real
      algorithm. Compare it with `atomcontend`'s hot-counter rows.
44. **Borůvka MST, and the cost of predicting dispatches from the initial buffers.**
    - The functional simulator carries buffer state across all 120 dispatches and reproduces the MST exactly.
    - `predict` times each dispatch from the initial buffers, so all 24 rounds are costed as round 1. There every
      edge joins two components and does two `atomicMin` operations; in later rounds most edges are inside one
      component and return early.
    - The default prediction (43 ms) is therefore expected to be above the M2 time.
    - `predict --stateful` times each dispatch from the buffers the previous ones left: the functional simulator
      advances them between dispatches, so it is sequential and slower (77 s for this workload). It gives 32.2 ms.
      - The hooking and pointer-jumping passes fall from 195 / 20 µs in round 1 to about 10 µs.
      - The winning-edge pass falls from 1176 to 617 µs.
      - The `atomicMin` pass stays at about 617 µs in every round: each edge still loads its 16 bytes and two random
        `parent` words before it can return early, so the pass is bound by those loads whatever the state.
    - The M2's per-dispatch times say which of the two predictions to trust for multi-pass algorithms.
      `verify --timing` therefore also predicts the multi-pass workloads (`bfs`, `sssp`, `ccl`, `pagerank`,
      `boruvka`, `lookback`, `cellsort`) with `--stateful`. Their rows carry `predicted_stateful_us` and
      `ratio_stateful`, and the summary's `stateful_vs_default` says which prediction each recording favours.
45. **Default vs stateful predictions of the multi-pass workloads** (large size, Mesa 26.0.3 code).

    | workload | default (µs) | `--stateful` (µs) | stateful / default | why |
    |---|---|---|---|---|
    | `bfs` | 919 | 2947 | 3.21 | from the initial buffers only the source is on the frontier, at every level |
    | `sssp` | 858 | 1929 | 2.25 | the same: relaxation work grows as distances settle in |
    | `cellsort` | 639 | 1154 | 1.81 | the scatter pass costed with zero cursors (all particles on few slots) vs real offsets |
    | `boruvka` | 42947 | 32230 | 0.75 | later rounds return early for most edges |
    | `ccl`, `pagerank`, `lookback` | equal | equal | 1.00 | control flow and addresses do not depend on the evolving data |

    - The default under-predicts frontier-driven algorithms by 2-3x and over-predicts converging ones. `verify --timing`
      reports both for these workloads (about 4.5 minutes extra on the M2), and the recording says which is right.
46. **Fixed-point biquads.**
    - Each sample step is a dependent chain through 4 sections, each with 5 64-bit multiply-adds, which the driver
      emits as 20 `imadd` to register pairs. The 4096 steps per invocation are serial, so the time is set by that
      chain's latency times the steps, divided by how many channels each core keeps resident (52 GPRs).
    - Predicted 4.5 ms. The M2 time measures the latency of 64-bit `imadd`, which no other workload depends on.
47. **Distance transform.**
    - Pass 0 has one invocation per row: 512 invocations, 8 workgroups on a 10-core GPU, each scanning 768 pixels
      twice through dependent loads. Predicted 295 µs, set by one core's latency on a mostly idle GPU: how the
      model handles under-filled dispatches.
    - Pass 1 is 200M multiply-add-min steps with coalesced row reads, predicted 466 µs.
48. **Mersenne Twister.**
    - Each workgroup first runs `init_genrand` serially: 623 dependent multiply-xor-shift steps on one invocation
      while 255 wait at a barrier. Then 17 twists, each in 3 barrier-separated phases plus a single-invocation step,
      and parallel tempering.
    - Predicted 0.48 ms. The serial seeding and the phases with 29-89 idle invocations of 256 make it sensitive to
      how the model charges barriers and partly idle workgroups.
49. **Bitap approximate matching.**
    - Each text position is a dependent chain of about 15 shift / and / or operations across the three error levels;
      1024 + 26 positions per invocation.
    - The base read `(text[i >> 2] >> ((i & 3) * 8)) & 3` compiles to `bfeil ..., i * 8, mask 0x3`, so the
      simulator finds no matches under hypothesis A and every match under B (note 37). Predicted 0.31 ms (under B).
50. **Batched Cholesky.**
    - The driver computes `sqrt` as `rsqrt_special` and `rcp`, and the division with `rcp`. Both are among the
      transcendentals where the functional simulator still differs from the recorded M2 by an ulp on some inputs (the
      30 suite failures waiting for more hardware data).
    - Expect CLOSE rather than PASS on the M2: `words_differing` shows how those last-bit differences spread through a
      factorisation. The backward-error check passes either way.
    - Predicted 0.57 ms. Each of the 16 columns costs three barriers with an active region shrinking from 256 to 1
      invocation.
51. **RSS hashing.**
    - The Toeplitz hash tests each of 96 (or 64) input bits and xors in a 32-bit window of the key, so the work is
      data dependent per lane (about half the bits are set). With the checksum fold and the byte swaps it is
      integer-ALU bound, at 24 bytes read per packet.
    - Its bit tests compile to masked `bfeil` with shifts of at most 31, so it passes under both hypotheses of note
      37. Predicted 1.24 ms for 1M packets.
52. **Poly1305.**
    - Each 16-byte block is 25 64-bit products and a carry chain, the arithmetic of the authenticator in TLS 1.3 and
      WireGuard. Lengths of 0-1023 bytes make lanes run 0-64 blocks, so a SIMD group runs as long as its longest
      message.
    - Correct under both hypotheses of note 37. Predicted 0.32 ms for 32 MB.
53. **Parallel CRC-32C.**
    - Pass 0 is 4096 invocations (64 workgroups), each a 4096-step chain where every step is a shared-memory table
      lookup that depends on the previous one: latency at low occupancy. Predicted 534 µs.
    - Pass 1 merges up a 12-level tree in one workgroup, each merge a 32-column GF(2) matrix-vector product
      (data-dependent bit tests). Predicted 168 µs.
    - Correct under both hypotheses of note 37.
54. **SipHash-2-4.**
    - Each 8-byte word costs 2 rounds of 14 64-bit operations, split into 32-bit halves. A 64-bit rotation by 32 is a
      swap of the halves, and 13 / 16 / 17 / 21 are two funnel shifts each. Lengths of 0-255 bytes give lanes 1-32
      words.
    - Correct under both hypotheses of note 37. Predicted 0.26 ms for 33 MB.
55. **BLAKE2s, and what scratch memory costs.**
    - `blake2s` is the straightforward GLSL: loops over the 8 G mixes with the state index and the SIGMA-permuted
      message index computed at run time. The driver cannot keep dynamically indexed arrays in registers. It puts
      them in 416 bytes of scratch per invocation and still uses all 256 GPRs.
    - `blake2sfast` is the same function with every G call written out (generated from SIGMA): about 90 GPRs and no
      scratch.
    - Predicted 13.3 ms against 0.15-0.18 ms, about 80x; lavapipe shows 36x. The M2 ratio measures scratch
      bandwidth and latency and register-limited occupancy together. The `spill` workload measures that in isolation.
56. **Q4_0 GEMV.**
    - The decode-time matrix-vector product of quantised LLMs. Each 18-byte block is read as bytes: 18 `device_load
      i8` per block in the compiled code, which no other workload uses at this rate. The bytes are unaligned to 4,
      as in llama.cpp's format.
    - Predicted 200 µs for 9.4 MB of weights. A memory-bound kernel whose bytes arrive through narrow loads: the M2
      time shows whether byte loads coalesce as well as word loads do.
    - `vk_run` gained `require storage_buffer_8bit` / `storage_buffer_16bit`. Its feature table length is now taken
      from the array, which it previously hard-coded.
57. **Attention decode.**
    - One workgroup per head is only 32 workgroups for 10 cores, each walking all 4096 positions twice. That is why
      production decoders (flash-decoding) also split the sequence across workgroups and merge the partial softmaxes.
    - Predicted 2.4 ms, latency-bound on an under-filled GPU while streaming 64 MB of fp16 K / V. Together with
      `q4gemv` it covers the two halves of an LLM decode step.
    - `attndecsplit` is the flash-decoding form: 512 chunk workgroups, each writing its max, exponential sum and
      partial V sum, then a 32-workgroup merge. Predicted 0.62 + 0.015 ms, 3.9x faster. On lavapipe it is slightly
      slower (40.7 against 35.6 ms): a CPU has few cores to fill.
    - The M2 ratio of the pair measures what occupancy is worth when streaming a KV cache.
58. **RMSNorm + RoPE, and sin / cos of large angles.**
    - RoPE rotates by pos · θ_i with θ_0 = 1, so at position 4095 the shader takes `sin` / `cos` of 4095 rad, as
      llama.cpp's Vulkan backend does. Vulkan specifies their precision only on [-π, π].
    - The driver computes them with `sin_pt_1` / `sin_pt_2` after its own range reduction, `pow` with `exp2`, and the
      normalisation with `rsqrt`.
    - Most of the error is the fp32 angle itself (one ulp of 4095 is 2.4e-4 rad). lavapipe reaches 3.7e-4 and the
      simulator 3.2e-4.
    - `probes` reports the M2's `max_rel_error`, which shows how the hardware's range reduction behaves at real RoPE
      angles. Predicted 0.2 ms.
59. **Packed int8 dot products.**
    - The shader uses `OpSDotKHR` with `PackedVectorFormat4x8Bit`, emitted through `GL_EXT_spirv_intrinsics`
      because the installed glslang predates `GL_EXT_integer_dot_product`. The SPIR-V is the same.
    - The Asahi driver advertises `shaderIntegerDotProduct`, but AGX has no packed-dot instruction. NIR lowers each
      dot to 8 byte sign-extensions (a `bfi` / `asr` pair each) and 4 `imadd`: 112 `bfi` / `asr` and 33 `imadd` in
      the loop.
    - Predicted 64 µs for 16 MB of weights, memory-bound if the lowering keeps up. The M2 time shows whether it
      does.
    - `vk_run` now queries and enables the Vulkan 1.3 features (`shaderIntegerDotProduct`) on 1.3+ devices and
      accepts `require integer_dot_product`.
60. **Camera ISP.**
    - Every pixel reads up to 9 raw samples around it, through 16-bit loads that neighbouring invocations share,
      branches on its colour-filter position, and does a 3 × 3 integer matrix and a gamma lookup.
    - Predicted 1.9 ms per 12 MP frame (about 520 frames per second). Whether it is bound by L1 hits or by the ALU,
      the M2 time says.
61. **Marching squares.**
    - Only about 2.7% of cells emit a segment, so most invocations exit after classifying. The rest diverge by case
      and do one or two 64-bit divisions per endpoint, which the driver expands into a long sequence (12 KB of code).
    - A first version counted cases with one global `atomicAdd` per cell (4M updates on 16 words). The model
      charged most of the time to those, at 1.6 ms. Counting per workgroup in shared memory first, as real code
      does, gives 0.47 ms.
62. **Hough transform.**
    - The kernel is 168 bytes: 180 iterations of two multiply-adds, a shift and a device `atomicAdd`. Its time is the
      device atomic rate: 6.8M atomics predicted at 239 µs.
    - Points are shuffled, so a SIMD group's lanes usually hit different counters. Peaks are hot because every
      point on a line votes into the same bin at that line's θ.
63. **Vector SpMV.**
    - With one subgroup per row, a row of n nonzeros costs about n / 32 steps instead of n. That is the point of the
      vector kernel when row lengths vary over 3 orders of magnitude.
    - Rows of fewer than 32 nonzeros leave lanes idle: most rows here are short. The M2 time against the model's
      405 µs shows how that idle-lane cost is charged.
    - The float64 references of `spmv` and `spmvsg` sum each row with `np.bincount` over per-nonzero row ids. The
      earlier `reduceat` form dropped the last nonzero of the final non-empty row when the matrix ends in empty rows
      (`spmvsg`'s small size does), and would have failed outright on `spmv` if its last row had been empty.
64. **Varint serialisation.**
    - The first workload storing single bytes through 8-bit storage (`device_store i8`): each value writes 1-5 bytes
      at offsets that neighbouring invocations share words with.
    - Predicted 2.1 ms, of which 1.77 ms is the serial offsets pass over 16384 workgroup counts, the same pattern as
      `rle`'s (note 30). The two M2 times give two measurements of a dependent device load-add-store chain.
65. **BVH ray traversal.**
    - Each lane walks its own path through the tree, so a SIMD group runs as long as its deepest traversal and
      re-converges only at the end. Node and triangle fetches are dependent loads from data-dependent addresses.
    - The traversal stack lives in shared memory (48 entries × 64 invocations = 12 KiB), so occupancy is limited by
      shared memory rather than by registers (80 GPRs, no scratch).
    - Predicted 0.46 ms for 65536 rays: the M2 time checks the model on divergent pointer chasing.
66. **AdamW, and a streaming kernel above the DRAM rate.**
    - 4M elements × 28 bytes is 117 MB, predicted in 202 µs: about 580 GB/s, above the 360 GB/s DRAM rate the model
      uses for this machine (the recorded suite's device is an M2 Max).
    - The cause is in the memory model. A load is charged L1 throughput plus the bandwidth of the level its lines
      come from. A store is charged L1 store throughput only, and its lines' level comes from the cache contents.
      `adamw` stores to the lines it just loaded, so its stores count as hits, and the write-back of the 50 MB of
      dirty lines to DRAM is never charged.
      - Per wave, each core touches 768 distinct lines: exactly the four read streams, 49 KB. At the per-core DRAM
        rate (360 GB/s over 38 cores) that is about 7250 cycles, against the 7630 simulated.
      - So the prediction is read-bandwidth bound: 67 MB / 360 GB/s ≈ 186 µs.
    - Read the M2 time against two numbers:
      - ≈ 190-200 µs: the model is right that write-back overlaps or is cheap.
      - ≈ 325 µs, 117 MB at 360 GB/s: write-back costs DRAM bandwidth like reads. Then dirty-line write-back should
        be charged to the DRAM bucket. That affects every store-heavy streaming workload, so check the change against
        `dispatch_check`'s recorded dispatches.
    - The driver turns the four divisions into one `rcp` and one `rsqrt_special`, folding the uniform bias
      corrections.
67. **STREAM: bandwidth by access kind.**
    - Predicted GB/s, counting bytes read and written: copy 357, scale 318, add 308, triad 291, read 316, fill 213.
    - `probes` puts the M2's measured GB/s next to these per kind. Read against fill shows the read / write asymmetry.
      Copy and triad against read show what write-back costs, the question `adamw` (note 66) raises for
      read-modify-write streams.
    - `AGX_CHARGE_WRITEBACK=1` makes every store take DRAM bandwidth for its lines when the working set does not fit
      the SLC (off by default). The two predictions (µs):

      | | default | `AGX_CHARGE_WRITEBACK=1` |
      |---|---|---|
      | `adamw` | 202 | 328 |
      | `stream` copy / scale / add / triad / read / fill | 377 / 422 / 654 / 693 / 212 / 315 | 377 / 481 / 993 / 969 / 212 / 571 |

    - Copy and read do not change. Fill rises most, so the default under-charges even stores to lines never loaded,
      not only `adamw`'s read-modify-write.
    - Whichever column the M2 times follow is the setting to keep. If neither, the right charge lies in between, and
      the per-kind GB/s from `probes` give it.
    - No hardware data recorded so far measures writes. The memory-bound dispatches in
      `results/2026-10-11-dispatch` (`stream` cases 22-24) are read-only loops: 8 loads summed per iteration, one
      store at the end. Case 24 reads 249 MB in 739 µs (337 GB/s), which is where the model's DRAM rate comes from.
      Write-back charging therefore moves `dispatch_check` by under 1% (its 36 dispatches pass either way), and only
      `adamw` and `stream` on the M2 can decide it.
68. **`motionest`: full-search motion estimation.**
    - 3600 workgroups of 256 invocations. Each invocation scores 4 or 5 of the 1089 candidates at 256 absolute
      differences each, two shared-memory loads per difference. 1089 = 4 × 256 + 65, so the fifth round runs 65 of
      256 lanes: about 22% of the issued SAD work is idle lanes.
    - Predicted 4.92 ms (19 waves of 5 workgroups per core). That is 2.0 × 10^11 useful absolute differences per
      second, or about 3.8 per core-cycle.
    - The tie order (SAD, |dx| + |dy|, dy, dx) makes the answer unique, so the check is word for word against numpy.
      The 25.2.8 and 26.0.3 builds both pass on the simulator under both `bfeil` semantics, so this one does not
      depend on question 1.
    - On the M2, the time against the prediction tests the model's shared-memory load throughput on a realistic
      kernel: the existing probes (`smembank`) measure it in isolation.
69. **`jsonidx`: JSON structural indexing.**
    - The masks of simdjson's first stage, for one 8 KiB document per workgroup. A byte is structural if it is one
      of `{ } [ ] : ,` and is not inside a string. A quote is real unless an odd run of backslashes precedes it.
    - Two carries cross the 32-byte chunks: whether the chunk starts escaped, and whether it starts inside a string.
      - The escape carry is a function of the carry in: an all-backslash chunk passes it on unchanged. Each
        invocation computes its escapes for both carries in, and a workgroup scan composes the functions.
      - The in-string carry is an xor scan of real-quote parities.
      - Both are Hillis-Steele scans in shared memory, 8 steps, 2 barriers each.
    - The test data exercises both carries. At the small size, 64 chunks end inside a backslash run, and 631 of
      1024 chunks change if escapes are ignored.
    - Mesa unrolls every loop, the scans included: the main program is 1141 instructions with no backward branch,
      447 of them `icmpsel` (the byte compares). Its 70 `bfeil` give the same result under both semantics of
      question 1.
    - Predicted 333 µs for 32 MiB, about 100 GB/s, well under the modelled DRAM rate.
    - The check is exact, so it must PASS whatever the timings.
70. **`cdc`: content-defined chunking for deduplication.**
    - FastCDC's gear rolling hash, `h = (h << 1) + gear[byte]`, cuts a stream where the top 13 bits of `h` are zero.
      Backup tools cut this way so that an edit only changes the chunks around it.
      - Pass 0 marks the candidate cuts, one invocation per 32 bytes. A 32-bit `h` forgets a byte after 32 shifts, so
        31 bytes of history give the exact hash and the invocations are independent.
      - Pass 1 is the greedy chunker, serial, one invocation per 1 MiB region. A chunk ends at the first candidate
        at least 2 KiB in, at 64 KiB, or at the region end.
    - The data is a backup stream: versions of six files, each version with a few small edits. The check reports what
      deduplication by chunk hash stores. At the large size, 13.9% of the stream with these chunks, against 99.7%
      with fixed-size chunks of the same mean size (10.1 KB), because an insertion shifts every later fixed
      boundary.
    - Predicted 1.84 ms for pass 0 (64 MiB, about 36 GB/s) and 4.12 ms for pass 1. Pass 1 is 64 serial invocations,
      a single workgroup on one core: on the GPU the serial pass costs more than the parallel one. The M2 time tests
      the model's single-core latency over a long data-dependent loop.
    - **It also depends on question 1 (note 37).** The shader reads byte `i` as
      `(data[i >> 2] >> (8 * (i & 3))) & 255`. Mesa drops the `& 3`, since `8 * (i & 3)` and `8 * i` agree modulo 32,
      and fuses shift and mask into `bfeil ..., mask 0xFF` with the shift `i << 3`.
      - Under the default semantics (7 bits of the shift), bytes 4 and on of every loop read 0. The simulator then
        gets 120 of 4096 candidate words wrong, at both Mesa versions.
      - Under `AGX_BFEIL_MASKED_SHIFT32=1` everything passes.
      - This is the most ordinary byte-unpacking idiom there is, which adds to the case for B.
71. **`swalign`, `swalignfit`, `swalignpad`: Smith-Waterman alignment and the shared-memory occupancy cliff.**
    - Local alignment with affine gaps (match +2, mismatch -3, gap of k bases -(5 + 2k)), one reference window per
      invocation, as GPU aligners do for short sequences. Each cell is a handful of integer max / add operations
      and one shared-memory load and store of the packed H / E column.
    - E and F are stored clamped at 0: H only ever uses max(0, E), and the clamped recursion gives the same H. The
      reference uses the same form, and the small size checks it against the textbook recursion.
    - At the large size every planted read is found (score at least 128 of a perfect 256). No random window gets
      there: the best scores 27-33.
    - The three:
      - `swalign` keeps the query in shared memory: 32768 + 512 = 33280 bytes per workgroup. The model's 64 KiB per
        core then holds one workgroup, two SIMD groups.
      - `swalignfit` reads the query from a uniform buffer, a word of 4 bases at a time (constant shifts, the inner
        loop unrolled by 4): exactly 32768 bytes, two workgroups per core.
      - `swalignpad` is `swalignfit` with 128 unused words added to the shared array by a specialization constant:
        33280 bytes. Its compiled main program is byte for byte `swalignfit`'s.
    - Predicted (busiest core, cycles per wave):

      | | shared bytes | workgroups per core | waves | Mcycles per wave | ms |
      |---|---|---|---|---|---|
      | `swalign` | 33280 | 1 | 7 | 14.4 | 72.2 |
      | `swalignpad` | 33280 | 1 | 7 | 11.3 | 56.8 |
      | `swalignfit` | 32768 | 2 | 4 | 11.7 | 33.5 |

      - `swalignpad` against `swalignfit` is occupancy alone. A second workgroup costs almost nothing per wave, so
        the model sees each core latency bound with two SIMD groups.
      - `swalign` against `swalignpad` is the code: per-cell shared-memory query loads against an unrolled
        uniform-buffer word.
    - On the M2, `swalignpad` / `swalignfit` measures whether 512 bytes of shared memory halve the occupancy, at
      identical code. That tests the 64 KiB-per-core rule (`tg_capacity`, `rust/src/dispatch.rs`) at its cliff.
    - All three use `bitfieldExtract` with an explicitly masked offset, or constant shifts, so none depends on
      question 1. All pass on the simulator at both Mesa versions under both semantics, and on lavapipe.
72. **`jpcolor`: Jones-Plassmann graph colouring.**
    - Each round, an uncoloured vertex whose priority beats every uncoloured neighbour's takes the smallest colour
      no coloured neighbour has. Vertices coloured in the same round are never adjacent, so the colouring is proper.
      For given priorities it equals sequential greedy colouring in decreasing priority order; the small size
      checks that.
    - Rounds read the previous round's colours and write the other copy (by a push-constant parity). Reading and
      writing one copy would race: a vertex could see a neighbour coloured in the same round and wrongly take it as
      settled.
    - At the large size: 25 rounds, 15 colours (maximum degree 24).
    - Predicted 28.8 ms by default: every round is timed from the initial, all-uncoloured buffers, 1151 µs each.
      `--stateful` predicts 9.1 ms, falling from 1151 µs in round 1 to 17 µs by the last rounds, as nearly every
      vertex is coloured and only copies its colour. It is in `STATEFUL`, so `verify --timing` reports both
      (question 3 of `tasks/m2-run-checklist.md`).
    - The check is exact and the colouring independently checked, so it must PASS whatever the timings. It passes
      on the simulator at both Mesa versions under both `bfeil` semantics.
73. **`postings`: posting-list intersection.**
    - One invocation per query walks the shorter list and gallops through the longer one from the last match: an
      exponential probe, then a binary search. This is how inverted-index engines answer AND queries.
    - The work per query is the shorter list's length (at most), which runs from 16 to over 10000. Lanes of a SIMD
      group wait for the longest, so only 27.9% of lane slots in the outer loop do work
      (`outer_loop_lane_utilisation`, from the list lengths).
    - Document ids are drawn skewed toward popular documents, so results are realistic: 291834 matches, 6.7% of
      queries empty.
    - Predicted 2.11 ms. All 1024 workgroups fit in one wave.
    - On the M2 the time tests the model on divergent, data-dependent device loads, a binary search's dependent
      chain. The check is exact against numpy, independently of the shader's algorithm. It passes on lavapipe, and on
      the simulator at both Mesa versions under both `bfeil` semantics.
