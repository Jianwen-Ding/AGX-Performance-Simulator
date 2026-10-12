# Task SPIR-V 7: algorithms (radix sort, FFT, CRC-32, k-means, int8 GEMM, 3-D stencil, int16, transcendentals)

**Workloads:** `radix_sort`, `fft`, `crc32`, `kmeans`, `int8dot`, `stencil3d`, `int16`, `trig`. These are recorded and
verified by `make asahi-verify` with the others; they are **PENDING** until then. The detailed procedure is in
`tasks/spirv-workloads.md`.

## Why these

| Workload | What it adds | In the Mesa 26.0.3 code (`spirv/compiled/mesa-26.0.3/<w>/main.txt`) |
|---|---|---|
| `radix_sort` | A multi-pass algorithm: 24 dispatches, each depending on the last; duplicate keys, so stability is tested | Per-thread digit arrays lowered to 284 `icmpsel` selects; threadgroup stores and loads |
| `fft` | Shared-memory butterflies with twiddles from `sin` / `cos` | Stage loop unrolled: 36 `sin_pt_1` + 36 `sin_pt_2`, `floor`-based range reduction, 44 threadgroup loads and stores |
| `crc32` | A lookup table built in shared memory, then data-dependent table reads | 64 `threadgroup_load`s at data-dependent addresses, `bfeil` byte extraction, `xor` chains |
| `kmeans` | Float compares choosing an index (argmin) | `fcmpsel` selects the best distance and index; centroids read with `threadgroup_load` in a `while_icmp` loop |
| `int8dot` | 8-bit signed loads (sign extension) into int32 multiply-add | `device_load 0, i8` byte loads; sign extension through `.sx` operands, `asr` and `bfi`; `imadd` |
| `stencil3d` | 3-D workgroups (8×8×4), six clamped neighbours | 3-D thread ids, `icmpsel` clamps |
| `int16` | 16-bit integer arithmetic and storage (wrapping, shifts, signed min, widening) | 16-bit `iadd` / `imadd` forms, 16-bit stores |
| `trig` | Every common transcendental, on ranges where Vulkan bounds its precision | `sin_pt_1` / `sin_pt_2`, `rcp`, `rsqrt_special`, `fmadd32` polynomial steps |

The integer workloads (`radix_sort`, `crc32`, `int8dot`, `int16`) are exact. `fft`, `kmeans`, `stencil3d` and `trig`
have tolerance-based references; their verification is the word-for-word comparison with the recording.

## Offline sanity checks (not hardware verification)

- **lavapipe:** all eight pass at both sizes. `trig` prints each function's worst error as a fraction of the Vulkan
  bound. lavapipe's `exp` reaches 1.04× on a million inputs, so the verdict allows 2×; the ratios themselves are the
  data.
- **Shim-compiled Mesa 25.2.8 and 26.0.3 code on the functional simulator:** 16 of 16 within the references.
- **Cost model:** a prediction for every dispatch (large).

## What to look at in the M2 results

- **`trig`:** the GPU's error ratios per function. With the recording, `make spirv-verify` compares the simulator
  word for word. The simulator's fp32 `rcp` / `rsqrt` / `exp2` / `log2` / `sin_pt` are correctly rounded and the
  hardware's are not (task #5), so expect CLOSE here, not PASS. Count the differing words per function; they are new
  data for fitting the hardware's transcendentals.
- **`fft`:** the same effect through the twiddles.
- **`radix_sort`:** time per pass (24 dispatches) vs `predict`; the single-group scan passes are latency-bound.
- **`int8dot`:** about 1.1 s on lavapipe at the large size. Compare the GPU's time with `predict` (byte loads, no
  tiling).
