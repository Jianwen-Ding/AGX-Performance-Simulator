# zluda_fft_agx's threadgroup-memory FFT (agx_fft_shared) against the GPU

M2 Max (G14C), zluda_agx (AIR route), captures of `zluda_fft_agx/tools/fftbench.c` launches (`ZLUDA_AGX_CAPTURE`),
hardware `agx_run_hw --back-to-back 1000` (warm), model `agx_dispatch --launch-profile zluda` (default rules; the
opt-in AGX_TG_SETTLE_ON_READ gives the same times here). Each launch transforms 8 MiB in place (16 MiB moved);
124 registers.

| launch (n x batch, threads, threadgroup bytes) | GPU us | model us | model wgs/core | error |
|---|---|---|---|---|
| 16 x 65536, 4 x 64, 32 KiB | 81.2 | 64.6 | 2 | -20% |
| 64 x 16384, 16 x 16, 32 KiB | 98.5 | 79.4 | 2 | -19% |
| 256 x 4096, 64 x 4, 32 KiB | 116.6 | 97.5 | 2 | -16% |
| 1024 x 1024, 256 x 1, 32 KiB | 136.2 | 119.9 | 2 | -12% |
| 16 x 65536, 4 x 64, 8 KiB | 69.8 | 56.8 | 3 | -19% |
| 256 x 4096, 64 x 4, 8 KiB | 98.2 | 86.2 | 3 | -12% |
| 1024 x 1024, 256 x 1, 8 KiB | 114.0 | 100.2 | 3 | -12% |
| 2048 x 512, 512 x 1, 16 KiB | 162.6 | 132.5 | 1 | -18% |

- The model predicted the gain of declaring 8 KiB instead of 32 KiB (more resident workgroups) in direction and
  size: -12% (64.6 -> 56.8) against the GPU's -14% (81.2 -> 69.8); ZLUDA 94735f1 made the change.
- It is 12-20% fast throughout, an amount that follows the bytes moved (11-19 us at 16 MiB) more than the
  workgroup count or shape: the next place to look is the memory model's read and write throughput for in-place
  streams of this size (8 MiB, inside the system-level cache), not the barrier or threadgroup-memory costs.

## The same gap on plain streams

`stream/readx.ptx` (16 bytes read per thread, no writes) and `stream/copyx.ptx` (16 read, 16 written), 256-thread
workgroups, captured and replayed the same way:

| kernel, bytes per direction | GPU us | model us | error |
|---|---|---|---|
| read 8 MiB | 24.5 | 8.4 | -66% |
| read 32 MiB | 88.7 | 97.8 | +10% |
| copy 8 MiB | 46.0 | 31.4 | -32% |
| copy 32 MiB | 185.3 | 183.5 | -1% |

The GPU moves ~340-380 GB/s (reads plus writes) whether the data fits the system-level cache or not; the model is
accurate for streams larger than the cache (DRAM, `dram_gbps` 360) and 2-3x fast for streams it takes as resident
(`resident_slc`). Ruled out as the cause: `slc_rate` (0.5 -> 0.106 segments per cycle per core, DRAM's
equivalent, moves read 8 MiB only 8.4 -> 11.7 us), `later_waves_warm` (off: 10.0 us), `max_waves` (2 / 8 / 64: no
change). Next: trace where a resident stream's time goes in `simulate_groups` (which bucket or latency binds).
