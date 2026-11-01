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

## Experiments on the resident-stream gap (not adopted)

Scored on the 40 llama.cpp / Geekbench captures (mean |error| 10.5%, 36 within 25%) and the 8 above (mean 21.4%):

- A GPU-wide cap on system-level-cache hits (each active core's `slc_rate` at most 408 GB/s / active cores) plus
  `later_waves_warm` only for footprints up to 6.5 MB: 40 captures mean 10.3% (36 within 25%), the 8 above 8.0%
  (read 8 MiB +2%, copy -2%, FFT -17..+13%). Not adopted: the calibration rows of stream.json (all 38 cores
  re-reading a 1-16 MiB window inside one dispatch) measure 1.1-1.9 TB/s, so the cache itself is not capped
  GPU-wide; with the cap their p90 error goes 0.395 -> 1.096 (max 2.30 -> 3.29) and dispatch_check's max 0.215 ->
  0.630.
- The physical reading that fits both: data a *previous command* left behind is not served at cache speed on its
  first touch (the 8 MiB read back to back runs at ~340 GB/s), re-reads inside a dispatch are. Turning residency off
  entirely (`resident_slc` false) gives the 8 above mean 9.9% but the 40 captures 12.4% (34 within 25%), because
  `later_waves_warm` then never applies.
- Charging only a resident line's first touch in the dispatch to the DRAM bucket (the rest at the cache's rate) did
  not reproduce the non-resident time (read 8 MiB: 9.6 us against 26.7 us without residency): the non-resident path
  differs by more than the bucket. Next: find that difference (per-level latency, the LRU sizes, or the store and
  first-wave handling), then model "first touch per dispatch" with it and re-score both sets.
