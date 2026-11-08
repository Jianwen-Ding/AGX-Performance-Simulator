# The cuFFT library's (fft_agx) threadgroup-memory FFT (agx_fft_shared) against the GPU

M2 Max (G14C), the CUDA driver (AIR route), captures of `fft_agx/tools/fftbench.c` launches (`AGX_RUNTIME_CAPTURE`),
hardware `agx_run_hw --back-to-back 1000` (warm), model `agx_dispatch --launch-profile runtime` (default rules; the
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
  size: -12% (64.6 -> 56.8) against the GPU's -14% (81.2 -> 69.8); the PTX front end's commit 94735f1 made the change.
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

## Partial warm (opt-in, `AGX_PARTIAL_WARM=1`)

A first attempt at "first touch per dispatch" failed because an access is re-evaluated while it waits for its
buckets: only its first evaluation saw new lines. Measured per wave instead (read 8 MiB, busiest core): 2135 cycles
resident, 7361 not (768 lines / 0.106 per cycle at the DRAM rate: the DRAM bucket binds). `later_waves_warm`
replayed each later wave's own data into the caches, so a stream's later waves cost nothing. `partial_warm` replays
only the segments the first wave also touched (the first wave runs first and records them); a later wave's new
data then comes from DRAM.

| model | 40 captures (mean, within 25%) | the 8 above (mean; read 8 MiB, copy 8 MiB, FFT) |
|---|---|---|
| default | 10.5%, 36 | 21.4%; -66%, -32%, -12..-19% |
| partial warm | 12.1%, 34 | 6.1%; -6%, -12%, -1..+11% |

It fixes the streams and pressure_solver (-14% -> +1%), but llama.cpp's mul_mat_vec_f (+8% -> +43%) and
mul_mat_vec_q<Q5_1> (+0% -> +23%) and depthwise_convolution_2d (+1% -> +30%) then come out slow: their later waves
read new data, yet fast on the GPU (about 110 GB/s, so not bandwidth: latency the full warm hid). So data a previous
command left behind is sometimes served from a cache at full speed (those kernels, 2-6 MB per wave) and sometimes
not (8 MiB streams). Left opt-in until that difference is measured: next, the same read kernel at 1 / 2 / 4 / 6 MiB
back to back, to find the size at which the previous command's data stops being fast.
`scripts/evalset_captures.sh` scores both sets (paths are this machine's captures).

## mul_mat_vec_f under the new runtime profile (+8% -> +21%)

Per simulated wave (busiest core, cycles): with partial warm 8326 (wave 0, 480 distinct lines), 10275 / 10359 / 9719
(waves 1-3, 608 lines each, sharing none with wave 0); before, waves 1-3 were warm (all hits) at ~8320. The GPU's
25.3 us leaves ~8k cycles per wave after the launch, about what wave 0 costs: the later waves are slower in the model
only because they touch 27% more lines, at L2 latency and the DRAM rate (608 lines / 0.106 per cycle = 5736 cycles
of DRAM time, so the rest is latency the model does not hide). Skipping the warm replay for waves that share
little with the first changes nothing here (they already share 0%). Next: the memory-level parallelism of
first-touch loads (how many a core keeps in flight), measured with a read kernel of 1 / 2 / 4 / 8 loads per thread.

Measured (`mlp/readL.ptx`: L independent 16-byte loads per thread, 256-thread workgroups, the same bytes in all,
runtime profile with partial warm): GPU / model us for 8 MiB: L1 24.4 / 25.2, L2 24.8 / 25.8, L4 25.2 / 26.8, L8
26.0 / 26.7; 32 MiB: 87.9 / 97.8, 89.5 / 97.5, 91.9 / 97.1, 93.5 / 96.6 (+3..+11%). Streaming reads are
bandwidth-bound on the GPU at every L and the model follows them: memory-level parallelism is not what
mul_mat_vec_f's later waves get wrong. What differs there: f16 loads of partial lines, the shared-memory reduction
after them, 6 workgroups of 128 threads per wave; next, its instruction profile against wave 0's.

Issue profile of mul_mat_vec_f (SIMD group 0, ticks before issue, `AGX_ISSUE_PROFILE_SKIP` 2 = wave 0, 3 = wave 3):
wave 0 79501 (threadgroup_barrier 20914, fadd32 17292, simd_shuffle_xor 16366, no device_load stall in the top);
wave 3 97181 (threadgroup_barrier 27506, fadd32 17519, simd_shuffle_xor 16588, device_load 14554). A later wave
pays a memory phase (its rows at the DRAM rate) and then the reduction; the model sums waves one after another,
while on the GPU workgroups are dispatched as others finish, so one wave's loads overlap the previous wave's
reduction. The old full warm hid that by making later waves' loads free. The fix to try: overlap consecutive waves
in `dispatch` (a wave's memory-bound part against the previous wave's compute-bound tail), checked on these 48
captures and verify-dispatch.

Tried (not kept): crediting each later wave the overlap min(memory phase, compute tail), the memory phase its
distinct lines at the core's DRAM share and the tail the rest of its time, averaged over simulated later waves.
40 captures 11.0 -> 11.6% mean (35 within 25%), the 8 above 5.9 -> 18.5% (the FFT captures -24..-40%): kernels with
several resident workgroups per core already overlap memory and compute inside a wave, so a blanket cross-wave
credit counts it twice. A cross-wave overlap would need the wave's own idle time (cycles in which no group issues)
from `simulate_groups`, not an estimate from its line count.

## Adopted: chained waves (runtime profile, `chain_waves`)

Each simulated wave runs chained with the next one in every slot (the second wave's trace of a slot appended to the
first's, step indices offset): on the GPU a slot starts its next workgroup as soon as the previous one ends, so one
wave's loads run under the other's reduction or tail; the pair is simulated together and each wave gets half its
cycles. The footprint, the written bytes and the sharing stay the first wave's own (computed before chaining: with
them doubled, store-bound kernels such as gradient went +1 -> +92%).

| runtime profile | 40 captures (median, mean, within 25%) | the 8 above | all 48 (mean) |
|---|---|---|---|
| partial warm | 8%, 11.0%, 36 | 5.9% | 10.2% |
| + chained waves | 7.5%, 10.2%, 37 | 5.9% | 9.5% |

Moves of 4 points or more: hough_scores +27 -> -12%, blur_gray +38 -> +32%, image_rotate +35 -> +28%,
convolve_vertical +13 -> +7%, raw_process +11 -> +2%, gradient +1 -> +5%, color_look_up_tetrahedral -3 -> -7%,
mul_mat_vec_q<Q5_1> -3 -> -7%, mul_mat_vec_f +21 -> +17%. verify-dispatch with chaining forced on: unchanged (max
0.215, p90 0.091).
