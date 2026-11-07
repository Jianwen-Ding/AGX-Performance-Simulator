# Stack-using kernels: each SIMD-group slot's first stack use in a command costs ~1.15 us, one slot at a time per core

Measured 2026-11-07 on the M2 Max (G14C) with the Rodinia bfs `Kernel2` capture (clang-built CUDA through ZLUDA,
80 bytes of stack per thread, /scratch/sim-captures/rodinia/bfs), replayed with `agx_run_hw --stack 80
--back-to-back 300` at other grid sizes and workgroup sizes (the kernel's bounds check keeps every access inside its
buffers). Times in us (firmware, warm):

| workgroups x threads | 1 x 32 | 1 x 128 | 1 x 256 | 1 x 512 | 38 x 512 | 244 x 512 | 977 x 512 | 1954 x 512 | 38 x 32 | 380 x 32 | 3800 x 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hardware (1000 back to back) | 7.7 | 11.6 | 16.6 | 26.5 | 46.2 | 132.5 | 140.1 | 147.8 | 8.5 | 21.4 | 45.9 (300) |
| model before | 11.3 | 11.3 | 11.4 | 11.6 | 11.7 | 14.6 | 21.0 | 31.6 | 11.3 | 11.6 | 20.5 |
| model after | 8.8 | 12.3 | 16.9 | 26.3 | 44.8 | 121.3 | 127.8 | 138.4 | 10.0 | 20.6 | 72.7 |

* One workgroup: +1.15 us per SIMD group beyond the first (4 / 8 / 16 groups). The same kernel launched empty
  (no stack instruction executed) with `--stack 80` costs only the known ~8 us set-up (5.6 -> 17.0 us at 244 x 512).
* Many workgroups: the time grows until every resident slot of a core (6 workgroups x 16 groups = 96) has been used
  once, then hardly at all (244 -> 1954 workgroups: +15 us for 8x the work): 96 slots x 1.15 us = 110 us.
* Workgroups of one SIMD group (38-3800 x 32) pay less than their resident slots predict (3800 x 32: +64% in the
  model): short groups free their slot before the next is set up. Not modelled.
* Synthetic stack probes (`scripts/probe_stack_fresh.py`) faulted (Bad address) even with the backend's first stack
  instruction copied in; 13 faults recorded, probing stopped. Only the captured kernel was used.

Model (dispatch.rs, zluda profile): a kernel with a stack adds (resident workgroups x groups per workgroup on the
busiest core - 1) x `stack_slot_ns` (1150 at 1398 MHz, scaled with `--clock`; `AGX_STACK_SLOT_NS`, 0 in the drm
profile) to the core's time. The fixed stack set-up of the zluda profile drops from 7.5 to 5.0 us: real one-group
stack kernels take 7.7 (bfs Kernel2 1 x 32) and 8.0-8.9 us (gaussian Fan2) in all at a warm clock, of which 2.3 is
the launch (an empty kernel launched with `--stack 80` measures +8-11 us, which these real kernels contradict).

**Measuring short kernels: 300 back-to-back commands are not enough.** The Rodinia warm times of capcheck_hw.txt
(`--b2b 300`) for kernels under ~40 us were taken before the clock ramped: at `--back-to-back 1000` (twice each)
gaussian Fan1 35.2 -> 26.3 us, Fan2 19.4 -> 8.0 / 8.9, particlefilter sum_kernel 13.6 -> 6.8 (model 6.5: the
"-52%" was the measurement), lud_internal 6.6 -> 5.5, bfs Kernel2 1 x 32 9.5 -> 7.7; 44 us and longer unchanged.
`capcheck_hw_b2b1000.txt` is capcheck_hw.txt with those times replaced.

## Held out

Rodinia warm captures (`capcheck.py --hw-from capcheck_hw.txt`, `rodinia-warm-*.txt`; bfs Kernel2 is the fit data):

| launch | before | after |
| --- | --- | --- |
| bfs `Kernel` (112 bytes of stack) | -72% | -12% |
| gaussian `Fan1` (1 x 512, 64 bytes) | -68% | -19% |
| pathfinder `dynproc_kernel` (160 bytes) | -29% | -4% |
| gaussian `Fan2` (1 group) | -39% | -39% |
| hotspot3D (in-run time only, cold clock) | -33% | -17% |
| srad_cuda_2 (in-run time only) | -37% | -34% |

Median over the 12 distinct warm launches: 30% -> 15.5%. With the 1000-command times (`capcheck_hw_b2b1000.txt`,
`rodinia-warm1000-*.txt`): original model (both 2026-11-07 changes off) 29.5%, threadgroup-load change only 29%, both
changes and the 5.0 us set-up **6.5%** (10 of 12 within 15%; open: particlefilter likelihood +31%, find_index -29%).

In-run (cold) times (capcheck.txt, the launches inside the Rodinia programs): no single `--clock` fits them
(`rodinia-inrun-clock*.txt`: median 42% at 1398 MHz, 32.5% at 500): most launches take 2.6-3.0x their warm time
(clock ~500 MHz) but heartwall (10 ms) runs at the top clock and bfs Kernel2 only 1.6x. A clock state that depends on
the program's recent GPU activity would be needed; not modelled. Kernels without a stack are unchanged; verify-dispatch
(drm profile) and verify-cost-model unchanged. By hand from the stored 2026-11-01 llama.cpp report (stack versions
of mul_mat_vec_f, captures no longer kept): 32 threads x 95 workgroups per core in 4 waves (24 slots) -72% ->
~-15%; 128 threads x 24 per core in 4 waves (24 slots) -39% -> ~+10%.
