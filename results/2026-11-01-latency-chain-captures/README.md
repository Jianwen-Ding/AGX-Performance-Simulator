# Latency-bound chains: the zluda profile's first touch at L2 latency is 50% fast on them

Eight captures where a few SIMD groups make dependent loads to lines new in the command, scored by
`scripts/evalset_captures.sh` as `lat8` (hardware: `agx_run_hw --back-to-back`, b2b median, M2 Max G14C):

| Capture | hw us | default (first touch in L2) | `AGX_FIRST_TOUCH_L2=0` (first touch in SLC) |
|---|---|---|---|
| chase, one line (playground first-touch-pages) | 8.79 | -7% | -5% |
| chase, new line per hop | 23.67 | -48% | +19% |
| chase, new page per hop | 24.21 | -50% | +16% |
| ZLUDA agx_qr_panel (1024 rows) | 514.7 | -3% | -1% |
| ZLUDA agx_jacobi_round_sg (n 256) | 7.50 | 0% | +53% |
| ZLUDA agx_sgemv_split (1024 x 16384) | 188.0 | +10% | +10% |
| ZLUDA agx_sgemv_sum | 23.2 | -52% | +16% |
| ZLUDA agx_strsm_sg (64-block, 64 vectors) | 62.5 | -55% | -29% |
| lat8 mean abs | | 28.1 | 18.6 |
| 40 llama / Geekbench captures, mean | | 10.2 | 10.8 |
| 8 stream / FFT captures, mean | | 5.9 | 4.9 |
| all 56, mean | | **12.1** | **11.0** |

## What happened

`--launch-profile zluda` turned on `partial_warm`, which does two things: later waves find only the first wave's
segments warm (this is what the stream / FFT captures need: without `partial_warm` they are up to 66% fast), and
`first_touch_dram`, which puts every segment of the dispatch into L2 before timing ("as if prefetched", fitted to
llama.cpp's matrix-vector kernels). The second makes a dependent load to a new line cost an L2 hit. The chase
captures were within +16% when the playground finding was written and fell to -48% once the profile gained
`partial_warm`; nothing in the 48-capture set was a dependent chain, so the evaluation did not see it.

Hardware says a new line in a command is not in L2: compute command boundaries invalidate the GPU-wide L2
(playground novel-results/2026-10-16-l2-command-boundary), and a dependent load to such a line costs ~237 ns more
than an L1 hit (2026-11-01-first-touch-pages), about the model's SLC latency.

`AGX_FIRST_TOUCH_L2=0` keeps `partial_warm`'s later-wave rule and leaves first touches at the SLC. It is the
physical rule and better over all 56 captures, but not everywhere: the 40-capture mean is 0.6 points worse and the
Jacobi round becomes +53% (128 groups reading two 1 KiB columns with independent loads, data written by the
previous command; whatever makes it and llama's matrix-vector reads fast on new lines, an L2 hit is not the
mechanism). Left opt-in until that is understood; the `lat8` group keeps the evaluation honest about chains.

## Follow-up probes (playground novel-results/2026-11-01-first-touch-chains)

Two explanations for the fast Jacobi round were tested on hardware and refuted: lines written by the GPU in the
previous command are not faster on first touch (chase storing into its lines 24.9 us vs 25.0 control), and wide or
sequential hops are not prefetched (SIMD-group chases of 64-256 B hops, sequential or 4 KiB apart: 21.8-26.0 us).
The default rule is 50% fast on all nine such probes and AGX_FIRST_TOUCH_L2=0 within +12-29% (sim 28.1-28.4 us).
The Jacobi round's issue profile (AGX_ISSUE_PROFILE) with the SLC rule: 75% of the group's 11.8k cycles in the two
row-loop waits (8 each, ~464 cycles a wait); hardware's 5.2 us of work allows ~140 ns a wait, or less of the rest.
Next: a modified-kernel probe (the round without its V rotation; with the sums only) to split the two.

Captures: /scratch/perf-goal/sim/chase/s{0,64,16384}, /scratch/perf-goal/sim/cap4 (regenerable: the playground's
chase.ptx with agx_bench_kernel; ZLUDA's qrbench / eigbench / l2bench with ZLUDA_AGX_CAPTURE, SKIP 5, MAX 1).
