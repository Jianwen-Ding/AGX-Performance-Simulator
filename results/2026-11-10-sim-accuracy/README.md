# Simulator accuracy round 2026-11-10

Hardware: M2 Max (G14C, 38 cores), Asahi, AC power. Kernels are clang-built CUDA (llama.cpp, Rodinia, Halide, tinygrad)
captured through ZLUDA (`ZLUDA_AGX_CAPTURE`) and replayed with `agx_run_hw --back-to-back 5000` (a copy of ZLUDA's
agx_runtime whose back-to-back limit is raised; the median of the last 1000 commands, three repeats, idle GPU checked
before and after every run with `gpuusers.sh`). Model: `agx_dispatch --launch-profile runtime`.

## Measurement notes
- `--back-to-back 1000` leaves kernels under ~20 us below the top clock (the GPU climbs an 8-step clock ladder one
  step per ~5-7 ms of busy time; playground novel-results/2026-11-10-gpu-clock-ladder); 5000 reproduces the
  2026-11-07 references (`hw/b2b*`): h_laplacian median ratio 1.00 at 5000, 1.11 at 1000.
- CPU load does not change GPU timings (`probes/021-clock-cpu.log`).

## Model changes (see the commit messages for per-kernel numbers)
1. Device loads after ALU work wait until pending loads have passed the L1, not `done - extra`
   (`sweeps/005-sc-sweep.log`: streamcluster one workgroup 84 -> 36 us, hw 35.5).
2. Store floor by 32-byte sectors per store instruction; a group's recent lines merged; first stores to a line
   merged across the workgroups of a 256-workgroup block; later re-writes cost their sectors.
3. Four chained waves per slot (was two): many-wave per-wave costs were 20-25% high (`sweeps/050-sweeps.log`).
4. One-wave sampling of every core also when n <= cores (llama.cpp MMQ: only workgroups 0 and 37 were simulated);
   cores ranked by traced work when not all fit the cycle budget.
5. Cold code fetch 70 ns per line run into, 300 ns per line first entered by a taken jump (was 104 for all).
6. One-resident grids up to 4 x 34 workgroups: ceil(n / 34) rounds (playground
   novel-results/2026-11-10-one-resident-placement).

## Tools
`gsweep.py CAPDIR GRID... [--hw]` (a capture at other grid sizes on hw and in the model), `trunc.py` (copies of a
capture with a `stop` at the pc group 0 first reaches at trace index N: hw vs model time along the kernel;
`sweeps/070,077,078`), `scripts/probe_fetch_groups.py` (instruction fetch shared by SIMD groups; `probes/024,027,028`),
`AGX_TRACE_STATS=1`, `AGX_TRACE_DUMP=FILE`, `AGX_DEBUG_TIME=1`, `AGX_DEBUG_STORES=1`.

## Error (|sim - hw| / hw), before (origin/main) -> after (this branch)
| set | launches | median | p90 | mean | within 10% |
|---|---|---|---|---|---|
| held-out Rodinia + Halide (`eval/heldout-*`) | 159 | 7 -> 5% | 20 -> 18% | 9.0 -> 7.5% | 113 -> 119 |
| llama.cpp tg (`eval/llama-*`) | 10 | 10 -> 9% | 26 -> 19% | 11.6 -> 9.1% | 5 -> 5 |
| llama.cpp pp512 (`eval/pp-*`, new) | 20 | 16 -> 11% | 64 -> 53% | 27.5 -> 22.2% | 6 -> 7 |
| llama.cpp flash attention (`eval/fa-*`, new) | 4 | 7 -> 8% | 44 -> 44% | 15.0 -> 15.4% | 3 -> 2 |
| tinygrad (`eval/tg-*`, new) | 5 | 11 -> 13% | 110 -> 110% | 30.3 -> 29.9% | 1 -> 2 |

Open problems: MMQ second-round pairing (+11..+43%), rope (+117%: groups that exit at once), flash-attention pp
kernel (-44%), k_bin_bcast (-48%), tinygrad's r_64_16_32 (+110%).
