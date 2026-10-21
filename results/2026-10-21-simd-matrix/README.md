# simd_matrix_fmadd32 / 16 in the simulator (2026-10-21)

Hardware records of blocker B15 (`tasks/ptx-re-blockers.md`; playground `results/2026-10-21-b15-simd-matrix`): 120 random
A, B, C sets per width run on the M2 Max (G14C), with the D words the GPU produced. `make verify-simd-matrix`
(`scripts/verify_simd_matrix.py`) replays them in agx_run: **0 of 11,520 words differ** (7,680 fp32, 3,840 fp16).

The functional model (`rust/src/isa/compiled.rs`, `K::Matrix`): element q = 2 lane + slot of each paired operand is
row ((q >> 2) & 3) + 4 (q >> 5), column (q & 3) + 4 ((q >> 4) & 1); D[r][c] = C[r][c] followed by
fma(A[r][k], B[k][c], acc) for k = 0..7, rounded to the operand type at every step (the fp32 unit's flush; fp16 keeps
denormals). A partial execution mask or source modifiers are refused (not measured).

The cost model uses the measured rows `simd_matrix_fmadd32` / `16` (16.2 cycles latency, 15.5 issue per SIMD group,
8.4 lanes per cycle per core), and `fmadd32_lane1` / `fmadd16_lane1` for the lane-offset fmadd (B18), merged into
`results/2026-10-11-instruction-bench/bench.json.gz`; `simd_shuffle_up` now has its own measured row.
