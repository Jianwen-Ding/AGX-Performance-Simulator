# Code fetch overlaps execution only partly: time = max + 0.3 min

After settle-on-read became the default, the four largest misses (mul_mat_vec_q 1 and 8 columns, soft_max,
gemm_non_block_size_bias, -15..-27%) all had the code fetch (`code_lines` x 104 ns) close to their execution time,
and the model took only the larger. The full overlap had been fitted to the Q8_0 matrix-vector kernel while threadgroup
loads were wrongly deferred (its execution time was inflated).

Probe (playground results/2026-11-01-fetch-overlap/fo.ptx, one thread, 32 dependent loads to new lines with P FMAs
after each, back to back 3000), ns:

| P | code lines | execution (model) | fetch | hw | (hw - launch - max) / min | model now |
|---|---|---|---|---|---|---|
| 0 | 10 | 10028 | 1040 | 14083 | (fetch too small) | 13340 |
| 8 | 16 | 10042 | 1664 | 12916 | 0.34 | 13542 |
| 24 | 28 | 10071 | 2912 | 13250 | 0.30 | 13944 |
| 56 | 52 | 10123 | 5408 | 14000 | 0.29 | 14745 |
| 120 | 100 | 10429 | 10400 | 15583 | 0.27 | 16549 |

Default now 0.3 (`AGX_FETCH_OVERLAP`). Evaluation: 40 captures 10.8 -> 9.2 (39 within 25%), lat8 7.2 -> 5.9, all 56
9.5 -> 8.2, all 64 9.9 -> 8.8 (0.5 fits the evaluation marginally better, 8.1, but 0.3 is the measured value).
Unit tests, dispatch_check, verify_cost_model, mem_check, scale_check unchanged.
