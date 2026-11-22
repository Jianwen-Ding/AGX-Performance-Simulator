# icmpsel / fcmpsel with `.discard` on a select operand (2026-11-22, hardware, not modelled)

The timing coverage rows `bench/{i,f}cmpsel_*_r8-cache_r9-discard` issue at 4.85 cycles per SIMD group, while the
model gives 2.10. `bench.py` and `bench2.py` separate the hints (run_bench, M2 Max, latency is 1.99-2.00 cycles in
every form):

| form (bench registers: X = r22, Y = r21) | simd_issue |
| --- | --- |
| `icmpsel ult d, d, r20, X, Y` (base, 4 x 32-bit reads) | 2.03 |
| X.cache / Y.cache | 2.03 / 2.04 |
| r20.cache (compare operand B) | 2.23 (fcmpsel 2.24) |
| r20.discard (compare operand B) | 2.04 |
| X.discard / Y.discard / both | 4.84 / 4.84 / 4.86 (fcmpsel Y.discard 4.87) |
| `icmpsel ult d, d, 0, X, Y.discard` (immediate B) | 3.90 |
| `icmpsel ult dl, dl, r20h, Xl, Yh.discard` (16-bit, high half) | 1.07 (no penalty) |
| `fcmpsel lt dl, dl, r20l, Xl, Yl.discard` (16-bit, low half; base 1.16) | 3.55 |

- `.discard` on a select operand adds about 2.8 issue cycles in the 32-bit form. A second discard adds nothing.
- `.discard` on a compare operand, and `.cache` on a select operand, cost nothing.
- **Not modelled.** No compiler on this project's path emits these hints. Across 561k instructions from the
  llama.cpp, Rodinia, tinygrad and Halide captures, 108k are cmpsel and none carries `.cache` or `.discard`. The
  16-bit half dependence and the immediate case would need more forms to fit a rule.
