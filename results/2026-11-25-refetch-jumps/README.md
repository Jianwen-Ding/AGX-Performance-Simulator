# Refetched jump targets: the fetch overlaps the straight run before the jump (2026-11-25, hardware)

`scripts/probe_fetch_groups.py` with the new `SKIP=R,D` option: a loop of T trips over a body of about 48 KB of
executed iadds. After every R executed 64 B lines, a `jmp_exec_any` skips D lines of `stop` filler. The body is
beyond the 12 KiB i-cache, so every trip after the first refetches it. One SIMD group per core, 38 workgroups, b2b
1000, two repeats agreeing within 0.4 us ([hw.txt](hw.txt)).

| Shape | hw T=2 / T=8 (us) | warm trip, hw | model (defaults) T=8 | model, experiment T=8 |
|---|---|---|---|---|
| no skip | 83.7 / 224.1 | 23.4 | 226.1 | 225.2 |
| R=4, D=4 | 114.4 / 304.0 | 31.6 | 569.7 (+87%) | 299.2 (-2%) |
| R=4, D=32 | - / 311.9 | - | 576.2 (+85%) | 305.0 (-2%) |
| R=1, D=8 | 424.3 / 1063.3 | 106.5 | 1623.9 (+53%) | 1070.2 (+1%) |
| R=16, D=16 | 85.5 / 230.2 | 24.1 | 313.5 (+36%) | 241.2 (+5%) |

- A warm jump target the i-cache lost costs ~138 ns when every line is one (R=1). After 4 streamed lines it costs
  only ~15 ns more than a streamed line, and after 16 nothing. The target's fetch starts well before the jump
  issues: it overlaps the run of lines before it.
- A `wait 0` with nothing outstanding does not stop the code stream (R=16 padding).
- The model charged a flat 200 ns from the jump's issue.

## Experiment (off by default): `AGX_REFETCH_LOOKAHEAD=1 AGX_IDLE_WAIT_STREAM=1`

A taken jump's target is requested when the group's current straight run began, at 140 ns warm
(`AGX_REFETCH_JUMP_NS`). A first touch keeps its jump-distance cost, which is measured from the run's start, +30 ns
(`AGX_COLD_RUN_NS`). An idle wait no longer restarts the stream. It fits the probe (-2..+5%) and keeps
`verify-jump-fetch` at 2.6%.

| Set | Defaults | Experiment |
|---|---|---|
| tinygrad (5) | 13.0% | **10.4%** (r_64_16_32 +21 → -3; E_8_8_8_16_4 +7 → -13) |
| llama fa (4) | 5.5% | 6.5% (prefill kernels 2 → -3, -9 → -12) |
| llama tg (10), pp512 (20) | 8.2%, 8.7% | identical |
| held-out (160) | 6.8%, w10 125 | 7.0%, w10 124 (bucketcount -21 → -40, mergeSortPass -15 → -23, likelihood +5 → -6) |

It is not adopted: the held-out set and fa get slightly worse. Bounding the overlap by the lines streamed in the run
(no overlap for 1-line runs) does not help bucketcount (-39%) and breaks the R=1 fit (1308 us). bucketcount's gap
is in how groups share each other's fetches (see AGX_STREAM_OWNER, 2026-11-10), not in this rule.
