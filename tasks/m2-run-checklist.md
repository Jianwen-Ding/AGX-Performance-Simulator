# After `make asahi-verify`: what the M2 run decides

`results/<date>-asahi-verify/README.md` is the report. This page lists the questions the first hardware run settles,
where each answer appears, and what to change for each outcome. The details are in the numbered notes of
`tasks/spirv-08-applications.md` ("note N" below).

## Questions with a code change attached

| # | Question | Where to look | If ... | then ... |
|---|---|---|---|---|
| 1 | How does `bfeil` with an immediate mask treat shifts of 32 or more? (note 37) | The report's "bfeil shift question" line (`scripts/bitfield_question.py` over the 70 recorded bitfield cases) | verdict B: the shift is taken modulo 32 | make `AGX_BFEIL_MASKED_SHIFT32` semantics the default in `bfeil_shift` (`rust/src/isa/exec.rs`); `viterbi`, `shiftfuse`, `lz4dec` and `bitap` should then PASS |
| | | | verdict A: 7 bits, as the full-width form | Mesa's `fuse_ubfe` is unsound for variable shifts; `viterbi` / `shiftfuse` (3 columns) / `lz4dec` / `bitap` FAIL on the GPU too; report upstream with note 37's root cause and `viterbifix` as the workaround |
| 2 | Do SR58 / SR59 count active lanes? (offline checks: `bfs`) | `bfs` status | PASS | confirmed; nothing to do |
| | | | FAIL with counts in multiples of 32 | the simulator's active-lane reading is wrong for this case |
| 3 | Should multi-pass workloads be timed from the evolving state? (notes 44-45) | the `stateful_vs_default` list in the verify summary | `stateful` is closer for the frontier / converging workloads | make `--stateful` the default for them in `predictall` |
| 4 | Are the 30 transcendental suite failures 1-ulp table differences? | `cholesky` (CLOSE expected), `rmsrope` `max_rel_error` in probes; `make record-transcendentals` records the exact fp32 `rcp` / `rsqrt` tables | words differ only in the last bit | the simulator's `rcp` / `rsqrt_special` / `sin_pt` need the recorded tables |

## Measurements to read against the model

| Workload(s) | What the M2 time measures | Model's prediction |
|---|---|---|
| `blake2s` vs `blake2sfast` (note 55) | cost of dynamically indexed arrays in scratch, at identical work | 13.3 ms vs 0.17 ms |
| `attndec` vs `attndecsplit` (note 57) | what filling the GPU buys when streaming a KV cache | 2.45 ms vs 0.63 ms |
| `spmv` vs `spmvsg` (note 63) | scalar vs subgroup-per-row SpMV, on different matrices: `spmv` rows of 0-63 nonzeros, `spmvsg` power-law rows with idle lanes | 802 µs, 405 µs |
| `atomcontend`, `cellsort`, `hough` (notes 26, 43, 62) | same-address atomic contention, which the model does not serialise | flat |
| `rle`, `varint` (notes 30, 64) | a serial chain of dependent device load-add-stores (16384 steps) | 1.77 ms each |
| `lookback` (note 32) | release / acquire hand-off between cores; `mean_lookback_steps` in probes | 55 µs (too low: no flag chain) |
| `q4gemv`, `varint` (notes 56, 64) | 8-bit loads and stores against word access | 200 µs, 2.1 ms |
| `q8dot` (note 59) | AGX's lowering of packed int8 dot products (no native instruction) | 64 µs |
| `biquad`, `poly1305`, `siphash` (notes 46, 52, 54) | latency / throughput of 64-bit `imadd` and 64-bit ARX on 32-bit ALUs | 4.5 ms, 0.32 ms, 0.26 ms |
| `edt` pass 0, `crc32c` pass 0 (notes 47, 53) | under-filled dispatches: latency at low occupancy | 295 µs, 534 µs |
| `boruvka` (note 44) | per-round dispatch times, default (43 ms) vs stateful (32 ms) prediction | 43 / 32 ms |
| `nms` (note 10) | a known under-prediction: work grows with the workgroup index | low |

## Results that must hold whatever the timings

These have published or independent references, so a FAIL is a real bug on one side:

- `rsshash`: Microsoft's RSS verification hashes.
- `poly1305`: RFC 8439 §2.5.2.
- `siphash`: the SipHash paper's outputs.
- `blake2s` and `blake2sfast`: RFC 7693 Appendix B.
- `mt19937`: 3499211612 and 4123659995 for seed 5489.
- `crc32c`: 0xE3069283.
- `sha3`: `hashlib`.
- `lz4dec`: streams that decode with the `lz4` library. It depends on question 1.
- `edt`: scipy's EDT.
- `bvhrays`: a brute-force ray-triangle test.
