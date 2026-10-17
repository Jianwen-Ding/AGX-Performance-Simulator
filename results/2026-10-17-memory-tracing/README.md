# Cached memory-trace operands and cross-lane replay

Offline x86_64 Linux, Rust 1.99.1. No GPU runs, new instruction semantics or
timing calibration changes. Baseline: `53c14cc60ff27102b2786768067f377f44b4e09b`.

## Implementation and accuracy boundaries

Both functional and approximate performance tracers now cache immutable memory
operands per visited PC. Lane masks, registers, addresses and memory remain
per execution. Memory-only boxed payloads keep ordinary instruction records
compact. Device segment vectors reserve space once, then retain the existing
sort/deduplicate behavior. Atomic and threadgroup vectors retain every active
lane/component, including duplicates and their original order.

The public uncached address helpers remain unchanged as compatibility/reference
paths. New tests compare capture against them across formats, shifts, uniform
and register bases, signed and immediate offsets, component masks, active masks
and missing fields. Separate expected-value tests pin broadcast segment sets
versus duplicate atomic and threadgroup words. The approximate tracer's existing
exclusion of `.TODO` accesses is preserved; the functional tracer still includes
them. No previously skipped format or operand is claimed as newly supported.

An adjacent-duplicate shortcut was tried but rejected after scattered-access
regressions. Its raw results are retained in `adjacent-dedup-trial.json`, explicitly
not the shipped implementation. The final version reserves once and inlines
the address helpers; `benchmark.json` measures that version.

## Focused CPU benchmarks

Five alternating baseline/current pairs, seven measured dispatches after a
warm-up per invocation; medians below are medians of invocation medians. Both
builds use the same release toolchain/options (`-C linker-features=-lld` on this
host). The new example is copied unchanged into the baseline checkout.

| Workers | Word stride | Tracer | Baseline ms | New ms | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1 | cost | 15.360 | 13.223 | 1.162x |
| 1 | 1 | functional | 16.034 | 13.763 | 1.165x |
| 1 | 16 | cost | 15.910 | 14.203 | 1.120x |
| 1 | 16 | functional | 16.039 | 14.312 | 1.121x |
| 4 | 1 | cost | 6.653 | 4.681 | 1.421x |
| 4 | 1 | functional | 5.003 | 4.682 | 1.069x |
| 4 | 16 | cost | 6.149 | 4.690 | 1.311x |
| 4 | 16 | functional | 5.013 | 4.747 | 1.056x |

Each dispatch traces 128 workgroups of 32 threads with 128 device loads/group.
Every complete PC sequence (17,280 instructions) and all 16,384 memory-access
entries are checked against independent expected segment sets. Register bases
let the approximate interpreter execute the loads rather than skip them. Stride
1 is coalesced; stride 16 touches a distinct segment per lane. Preparation and
tracing are timed; result verification is outside the timer.

`alu-controls.json` records the existing ALU-only controls. Compact standalone
cost tracing is 12.223 -> 11.846 ms; the ALU-only full-dispatch CLI control is
16.356 -> 16.767 ms (2.5% slower). This is a focused memory-tracing improvement,
not a claim that every workload gets faster.

Reproduce from the repository root, after building the examples and CLI:

```sh
python3 scripts/bench_memory_trace.py \
  --baseline /path/to/baseline/memory_trace --output /path/to/benchmark.json
python3 scripts/bench_cost_trace.py \
  --baseline /path/to/baseline/cost_trace \
  --baseline-dispatch /path/to/baseline/agx_dispatch --output /path/to/controls.json
python3 scripts/check_dispatch_reports.py --output /path/to/prediction-check.json
```

## Validation

- 23 Rust tests and documentation checks pass, including matrix-based cached versus
  uncached address checks and successful/empty prefix-selection CLI checks.
- All 276 selected retained functional recordings compare exactly: the previous
  146 cases plus 130 full/masked SIMD/quad reductions, scans, shuffles and ballots.
  This is not the whole testsuite. `replay-manifest.tsv` records compressed-file
  Git blob hashes; retained evidence at the baseline commit is unchanged.
- Playground `make replay-cross-lane-suite` passes all 130 cases. This does not
  replace its separate 16/64-bit `verify-cross-lane-width` evidence checker.
- All 74 atomic-variant replays pass; the cost-model verifier has zero failures.
- All 36 complete prediction reports match the prior paired-comparison SHA-256,
  including warnings and full-precision wave/timing values, not display rounding.
  `check_dispatch_reports.py` reads but never rewrites the committed reference;
  the result is recorded in `prediction-check.json`.

Next candidates: malformed performance-launch guards, cached atomic execution
operands and profiling real data-dependent memory traces. Float/u64 atomic timing
still needs hardware evidence; this pass does not calibrate it.
