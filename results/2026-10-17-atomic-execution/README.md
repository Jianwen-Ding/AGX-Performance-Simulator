# Cached functional atomic operands and integer replay coverage

Offline x86_64 Linux, Rust 1.99.1. Baseline:
`14553e49fe6bbfddfa323f0996046ab4bc120e3f`. No new GPU runs or timing calibration.

## Change and accuracy boundary

Functional device and threadgroup atomics cache immutable encoding fields once
per visited PC, not once per execution/group. Register values, addresses, active
masks, memory and execution order remain dynamic. Boxed atomic payloads avoid
enlarging ordinary decoded instructions. Established float/u64 variants, access
suppression and whole-group termination behavior are preserved.

Threadgroup non-CAS operations no longer read the unused next source register.
Previously, an add with its source at the last modeled register failed even
though no compare operand was needed. Synthetic regressions cover all ten
non-CAS integer operations, returning and non-returning, at that boundary.
CAS tests check both matching and non-matching comparisons and still require
the next register. `register-boundary.json` records baseline failure/current
success with the same checked example (`1 2 1 1 31`). This boundary case is not
a new hardware observation. Existing integer semantics are checked against the
retained standard suite; variant behavior is checked against the playground's
74 recordings and independent evidence verifier.

## Focused CPU benchmark

Five alternating baseline/current pairs, seven measured executions after a
warm-up per invocation. Both builds use the same release toolchain/options
(`-C linker-features=-lld`); the example is identical in both checkouts. Final
timings were collected after tests, evidence replay and report checking ended.
Values below are medians of invocation medians; raw runs are in `benchmark.json`.

| Active lanes | Atomic kind | Baseline ms | New ms | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 1 | device | 5.507 | 3.229 | 1.705x |
| 1 | threadgroup | 4.012 | 2.183 | 1.838x |
| 32 | device | 42.509 | 42.298 | 1.005x |
| 32 | threadgroup | 14.444 | 13.050 | 1.107x |

Each dispatch executes 512 groups and 128 returning atomic adds/group on
independent per-lane counters. Every lane's final return and final memory word
are checked, as are instruction and group counts. Preparation, dispatch setup
and execution are timed; mapping and result validation are outside the timer.
The 32-lane device result is effectively flat, not a meaningful speed claim.

ALU-only controls are retained in `alu-controls.json` and a second complete
five-pair run in `alu-controls-repeat.json`. Full dispatch is effectively flat
(16.486 -> 16.489 ms; repeat 16.345 -> 16.358 ms). Compact standalone cost tracing
varies from 5.3% slower to 5.8% faster, and four-worker padded tracing also varies.
Single-worker padded standalone tracing is consistently slower: 14.642 ->
16.671 ms (13.9%) and 14.895 -> 15.899 ms (6.7%). Cost-trace source is unchanged;
the cause is not established. Profile this regression before claiming an
overall improvement. These measurements describe CPU simulation, not GPU speed.

Reproduce after building examples and binaries in both checkouts:

```sh
python3 scripts/bench_atomic_execution.py \
  --baseline /path/to/baseline/atomic_execution --output /path/to/benchmark.json
python3 scripts/bench_cost_trace.py --baseline /path/to/baseline/cost_trace \
  --baseline-dispatch /path/to/baseline/agx_dispatch --output /path/to/controls.json
cargo test --offline --release --manifest-path rust/Cargo.toml
rust/target/release/suite_check /path/to/retained-suite
python3 scripts/check_dispatch_reports.py
```

## Validation

- All 25 Rust tests and doc checks pass (`rust-tests.txt`).
- All 358 selected retained recordings compare exactly (`suite-check.txt`),
  including all 44 device and 44 threadgroup standard integer-atomic cases:
  add/sub/exchange/CAS, signed/unsigned min/max and bitwise operations,
  returning/non-returning, distinct/shared addresses. This is the previous
  276-case selection plus 82 additional atomic cases, not the whole testsuite.
- `replay-manifest.tsv` lists compressed-file Git blob hashes, checked against
  the baseline commit's tree. Retained evidence is unchanged.
- All 74 atomic-variant records replay with zero failures; the independent
  playground evidence verifier also reports zero failures (logs alongside).
- Playground `make replay-integer-atomic-suite` passes all 88 standard cases.
- The cost-model verifier reports zero failures (`cost-verifier.txt`).
- All 36 complete prediction reports match the existing SHA-256, without
  rebaselining (`prediction-check.json`).

No M2 device is available in this x86_64 worker (`/dev/dri` absent). Next:
investigate the padded standalone control regression, guard malformed performance
launch dimensions, and profile real data-dependent memory/atomic workloads.
