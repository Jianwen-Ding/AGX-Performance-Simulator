# Cached functional special-register execution

Offline x86_64 Linux, Rust 1.99.1. Baseline:
`a0622df34d5f772c92d6f10f034413c1e6f4598c`. No new GPU execution or calibration.

## Implementation and semantics

Functional decoding caches `get_sr`'s selector, destination/width and value
class once per visited PC. Group-uniform values are resolved once per executed
instruction, not per active lane. Lane-dependent coordinates remain per lane.
Quad counts are computed once per quad; quad/SIMD ranks advance over the current
mask. No active-mask work is done for ordinary uniform/lane selectors.

Only encoding metadata is retained. Values are not cached across instructions,
groups, barrier resumes or launch contexts. Unknown/fallback values are resolved
lazily only when an active lane needs them. Destination validation, half-register
upper-half preservation, inactive writes and placement-read flags are unchanged.
The no-longer-used duplicate `Info.ops` operand vector is removed; compiled and
semantic helper operands remain owned by their existing execution paths.

The [retained special-register finding](https://github.com/Jianwen-Ding/Apple-M2-Playground/blob/0e62492a53b42e145532ca74826756b2e835a6a0/novel-results/2026-10-16-special-registers/README.md)
establishes that only SR56-59 follow divergence. SR16 still reports physical quad
position. SR17/SR20 remain placement-dependent, and SR113 remains unmodelled:
`AGX_SR_DEFAULT` is still an explicit software fallback, not a model of its
time-varying hardware state. No unknown encoding or new hardware behavior is
claimed. The approximate cost tracer's interpretation is not changed; functional
paths used by real-memory performance tracing share the optimized executor.

## Checked CPU benchmark

Five alternating baseline/current pairs, seven measured dispatches after a
warm-up per invocation, identical example source in both checkouts. Both use
release opt-level 3/fat LTO/one codegen unit and `-C linker-features=-lld`.
Builds and validation finish before timed comparisons. Baseline binary/source
SHA-256 hashes are in `baseline-metadata.json`; raw runs are in `benchmark.json`.

Each case has 256 workgroups of 32 lanes and 128 repeated target reads/group,
with all 32 lanes or the first 13 active. Every output word, inactive sentinel,
group count and instruction count is checked against independent expected values.
Preparation, group initialization and execution are timed; code construction,
mapping and result verification are outside the timer. These are CPU simulator
timings, not GPU instruction latency or overall-workload speed guarantees.

| SR | Active lanes | Baseline ms | New ms | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 13 | 3.056 | 1.565 | 1.953x |
| 7 | 32 | 6.488 | 2.756 | 2.354x |
| 18 | 13 | 3.157 | 1.385 | 2.279x |
| 18 | 32 | 6.896 | 2.055 | 3.356x |
| 19 | 13 | 3.045 | 1.428 | 2.132x |
| 19 | 32 | 6.203 | 2.028 | 3.059x |
| 56 | 13 | 1.905 | 1.368 | 1.393x |
| 56 | 32 | 3.850 | 2.141 | 1.798x |
| 57 | 13 | 1.712 | 1.594 | 1.074x |
| 57 | 32 | 2.902 | 2.715 | 1.069x |
| 58 | 13 | 1.590 | 1.386 | 1.147x |
| 58 | 32 | 2.675 | 2.175 | 1.230x |
| 59 | 13 | 1.715 | 1.269 | 1.351x |
| 59 | 32 | 2.894 | 1.973 | 1.467x |
| 80 | 13 | 3.236 | 3.204 | 1.010x |
| 80 | 32 | 6.495 | 6.593 | 0.985x |

Uniform reads improve 1.953-3.356x; global-X reads are effectively flat, with
the full-mask result 1.5% slower. This is a focused improvement, not a claim
that all shader paths get faster.

`alu-controls.json` records full-dispatch ALU CLI 16.964 -> 17.427 ms (2.7%
slower). `memory-controls.json` records mixed controls: functional memory tracing
is 1.0-4.5% slower and four-worker scattered cost tracing is 3.797 -> 4.475 ms
(17.9% slower). Cost-trace source is unchanged; code-layout/host effects are not
separately attributed. Prior same-binary A/A variability does not establish the
cause or erase these regressions. CPU profiling remains a follow-up, not a
resolved result. Both archived/current CLI controls use the same calibration
root through `AGX_SIM_REPO`; no parameter or timing expectation was changed.

## Validation

- All 36 Rust tests and doc checks pass (`rust-tests.txt`). New matrix tests
  cover all 255 modelled selectors, two destination widths and eight masks;
  mask ranks/counts use independent oracles. They also test changed launch
  context values, inactive unknown reads and invalid destinations.
- All 363 selected retained recordings compare exactly (`suite-check.txt`).
  This is not the whole testsuite; hashes match the existing replay manifest
  (`evidence-check.json`).
- All six retained SR programs pass on baseline and current: 1,518 selector/
  pattern vectors with every active lane value and all inactive table words
  checked. Exclusions remain SR17/SR20 and SR113. `sr-evidence.json` pins all
  seven unchanged program/readback blobs to the baseline Git tree.
- SR replay now rejects missing patterns/selectors or truncated lane values.
  Three schema/failure-injection tests pass under normal and optimized Python;
  invalid fixtures must fail before invoking the simulator.
- Playground `make replay-simulator-sr` passes; all 74 atomic-variant replays,
  independent atomic evidence verification and the launch/batch gate pass.
- The cost verifier reports zero failures. All 36 complete prediction reports
  match the existing SHA-256 without rebaselining (`prediction-check.json`).

Reproduce after building release examples and binaries in both checkouts:

```sh
python3 scripts/bench_special_registers.py \
  --baseline /path/to/baseline/examples/special_registers --output /path/to/benchmark.json
AGX_SIM_REPO="$PWD" python3 scripts/bench_cost_trace.py \
  --baseline /path/to/baseline/examples/cost_trace \
  --baseline-dispatch /path/to/baseline/agx_dispatch --output /path/to/alu-controls.json
python3 scripts/bench_memory_trace.py --baseline /path/to/baseline/examples/memory_trace
cargo test --offline --release --manifest-path rust/Cargo.toml
python3 scripts/test_sr_check.py
python3 -O scripts/test_sr_check.py
make -C ../Apple-M2-Playground replay-simulator-sr
python3 scripts/check_dispatch_reports.py
```

No M2 device is available (`/dev/dri` absent). Next: lower-level geometry guards,
resource-width replay and controlled CPU profiling of real data-dependent paths
and control regressions. No new hardware validation or timing rebaseline.
