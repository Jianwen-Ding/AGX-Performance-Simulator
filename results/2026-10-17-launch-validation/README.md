# Checked launch geometry, batch isolation and no-change controls

Offline x86_64 Linux, Rust 1.99.1. Baseline:
`38e13adde4a10dce7e60131c400352b0452d5f6a`. No GPU execution or calibration changes.

## Change and scope

`Launch::validate` rejects invalid grid/workgroup dimensions, workgroup-count
overflow, zero core/cluster entries, impossible configured occupancy/shared
memory, invalid uniform indices and non-finite/negative timing inputs. Validation
runs before instruction decoding/external tracing. The CLI validates before
reading shaders/preambles or materializing mapped buffers. Zero register halves
still normalize to eight; custom-core fallback and zero launch costs remain valid.
No arbitrary host-memory allocation limit is added.

Occupancy arithmetic uses a wide intermediate instead of overflowing `2 * regs`.
Impossible resource declarations no longer invent one resident workgroup; public
occupancy/topology helpers return zero rather than divide by zero on bad inputs.
Valid retained rules are unchanged, including register-pair allocation and the
configured threadgroup-memory capacity (64 KiB by default).

Workgroup coordinates retain their high bits instead of truncating a linear
index to u32 or overflowing an XY product. Representative workgroup selection
and group-count multiplication also use overflow-safe arithmetic. A synthetic
8,589,934,592-workgroup case tests the first/last waves and high Z coordinates
without executing that many workgroups. This is not hardware support or timing
validation for an extreme dispatch. Non-finite derived prediction time is an
error, not a silently serialized null.

CLI argument parsing is fallible. Dimensions require one to three components;
malformed numeric/map/uniform options fail cleanly. Each nonblank batch line
gets one ordered report/error, including malformed JSON and empty argument
arrays; bad jobs cannot abort later controls. Empty batches fail explicitly.

Intentional input-domain change: an empty grid now returns an explicit
not-modelled error instead of a nominal 5.6 us launch report. The functional
executor's separate empty-grid behavior is unchanged. No established AGX opcode
semantics, measured timing parameters or accuracy thresholds were changed.

`before-after-repro.json` records four baseline aborts now returning errors,
ignored fourth dimensions and NaN reports now rejected, the empty-grid policy,
and the large-coordinate baseline abort/current success. All fixtures are
synthetic stop-only programs, not hardware observations.

## Retained evidence and validation

- All 34 Rust tests and doc checks pass (`rust-tests.txt`), including occupancy
  reference values, pre-decode rejection, large coordinates, non-finite derived
  time and serial/parallel CLI batch isolation.
- All 363 selected retained functional recordings compare exactly. This is not
  the whole testsuite. `evidence-check.json` verifies every recording against the
  existing `../2026-10-17-decode-pages/replay-manifest.tsv`; no evidence changed.
- All 74 atomic-variant replays and independent evidence verification pass.
- The cost verifier reports zero failures. All 36 complete prediction reports
  match the existing SHA-256 without rebaselining (`prediction-check.json`).
- Playground `verify-simulator-launch` checks 34 invalid jobs and three malformed
  JSON lines with unchanged valid controls, in one/four-worker batches. Checks
  remain enabled under Python `-O`; the gate rejects the old binary under both
  normal and optimized Python (`checker-selftest.json`).
- Playground `verify-occupancy` reports zero failures over six retained evidence
  files, including all 60 points of the 128-lane register sweep and both shared
  memory sweeps. Their unchanged Git blob hashes and pinned revision are in
  `occupancy-evidence.json`. The [occupancy finding](https://github.com/Jianwen-Ding/Apple-M2-Playground/blob/1f43de7679229cc3973f63fd0f5704d45e7c7e40/novel-results/2026-10-11-occupancy/README.md)
  establishes the resource rules; this run does not extend their hardware scope.

## CPU measurements and limits

Five alternating baseline/current pairs, seven measured samples after a warm-up
per invocation, same release compiler/options (fat LTO, one codegen unit,
`-C linker-features=-lld`). Builds and validation jobs finish before timing.
Every complete dispatch report/trace is checked by the existing benchmark tools.
Values are medians of invocation medians; raw samples are retained.

| Checked CPU workload | Baseline ms | Current ms | Ratio |
| --- | ---: | ---: | ---: |
| ALU full-dispatch CLI | 16.678 | 16.179 | 1.031x |
| Compact standalone cost tracing, 1 worker | 11.687 | 12.052 | 0.970x |
| 64 KiB padded cost tracing, 1 worker | 12.594 | 12.204 | 1.032x |
| 64 KiB padded cost tracing, 4 workers | 4.120 | 4.172 | 0.988x |

Memory-tracing control ratios span 0.956-1.070x (`memory-controls.json`). This
pass changes launch handling, not the tracing/execution hot loops. No simulator
speedup is claimed from these small and mixed differences.

The separate **same-binary A/A** experiments use the exact same executable path
and bytes for both labels (`baseline-metadata.json` records SHA-256 hashes).
ALU CLI labels measure 20.654 vs 19.773 ms despite identical code (ratio 1.045x).
Memory-tracing A/A ratios span 0.900-1.078x; four-worker coalesced functional
tracing appears 11.1% slower with no code change. This establishes variability,
not the cause of the previous pass's regressions: those are not erased or
rebaselined, and need controlled profiling/repeated measurements.

`perf` and `valgrind` are not installed, so no instruction-level CPU samples
were collected. M2 hardware is also unavailable (`/dev/dri` absent). Next:
controlled profiling when available, lower-level geometry invariants, resource
SR replay, and real data-dependent workload profiles.

Reproduce from the simulator root (after building release binaries/examples):

```sh
cargo test --offline --release --manifest-path rust/Cargo.toml
rust/target/release/suite_check /path/to/retained-suite
python3 scripts/check_dispatch_reports.py
python3 scripts/bench_cost_trace.py --baseline /path/to/baseline/cost_trace \
  --baseline-dispatch /path/to/baseline/agx_dispatch --output /path/to/benchmark.json
python3 scripts/bench_memory_trace.py --baseline /path/to/baseline/memory_trace
# A/A: point both labels at the same binaries explicitly.
python3 scripts/bench_cost_trace.py --baseline /path/to/same/cost_trace \
  --trace /path/to/same/cost_trace --baseline-dispatch /path/to/same/agx_dispatch \
  --dispatch /path/to/same/agx_dispatch --output /path/to/no-change-alu.json
make -C ../Apple-M2-Playground verify-simulator-launch verify-occupancy
```
