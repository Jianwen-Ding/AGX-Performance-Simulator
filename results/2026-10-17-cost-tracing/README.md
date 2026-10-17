# Cost-model tracing: compact lazy metadata shared across dispatch groups

Offline x86_64 Linux, Rust 1.99.1. No GPU was available; no hardware evidence,
cost parameters or timing expectations were changed.

## Changes and scope

`cost::PreparedTrace` keeps immutable lazy decodes in halfword-indexed once-cells.
Only visited PCs allocate an instruction record. Dispatch groups, waves and
memory-overlap probes share that metadata; `mem_check` also reuses it. Every
trace still starts with separate registers, execution masks, taint and zeroed
approximate interpreter memory. This does not make the performance tracer a
real-memory functional simulator or add support for previously skipped operands.

Legacy `trace`, `trace_group` and `trace_reference` entry points remain available.
They benefit from compact storage but do not share metadata across calls.
Unused retained register-list vectors were removed from cached records.

Every visited instruction now consumes the step budget, including raw `78 00`
and `78 09` barriers, which previously bypassed the check. Truncated instructions,
invalid taken jump targets and out-of-range uniform indices return errors.
The dispatch timing checker now strips whole zero padding halfwords, preserving
the complete final `88 00` stop. The barrier behavior and calibration established
in the playground's `2026-10-11-sync-and-end-opcodes` finding remain unchanged.

## Benchmarks

Baseline: `6db2e9c2f0431d242a40a323c8b3d3e0948c4ae7`. Both builds use the same
release toolchain/options (`-C linker-features=-lld` on this host). Five pairs
alternate baseline/current order. Each invocation discards one warm-up and
measures seven runs; figures below are medians of the five invocation medians.
Raw samples, complete synthetic dispatch report and retained-report comparison
hash are in `benchmark.json`.

| Workload | Baseline ms | New ms | Speedup |
| --- | ---: | ---: | ---: |
| Standalone tracing, 1 worker, compact | 12.848 | 12.157 | 1.057x |
| Standalone tracing, 1 worker, 64 KiB unreachable padding | 48.006 | 14.777 | 3.249x |
| Standalone tracing, 4 workers, 64 KiB unreachable padding | 17.652 | 4.566 | 3.866x |
| Full dispatch CLI, compact ALU program | 24.108 | 15.435 | 1.562x |

Standalone traces cover 128 independent groups running get_sr, 128 iadds and
stop. Every complete PC sequence is checked (16,640 instructions total), as are
empty memory-access, skipped-op and warning collections. Unreachable suffixes
isolate table-allocation overhead; they are not representative instruction mixes.
For the four-worker padded case, median child peak RSS is 107,512 -> 8,576 KiB
(about 105.0 -> 8.4 MiB), including the helper's fork/exec RSS floor.

The full CLI workload dispatches 1,048,576 threads in 64-thread workgroups with
8 declared register halves and `--no-mem`. Timing includes startup, parameter
loading, decoding, tracing and prediction. Every complete JSON report, including
warnings and wave details, matches the baseline exactly (prediction: 48.8 us,
not a hardware measurement). This exercises shared preparation across groups
and waves. These focused workloads are not a whole-corpus speedup claim.

Reproduce by building `cost_trace` and `agx_dispatch` in both checkouts (copy the
new example into the baseline), then run from the repository root:

```sh
python3 scripts/bench_cost_trace.py \
  --baseline /path/to/baseline/cost_trace \
  --baseline-dispatch /path/to/baseline/agx_dispatch \
  --compare-retained --output /path/to/results.json
```

## Validation

- 21 Rust tests and documentation checks pass, including compiled/reference
  trace equivalence and two new cost-tracer regressions.
- Shared/fresh traces are compared under four workers with varied uniforms,
  geometry, partial groups and valid/skipped loads. Full PCs, memory segments,
  taint-derived warnings and skipped sets agree. Independent expected segment
  addresses and warning/skipping conditions are also checked.
- Bounds tests cover both tracer engines, both barrier forms, all-barrier
  programs, truncated instructions, invalid taken jumps, unreachable malformed
  bytes and overflowing uniform indices.
- 146 selected retained functional cases match exactly, including all 23 control
  recordings and all 102 memory recordings. `replay-manifest.tsv` records hashes;
  evidence is unchanged at the baseline commit. This is not the full testsuite.
- All 74 playground atomic-variant replays pass.
- Cost-model verifier: zero failures. Dispatch timing check: 36 retained rows,
  median absolute error 0.048, p90 0.092, max 0.215, unchanged acceptance limit.
- Complete baseline/new JSON reports for all 36 retained dispatch rows are
  exactly equal, not merely equal after display rounding.

The playground adds `make replay-control-suite` for repeatable exact functional
replay. Next candidates: cached cost-trace memory operands and broader cross-lane
replay. Malformed performance launch dimensions also merit a separate guard pass.
