# Bounded decode tables and complete standard geometry replay

Offline x86_64 Linux, Rust 1.99.1. Baseline:
`d1faa5b8b572175a87fd51fcf8f2550d73e2f140`. No GPU execution or calibration changes.

## Implementation and accuracy boundary

The tracers' decode-table allocation no longer scales with every unreachable
byte/halfword of padded code. A shared container keeps an 8 KiB dense prefix
and a small directory of lazy 512-byte tail pages. Metadata remains immutable;
register values, masks, addresses, memory and execution order are unchanged.
Only visited tail pages allocate instruction cells; only visited PCs allocate
decoded instructions. A failed decode leaves its cell uninitialized, not poisoned.

Cost tracing remains halfword-indexed, with its original alignment guards.
Functional tracing remains byte-indexed, so malformed odd PCs cannot alias valid
cells. Functional programs up to 8 KiB retain their original dense lookup loop;
larger programs use the paged specialization. That choice happens once per
group/barrier resume, not per instruction in the dense loop. This is a storage
and execution-speed change, not new AGX instruction support.

Synthetic regressions cover dense/page boundaries, instructions straddling
boundaries, jumps over malformed unreachable bytes, truncated tail instructions,
failed-call reuse, concurrent page/cell initialization, and distinct odd/even
functional cache cells. Integration checks pin complete PC sequences, memory
segment sets and every output word. Existing state-isolation tests remain intact.

Two rejected layouts are retained in `small-prefix-trial/` and
`always-paged-trial/`, with raw paired data and explanations. Their measurements
must not be confused with the selected implementation in this directory.

## Benchmark method

Five alternating baseline/current pairs, seven measured executions after a
warm-up per invocation; reported values are medians of invocation medians.
Both binaries use the same release compiler/options (opt-level 3, fat LTO,
one codegen unit, `-C linker-features=-lld`). Builds and validation jobs finish
before timed comparisons. The existing checked examples are unchanged.

`benchmark.json` compares 128 groups/workgroups, 128 adds and 0/64/1024 KiB of
unreachable padding, with one/four workers. Every complete PC sequence is
checked: 16,640 cost-trace instructions or 33,280 functional-trace instructions
(two SIMD groups/workgroup). Cost standalone tracing intentionally prepares
each group separately; functional tracing prepares once for all workgroups.
Consequently the large standalone cost-tracing gain is not a claim about the
whole-dispatch predictor, which already shares preparation.

`active-tail-controls.json` uses 2048 adds (over 16 KiB of executed code), not
just unreachable padding. Additional checked controls cover full ALU dispatch
(`alu-controls.json`), coalesced/scattered memory tracing (`memory-controls.json`)
and returning device/TG atomics (`atomic-controls.json`). All raw samples are
retained, including regressions. Peak child RSS is measured in a fresh helper;
the inherited helper high-water mark limits resolution for small cases.

## Selected implementation: measured CPU results

| Tracer | Workers | Padding KiB | Baseline ms | New ms | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| cost | 1 | 0 | 11.225 | 11.101 | 1.011x |
| cost | 1 | 64 | 13.548 | 11.474 | 1.181x |
| cost | 1 | 1024 | 44.186 | 11.946 | 3.699x |
| cost | 4 | 0 | 3.226 | 3.358 | 0.961x |
| cost | 4 | 64 | 3.824 | 3.646 | 1.049x |
| cost | 4 | 1024 | 18.055 | 3.517 | 5.134x |
| functional | 1 | 0 | 4.774 | 4.628 | 1.032x |
| functional | 1 | 64 | 4.905 | 4.943 | 0.992x |
| functional | 1 | 1024 | 5.753 | 4.913 | 1.171x |
| functional | 4 | 0 | 1.478 | 1.556 | 0.950x |
| functional | 4 | 64 | 1.665 | 1.552 | 1.073x |
| functional | 4 | 1024 | 2.214 | 1.784 | 1.241x |

For 1 MiB padding/four workers, median observed peak RSS falls from 37,520 to
8,576 KiB for cost tracing (36.6 -> 8.4 MiB) and 20,496 to 8,576 KiB for
functional tracing (20.0 -> 8.4 MiB). Small-case RSS is helper-floor limited.

The 2048-add active-tail control is not uniformly faster: cost tracing is
179.062 -> 180.740 ms with one worker (0.9% slower), 53.331 -> 49.123 ms
with four (1.086x). Functional tracing is 75.612 -> 78.319 ms (3.6% slower)
and 23.384 -> 24.930 ms (6.6% slower), respectively. Tail lookup overhead and
code-layout/host effects are not separately attributed.

The ALU-only full-dispatch CLI control is 15.611 -> 16.195 ms (3.7% slower).
Memory controls are roughly flat with one worker; four-worker coalesced cost
tracing regresses 3.729 -> 4.370 ms (17.2%), and functional memory tracing is
5.9-6.3% slower for coalesced/scattered accesses. Device atomic execution with
one active lane is 2.864 -> 3.120 ms (8.9% slower), but 32-lane device execution
is 39.778 -> 37.366 ms (1.065x). TG atomic execution improves 1.117x/1.047x
with one/32 active lanes. These controls are not hidden by the padded results.

This is a focused allocation/memory-footprint improvement, not an overall
speedup claim. Compact/control regressions remain unresolved. Next: profile
them with no-change A/A controls and instruction-level CPU profiling where
available; consider avoiding page lookup overhead for long active bodies.
Malformed performance launch guards and real data-dependent profiles remain
separate candidates. No timing expectation or accuracy threshold was changed.

## Validation and evidence

- All 29 Rust tests and doc checks pass (`rust-tests.txt`).
- All 363 selected retained recordings compare exactly (`suite-check.txt`): the
  prior 358-case selection plus five more standard geometry cases. This is not
  the whole testsuite. `replay-manifest.tsv` hashes compressed files and was
  checked against the baseline Git tree; no retained recording changed.
- All eight standard 1-D/2-D/3-D geometry cases pass the playground's new
  `replay-geometry-suite`, including non-warp-sized workgroups. This supplements
  the [retained geometry/SR findings](https://github.com/Jianwen-Ding/Apple-M2-Playground/blob/c986152a7bf6fab588a22302c3c89d4a3ac0bc8d/novel-results/2026-10-10-sr-geometry-core-id/README.md),
  not a new SR20 placement or hardware observation.
- All 74 atomic-variant replays and the independent atomic evidence verifier
  pass. The cost-model verifier has zero failures (logs alongside).
- All 36 complete prediction reports match the existing SHA-256 without
  rebaselining (`prediction-check.json`).

Reproduce after building examples/binaries in both checkouts:

```sh
python3 scripts/bench_decode_cache.py --baseline-dir /path/to/baseline/examples \
  --output /path/to/benchmark.json
python3 scripts/bench_decode_cache.py --baseline-dir /path/to/baseline/examples \
  --steps 2048 --padding-kib 0 --output /path/to/active-tail-controls.json
python3 scripts/bench_cost_trace.py --baseline /path/to/baseline/cost_trace \
  --baseline-dispatch /path/to/baseline/agx_dispatch --output /path/to/alu-controls.json
python3 scripts/bench_memory_trace.py --baseline /path/to/baseline/memory_trace
python3 scripts/bench_atomic_execution.py --baseline /path/to/baseline/atomic_execution
cargo test --offline --release --manifest-path rust/Cargo.toml
rust/target/release/suite_check /path/to/retained-suite
python3 scripts/check_dispatch_reports.py
```

M2 hardware remains unavailable in this worker (`/dev/dri` absent). No hardware
validation or cross-run attribution of the previous tracing regression is claimed.
