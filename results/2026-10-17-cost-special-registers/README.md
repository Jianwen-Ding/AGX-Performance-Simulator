# Context-aware cost-tracer special registers

CPU-only on x86_64 Linux with Rust 1.99.1 (58cf8f9c3, 2026-10-13); no M2 device
or new GPU timing. Simulator baseline: `99b9179f8ac79a1110f2fbcb95996d5c63c8ce59`;
playground evidence: `05c8db782cdd19eeef94313585d898c8a10fb12b`.

## Changes and evidence

Cache get_sr selector/destination/width in each decoded cost-trace instruction.
Resolve group-uniform values once per executed read; preserve the approximate
interpreter's skip/taint fallback for unknown state. Support 16-bit writes with
the other half preserved. SR56/57 quad rank/count and SR58/59 SIMD rank/count
now follow the current execution mask, rather than physical positions/full-group
counts. SR16 remains physical quad position. Values/masks are never cached across
instructions, calls or contexts. Local coordinate divisions are now evaluated
only for the requested coordinate axis. Context-free tracing/reference behavior and
calibration are unchanged. Invalid public GroupCtx input guards remain future work.

Semantics come from retained playground findings:
`novel-results/2026-10-10-sr-resource-registers`,
`novel-results/2026-10-16-special-registers`, and their pinned evidence. This does
not add 64-bit SR reads: the odd destination encoding still writes one 32-bit
register, leaving its adjacent register intact.

## Reconstructed value replay

Playground `replay-simulator-cost-sr` sends 6,171 synthetic SIMD-group probes
through the actual cost tracer. A 64-bit address encodes every active pair as
64*(32*value+lane), making the captured 64-byte segment injective over the full
u32 value range and all 32 lanes. The memory write itself is intentionally skipped
at the unmapped synthetic address; its captured address, exact PC path, warnings
and skipped-instruction set are checked. No other skipped instruction is accepted.
Unit tests separately check inactive registers and half preservation.

| Evidence | Group probes | Checks and exclusions |
| --- | ---: | --- |
| Six divergent SR scans | 3,036 | Every active lane; 253 selectors, two groups/pattern. Exclude placement SR17/20 and time-varying SR113. SR21 uses recorded context. |
| 16-bit destination readbacks | 1,512 | 252 selectors, both repetitions, three groups including a one-lane edge. Exclude SR17/20/21/113. Compare recorded full word, not only low half. |
| Odd/64-bit destination readbacks | 1,530 | 255 selectors, both repetitions, three groups. Compare adjacent-register preservation only, exactly what this recording observed; exclude SR113. Low read value is not observed/compared here. |
| SR18/19 metadata sweeps | 93 | All 31 sweep points, three groups each; includes preamble-register independence. |

These probes reconstruct masks/launch inputs and rename destination registers;
they do not execute the original GPU shader binaries. The independent resource
verifier also passes its nine corruption tests. The new gate rejects missing
patterns/selectors/lane values and a wrong simulated lane value (four injections),
with normal and optimized Python. The baseline fails a retained divergent-rank
probe; that rejection is retained in `baseline-rejection.json`, not timed as a gain.

## Benchmark method

Build the identical `cost_special_registers` example in baseline and current
checkouts. Release opt-level 3, fat LTO, one codegen unit, panic abort; same offline
dependencies/toolchain and bfd linking (`-C linker-features=-lld`). The restored
baseline has one extra terminal blank line in 22 untouched Rust files; their
contents otherwise match pinned main. Cargo normalized its lockfile to exactly
match remote. Both builds use the same materialized modules; existing local
changes are left untouched.

128 groups, 128 SR reads/group, eight selectors, one warm-up plus seven measured
repetitions; five alternating pairs. Only already-supported full-mask 32-bit
cases are timed against baseline. Every lane/value, adjacent register, PC and
memory access is checked after timing. Timer covers tracing, not fixture assembly
or assertions. Decodes are shared across groups/repetitions. JSON retains every
paired median, not each of the seven internal samples. Same-binary A/A and
functional/memory/ALU controls are recorded separately. No task-owned build or
validation process overlaps timed runs; no workload-average speedup is implied.

```sh
cargo build --offline --release --manifest-path rust/Cargo.toml --examples --bins
python3 scripts/bench_cost_special_registers.py --baseline BASE/rust/target/release/examples/cost_special_registers --current rust/target/release/examples/cost_special_registers --output benchmark.json
python3 scripts/bench_memory_trace.py --baseline BASE/rust/target/release/examples/memory_trace --output memory-controls.json
python3 scripts/bench_special_registers.py --baseline BASE/rust/target/release/examples/special_registers --output functional-controls.json
AGX_SIM_REPO="$PWD" python3 scripts/bench_cost_trace.py --baseline BASE/rust/target/release/examples/cost_trace --baseline-dispatch BASE/rust/target/release/agx_dispatch --output alu-controls.json
```

## Validation reproduction

```sh
cargo test --offline --release --manifest-path rust/Cargo.toml
rust/target/release/suite_check /path/to/retained-suite
AGX_THREADS=2 rust/target/release/verify_cost_model
AGX_THREADS=2 python3 scripts/check_dispatch_reports.py
make -C ../Apple-M2-Playground verify-sr-resources replay-simulator-cost-sr verify-simulator-launch verify-simulator-geometry SIMREPO="$PWD"
python3 -O ../Apple-M2-Playground/scripts/replay_cost_sr.py --simrepo "$PWD" --selftest
AGX_RUN="$PWD/rust/target/release/agx_run" python3 scripts/sr_check.py
python3 ../Apple-M2-Playground/scripts/replay_atomic_variants.py --run "$PWD/rust/target/release/agx_run"
```

Pinned source/evidence/executable hashes and full validation logs are adjacent.
All 363 suite blobs match the manifest from `2026-10-17-decode-pages`.

## Intermediate candidates

`initial-*` retains the first implementation: cost SR cases other than global-X
improved 1.505-1.843x, but global-X regressed 31.8% (3.399 -> 4.479 ms).
Cost-memory controls regressed 4.1-6.3%; some functional SR controls regressed
10-13%. Initial same-binary SR ratios were 1.001-1.036x, so those regressions
cannot simply be dismissed as equivalent A/A noise.

`axis-benchmark.json` evaluates only the requested coordinate axis: cost-SR ratios
1.548-1.931x except global-X, still 18.6% slower. `coordinate-benchmark.json`
additionally resolves the axis/global origin outside the lane loop; ratios
1.548-2.024x, including global-X 1.674x. These checked exploratory medians remain
separate from the final build's measurements and do not rebaseline prior results.

## Selected build results

All eight checked full-mask cost-SR cases improve 1.572-1.962x over five paired
medians. Uniform SR7/18/19 improve 1.915-1.962x; global-X improves 1.655x.

| Checked CPU workload | Baseline ms | Selected ms | Change |
| --- | ---: | ---: | --- |
| Cost SR7 | 3.356003 | 1.710224 | 1.962x faster |
| Cost global-X SR80 | 3.326239 | 2.009366 | 1.655x faster |
| Cost SIMD rank SR58 | 3.354982 | 2.134781 | 1.572x faster |
| Four-worker coalesced cost-memory control | 3.874 | 4.149 | 7.1% slower |
| Four-worker scattered functional-memory control | 4.606 | 4.963 | 7.8% slower |
| Full ALU CLI control | 15.974018 | 16.229566 | 1.6% slower |

Functional SR57 with 13 active lanes is 7.9% slower (1.288 -> 1.390 ms), and
compact one-worker ALU cost tracing is 4.5% slower. Other raw controls are retained
in the adjacent JSON. Same-binary cost-SR A/A ratios span 0.949-1.041x; this does
not establish the cause of other control regressions or prove them resolved.
No overall speedup claim. Prior runs' open memory/ALU regressions remain open.

Final validation: 44 Rust tests/doc checks; 363 exact retained cases with pinned
hashes; 6,171 cost-SR group probes; 1,518 original-program functional SR comparisons;
74 atomic replays plus independent atomic/resource evidence verification; four
new failure injections in normal/optimized Python; launch/geometry gates; zero
cost-verifier failures. All 36 complete prediction reports retain SHA-256
`705e21e3659b343d6afac5cc37c75ad21aec3e8622f99102197f34fcf56a8f2e`.
No retained output, calibration or hardware timing reference changed.

Next candidates: compact/boxed optional SR metadata and controlled memory-layout
profiling (measure control regressions, do not assume their cause), public
GroupCtx input guards, and original resource-shader functional replay. Hardware-
only timing and SR113 remain outside offline validation.
