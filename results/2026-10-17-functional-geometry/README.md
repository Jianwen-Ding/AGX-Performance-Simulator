# Functional geometry and interpreter setup

CPU-only work on x86_64 Linux, Rust 1.99.1. No M2 hardware or new GPU timing.
Baseline: simulator `5019a4630fd11da0b802e46dee94ae4f95cd9728`;
retained playground evidence: `4d35ca1aec3044753406277a4d33055823b822b2`.
Both use release opt-level 3, fat LTO, one codegen unit, panic abort and the
same offline dependencies/toolchain; bfd linking (`-C linker-features=-lld`).

## Scope

Prepare immutable grid dimensions/group counts once. Compute the workgroup's
linear index and race-log writer base once, instead of allocating a three-element
vector per SIMD group or barrier resume. Validate requested trace coordinates
before copying memory/starting workers. Empty grids return immediately on every
axis. Reject more than 2^48 workgroups: the existing writer tag uses 48 index
bits plus 16 group bits. This is a simulator representation limit, not a hardware
limit. Placement rules, edge masks, barriers and writer-tag encoding are unchanged.

The geometry semantics remain those established in playground
`novel-results/2026-10-10-sr-geometry-core-id`; no fresh placement observations
or hardware constraints are inferred here. The synthetic boundary tests are
separate from the eight retained geometry recordings.

## Method

`functional_geometry` checks instruction/group counts and the final writer tag
for 2,048 workgroups, 1/32/256 threads, 0/8 barriers and placement seeds 0/17.
It times `func::run` only, excluding fixture construction and assertions.
One warm-up and seven measured repetitions produce each printed median.
Five alternating baseline/current pairs are retained without dropping pairs.
JSON stores every paired median (not the individual seven internal samples).
Do not interpret these stop/barrier-heavy fixtures as a workload-average speedup.
Memory and special-register controls check real outputs/accesses/instruction counts.
No builds, replay checks or other task-owned CPU-heavy work overlap timing runs.

Reproduce after building the examples in both checkouts (copy the new checked
`rust/examples/functional_geometry.rs` unchanged to the baseline):

```sh
python3 scripts/bench_functional_geometry.py --baseline BASE/rust/target/release/examples/functional_geometry --current rust/target/release/examples/functional_geometry --output geometry.json
python3 scripts/bench_special_registers.py --baseline BASE/rust/target/release/examples/special_registers --output sr.json
python3 scripts/bench_memory_trace.py --baseline BASE/rust/target/release/examples/memory_trace --output memory.json
```

`initial-*` retains the first geometry-only candidate and same-binary A/A checks.
It improved most setup medians 1.06-1.23x, but one case was 0.4% slower.
SR slowdowns persisted in a second paired run: ratios 0.723-1.071x. Same-binary
SR ratios ranged 0.819-1.144x, so host noise is substantial but does not establish
the cause of the slowdowns. `outlined-*` additionally prevents inlining the large
interpreter loop into workgroup setup: setup ratios 1.278-1.953x, while SR ratios
0.657-1.112x remain unacceptable as a general performance claim. These intermediate
results are retained rather than replaced by later samples.

The selected version also gives uniform SR/SIMD-count broadcasts a checked
lane-ordered loop without per-lane selector-class dispatch. It retains the
independent 255-selector, two-width, eight-mask oracle test and unknown-SR
fallback/error-order checks. No instruction value, calibration or reference
timing is changed.

## Selected build measurements

Final five-pair medians: setup cases 1.215-2.036x faster; uniform SR7/18/19
reads 1.482-2.182x faster; SIMD-count SR59 1.615-1.631x faster. The earlier
`broadcast-*` measurement of the same semantic implementation is also retained:
setup 1.167-1.844x; uniform SR7/18/19 1.298-1.685x. Do not replace either sample
set with the other. Final source/executable hashes are in `environment.json`.

| Checked CPU workload | Baseline ms | Selected ms | Observation |
| --- | ---: | ---: | --- |
| 2,048 workgroups, 256 threads, stop, placement 17 | 5.774312 | 2.835677 | 2.036x faster |
| SR19, 256 groups, 128 reads, full mask | 2.884 | 1.518 | 1.900x faster |
| Functional scattered loads, one worker | 17.362 | 18.125 | 4.4% slower |
| Full ALU CLI dispatch control | 24.115383 | 24.383841 | 1.1% slower |

Final memory ratios span 0.958-1.122x; compact/padded cost trace controls include
a 2.2% slower one-worker padded case. The earlier `broadcast-*` run has a 12.2%
slower partial-mask quad-rank SR56 case, though the final run does not. Keep it
open rather than treating one later sample as proof of resolution. Prior runs'
ALU/scattered-cost regressions likewise remain unresolved. Same-binary A/A
ratios are 0.926-1.127x (setup) and 0.834-1.152x (SR), demonstrating significant
variability, not establishing its cause. There is no overall speedup claim.

All intermediate/final paired medians and controls are in the adjacent JSON
files. `cost-dispatch-controls.json` checks equality of complete CLI reports;
`prediction-check.json` separately preserves the unrebaselined 36-report hash.

Validation: 41 Rust tests and doc checks; 363 exact recordings with all pinned
blob hashes verified; 1,518 SR selector/pattern comparisons; 74 atomic replays
and independent evidence verification; normal/optimized SR schema tests;
launch/batch gate; normal/optimized geometry gate; zero cost-verifier failures;
all 36 complete prediction reports retain SHA-256
`705e21e3659b343d6afac5cc37c75ad21aec3e8622f99102197f34fcf56a8f2e`.
Full validation/build logs and `validation.json` are retained here.

## Validation reproduction

```sh
cargo test --offline --release --manifest-path rust/Cargo.toml
cargo build --offline --release --manifest-path rust/Cargo.toml --examples --bins
rust/target/release/suite_check /path/to/retained-suite
AGX_THREADS=2 rust/target/release/verify_cost_model
AGX_THREADS=2 python3 scripts/check_dispatch_reports.py
python3 scripts/test_sr_check.py
python3 -O scripts/test_sr_check.py
make -C ../Apple-M2-Playground verify-simulator-geometry verify-simulator-launch replay-simulator-sr SIMREPO="$PWD"
python3 ../Apple-M2-Playground/scripts/replay_atomic_variants.py --run "$PWD/rust/target/release/agx_run"
python3 ../Apple-M2-Playground/scripts/verify_atomic_variants.py
```

The suite manifest and SR blob hashes are pinned in the environment metadata.
The new playground geometry gate also passes with `python3 -O`; it uses explicit
exceptions, not removable assertions. `gate-failure-injection.json` preserves
normal/optimized rejection of the old pre-SR-cache binary (capacity-case timeout,
not a successful validation or a speed measurement). M2 is not exposed here
(`x86_64`, no `/dev/dri`); hardware-only timing and SR113 remain unvalidated.
