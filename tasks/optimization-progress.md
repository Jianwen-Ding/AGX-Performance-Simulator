# Simulator improvement log

## 2026-10-17: measured atomic variants and bulk dumps

Implemented the G14C device atomic variants confirmed in Apple-M2-Playground's
`novel-results/2026-10-17-atomic-variants`:

- op 1 + bit 45: float32 add, nearest-even, signed input/output flush, canonical NaN;
- op 3 + bit 45: unsigned 64-bit min/max (bit 46 selects max), word-scaled index,
  with a changed flag when returning;
- device atomic ops 11..15 at the recorded u5=2 terminate the issuing SIMD group;
- the recorded threadgroup sub + bit 6 suppresses memory access and returns zero.

Ordinary integer atomics retain their behavior. The unused CAS compare register
is no longer read by unrelated operations. New functional semantics do not imply
calibrated float/u64 atomic timing; that still needs hardware measurements.
Equal-value u64 changed flags follow the inferred `new != old` rule; equality was
not present in the retained hardware sample.

`agx_run --dump` now copies contiguous mapped ranges in bulk, preserving byte-wise
behavior for adjacent and overlapping regions. Scalar memory accesses reject
wrapping address ranges and widths beyond 8 bytes. Copy-on-write is preserved.

Validation on x86_64 Linux, Rust 1.99.1 (no M2 hardware available):

- Rust library tests and both `kernels_match` integration tests pass.
- `verify_cost_model`: zero failures, including held-out kernels and core scaling.
- Playground `verify_atomic_variants.py`: all independent evidence/hash checks pass.
- Playground `replay_atomic_variants.py`: 74 records, zero failures; deterministic
  cases compare every output and table word, contention uses an independent
  rational float model to validate an allowed serialization.
- Focused 16 MiB dump benchmark, 8 repeats: scalar 1341.176 ms, bulk 51.368 ms,
  26.11x. This measures copying, not overall simulation; bulk timing also includes
  equality verification. Reproduce with `cargo run --release --example memory_dump`.

## 2026-10-17: special-register execution and partial workgroups

Known special-register reads no longer query `AGX_SR_DEFAULT` once per lane.
The fallback is consulted lazily for an unmodelled selector, at most once per
instruction execution. Active-rank/count behavior under divergence is unchanged.

Partial workgroups now mask nonexistent threads in X, Y and Z, using wide global
coordinate arithmetic. Previously only X was checked, so a partial Y/Z edge
could write outside the requested grid. Added output-buffer regressions over
three geometries, including multi-SIMD workgroups. Invalid zero/oversized or
overflowing workgroups and out-of-range uniform indices now return errors in
functional execution and functional tracing rather than panicking.

Validation: all 12 Rust library tests plus both integration tests pass; all 74
retained atomic programs replay with zero failures. The cost-model code and its
calibration are unchanged.

`python3 scripts/bench_func_sr.py --baseline /path/to/previous/agx_run` runs a
checked special-register workload (1024 groups, 128 reads/group, five measured
runs after warm-up). Before: median 370.861 ms; after: 28.226 ms; 13.14x on this
x86_64 host. Timing includes process startup, execution and output dumping;
every output word and the dynamic instruction count are checked. This is a
focused workload, not an overall workload-average speed claim.

## 2026-10-17: shared lazy decode cache and bounded execution

Added `PreparedDispatch`, a reusable thread-safe functional tracer. Instruction
metadata is decoded lazily into shared once-cells; workgroup execution and memory
snapshots remain independent. `agx_dispatch` reuses this prepared state across
its callbacks. Replaced full per-byte instruction records with compact pointers.

The instruction budget is now cumulative per SIMD group across barrier resumes.
Truncated instructions and invalid taken jumps return errors. Fixed
`agx_dispatch` padding removal so it preserves the complete final stop.

Validation: 18 Rust tests, all 74 atomic replays, and 30 representative retained
hardware cases pass; twelve functional prediction JSON reports match the prior
build exactly. The checked tracing benchmark is 3.06x faster on compact code,
8.36x faster on a padded four-worker case, and uses 8,448 rather than 94,780 KiB
peak RSS in that case. Direct SR execution remains within normal run variation.
See `results/2026-10-17-functional-cache/README.md` for scope and reproduction.

Next candidates: extend retained-hardware regression coverage, profile real
data-dependent dispatches, and reduce repeated memory operand decoding where
profiles justify it. Float/u64 atomic timing still needs hardware measurements.
Preserve calibration and evidence unless measurements justify changes.

## 2026-10-17: cached memory operands and reliable replay status

Device and threadgroup loads/stores cache immutable operand fields per visited
PC. Dynamic registers, addresses, masks and scoreboard state remain per execution.
Memory-only boxed payloads keep common decode records compact.

The checked device-load benchmark (2048 groups, 256 loads/group) measures
323.578 -> 297.460 ms, 1.088x, over five alternating pairs of nine measured runs.
Every run checks all 65,536 output words and 532,480 instructions. The compact
one-worker ALU trace control was 4.8% slower; the padded four-worker control was
10.7% faster. These focused measurements are not a whole-corpus speed claim.
Raw results and reproduction: `results/2026-10-17-memory-operands/README.md`.

Strengthened `suite_check`: failures and empty selections now return nonzero;
shortened deterministic expected buffers fail before zip-based comparisons.
A synthetic CLI regression covers success and these failure paths. Playground
`make replay-memory-suite` exposes the three retained memory categories.

Validation: 19 Rust tests and documentation checks pass; all 126 selected
retained cases match exactly (including all 102 memory recordings); all 74
atomic replays pass; cost-model verifier has zero failures. The manifest records
all replayed file hashes. No new GPU measurements or timing rebaseline.

Next: profile memory-heavy data-dependent loops, consider cached atomic operands
and shared cost-trace memory decoding, and extend cross-lane/control-flow replay.
Check the ALU controls when changing decode layout. M2 hardware remains unavailable;
float/u64 atomic timing still requires real hardware evidence.

## 2026-10-17: shared preparation for performance-model traces

Added `cost::PreparedTrace`: halfword-indexed lazy decode records shared across
dispatch groups, waves and overlap probes, also reused by `mem_check`. Registers,
masks, taint and the approximate interpreter's zero memory remain per call.
No formerly skipped operand or new hardware timing is claimed as supported.

Raw 0x78 barriers now consume the trace instruction budget. Truncated fetches,
invalid taken jumps and invalid uniform indices fail cleanly. Corrected the
dispatch timing checker's byte-wise padding trim to preserve the final stop.

Checked five-pair benchmark: full compact ALU dispatch CLI 24.108 -> 15.435 ms
(1.562x); standalone four-worker padded tracing 17.652 -> 4.566 ms (3.866x),
observed peak RSS 107,512 -> 8,576 KiB. Compact standalone tracing improved
12.848 -> 12.157 ms. These are focused workloads, not an overall speed claim.
Raw paired results and reproduction: `results/2026-10-17-cost-tracing/README.md`.

Validation: 21 Rust tests and doc checks pass; 146 exact retained functional
replays (all 23 control cases, all 102 memory cases); 74 atomic replays; zero
cost-model failures. All 36 complete dispatch prediction reports equal the
baseline exactly; retained dispatch timing check median absolute error 0.048.
Playground `make replay-control-suite` passes all 23 recordings. Calibration
and evidence remain unchanged; there is no new M2 hardware validation.

Next: cached cost-trace memory operands, broader cross-lane replay and guards for
malformed performance launch dimensions. Keep approximate cost-emulation paths
distinct from real-memory functional traces. Hardware-only timing work remains
blocked on an M2 device.

## 2026-10-17: cached memory capture and 130 cross-lane replays

Both tracers cache memory-address operands once per visited PC. Device segment
vectors reserve once before sorting/deduplication; atomic/TG lane-component
order and duplicates are preserved. Tests pin the unchanged legacy helpers'
missing-field, inactive-lane, format and signed-offset behavior. A scattered-
access regression in an adjacent-dedup trial was measured and that variant was
rejected; raw trial data is retained separately.

Five alternating checked benchmark pairs: cost memory tracing 1.120-1.421x and
functional tracing 1.056-1.165x faster over coalesced/scattered, 1/4-worker cases.
The ALU-only CLI control was 2.5% slower; no general speedup is claimed. Details,
raw samples and reproduction: `results/2026-10-17-memory-tracing/README.md`.

Added strict `suite_check --case-prefix` selection and playground
`make replay-cross-lane-suite`. All 276 selected recordings compare exactly,
including 130 full/masked reductions, scans, shuffles and quad ballots. All 74
atomic replays pass, 23 Rust tests and doc checks pass, and the cost verifier has
zero failures. All 36 complete prediction reports match the committed baseline
hash; a new read-only reference checker makes that regression repeatable.

No hardware evidence, calibration or instruction semantics were changed. Next:
malformed performance launch guards, cached atomic execution operands and real
data-dependent memory profiling. Hardware-only timing remains blocked on M2 access.

## 2026-10-17: cached atomic execution and complete integer replay

Cache immutable device/TG atomic fields per visited PC. Fix TG non-CAS operations
reading an unused compare register; synthetic tests cover all ten operations
with/without returns at the register boundary, while CAS still requires and
uses its comparison source. Retained float/u64, suppression and group-termination
behavior is unchanged. No new hardware semantics or calibration are claimed.

Five alternating checked benchmark pairs: one-active-lane device 5.507 ->
3.229 ms (1.705x), TG 4.012 -> 2.183 ms (1.838x); 32-lane TG 14.444 ->
13.050 ms (1.107x), device effectively flat. ALU-only full dispatch is flat,
but padded single-worker standalone cost tracing is 6.7-13.9% slower over two
complete control runs. This regression is unresolved; do not claim a general
speedup. Raw samples, methodology and boundary repro:
`results/2026-10-17-atomic-execution/README.md`.

Validation: 25 Rust tests/doc checks; 358 exact retained recordings, including
all 88 standard integer atomic cases (82 more than the previous selection);
74 variant replays and independent evidence verification; zero cost-verifier
failures; all 36 complete prediction reports match the unrebaselined reference.
Playground's new `replay-integer-atomic-suite` target passes all 88 cases.

Next: profile the padded standalone control regression, malformed performance
launch guards, and real data-dependent memory/atomic workloads. No M2 hardware
is available, so hardware-only timing remains blocked.

## 2026-10-17: bounded decode tables and eight geometry recordings

Replace code-length-proportional decode tables with an 8 KiB dense prefix plus
lazy 512-byte tail pages. Keep byte indexing in functional execution and halfword
indexing in cost tracing, existing alignment/error checks, and per-group state.
Small functional programs retain a specialized original dense lookup loop;
cache mode is selected once per group/resume. Add boundary/crossing/jump/error
and concurrent initialization regressions. Two rejected layouts and raw paired
controls are retained, not silently discarded.

Five alternating checked pairs: at 1 MiB unreachable padding, standalone cost
tracing 3.699-5.134x and functional tracing 1.171-1.241x faster. Four-worker peak
RSS is 36.6 -> 8.4 MiB (cost) and 20.0 -> 8.4 MiB (functional). This is not a
whole-predictor speed claim: cost's standalone benchmark prepares each group.
ALU full dispatch is 3.7% slower; four-worker coalesced cost memory tracing is
17.2% slower; long active functional code is 3.6-6.6% slower. These regressions
remain unresolved. All raw samples and methodology:
`results/2026-10-17-decode-pages/README.md`.

Validation: 29 Rust tests/doc checks; all 363 selected retained cases exact,
including five additional geometry cases (all eight now replay); 74 atomic
variant replays; independent atomic evidence verification; zero cost-verifier
failures; all 36 complete prediction reports match their existing hash.
Playground's new `replay-geometry-suite` gate passes all eight recordings.
No hardware observation, semantics, calibration or expected output changed.

Next: no-change A/A controls and CPU profiling of compact/active-tail regressions,
performance launch guards, resource replay and real data-dependent profiles.
New M2 hardware validation remains unavailable.

## 2026-10-17: launch input domain and batch fault isolation

Add `Launch::validate` before decode/tracing; reject zero/overflowing dimensions,
unrepresentable workgroup counts, invalid topology/resources/uniforms and bad
timing inputs. Fix occupancy overflow, impossible one-workgroup fallback and
u32 truncation/XY overflow in large-grid coordinates. Reject non-finite derived
time. Empty grids now explicitly report not modelled (functional no-op unchanged).
Keep all valid occupancy/launch rules and custom-core fallback unchanged.

CLI parsing is fallible; malformed JSON/argument arrays produce ordered batch
errors instead of aborting later jobs. Validate before materializing buffers or
reading shaders/preambles. Playground's new `verify-simulator-launch` gate checks
34 invalid jobs, three bad JSON lines and unchanged serial/parallel controls;
failure injection rejects the old binary even with optimized Python.

Validation: 34 Rust tests/doc checks; 363 exact retained replays; 74 atomic
replays and independent evidence verification; zero cost-verifier and retained
occupancy-verifier failures; all 36 complete predictions match the existing hash.
No hardware evidence, opcode semantics, calibration or expected output changed.

Five-pair ALU dispatch control: 16.678 -> 16.179 ms, but no speedup claim.
Same-binary A/A dispatch labels differ 4.5% in ratio, and memory A/A ratios span
0.900-1.078x. Prior regressions remain unresolved; this variability does not
establish their cause. Raw samples, repros and scope:
`results/2026-10-17-launch-validation/README.md`.

Next: controlled CPU profiling (perf/valgrind unavailable here), lower-level
geometry invariants, resource SR replay and data-dependent workloads. New M2
hardware validation remains unavailable.

## 2026-10-17: cached functional special-register execution

Cache immutable get_sr selector/destination/width/class at decode. Resolve group-
uniform values once per execution, keep lane coordinates per lane, and compute
quad/SIMD ranks/counts from the current mask only. Preserve half-register writes,
inactive behavior, placement flags and lazy unknown-selector fallback. SR113 is
still unmodelled; values are not cached across launches/groups/resumes. Remove
the now-unused duplicate operand vector from functional decode records.

Five alternating checked pairs: uniform SR7/18/19 reads 1.953-3.356x faster;
quad rank 1.393-1.798x; global-X effectively flat (full mask 1.5% slower).
ALU full-dispatch control is 2.7% slower and four-worker scattered cost tracing
17.9% slower; regressions remain open, with no general speedup claim. Raw data,
scope and reproduction: `results/2026-10-17-special-register-cache/README.md`.

Validation: 36 Rust tests/doc checks; three SR schema/failure tests in normal
and optimized Python; 363 exact retained replays; all six SR programs / 1,518
selector-pattern comparisons pass on baseline and current; 74 atomic replays
and independent verification; launch/batch gate; zero cost-verifier failures;
all 36 complete predictions match their existing hash. SR replay now fails on
missing patterns/selectors or incomplete lane values. Playground exposes it as
`replay-simulator-sr`. No retained evidence, semantics or calibration changed.

Next: lower-level geometry guards, resource-width replay and controlled CPU
profiling of real data-dependent paths/control regressions. M2 hardware remains
unavailable; hardware-only timing and unmodelled SR113 remain outside scope.

## 2026-10-17: functional geometry preparation and checked broadcasts

Cache dispatch dimensions/SIMD counts; compute linear workgroup indices and
writer bases once per workgroup, removing per-group/resume vector allocation.
Keep the large interpreter loop separate from setup; uniform SR/SIMD-count
broadcasts bypass per-lane selector classification with checked lane-ordered
writes. Preserve masks, placement, barriers, half writes and unknown-SR fallback.

Reject out-of-dispatch trace coordinates before workers/memory copies and reject
more than 2^48 workgroups rather than aliasing the existing 48+16-bit writer tag.
This is an executor representation limit, not hardware geometry evidence.
Empty grids remain no-ops on every axis, without traversing other dimensions.
Playground adds `verify-simulator-geometry`, including optimized-Python checks
and old-binary failure injection.

Final five-pair medians: setup 1.215-2.036x; uniform SR7/18/19 1.482-2.182x.
One-worker scattered functional tracing is 4.4% slower; full ALU CLI 1.1% slower.
Earlier candidate/control regressions and noisy A/A samples are retained; no
overall speedup or resolution of prior regressions is claimed. Raw paired
medians, rejected/intermediate candidates, hashes and reproduction:
`results/2026-10-17-functional-geometry/README.md`.

Validation: 41 Rust tests/doc checks; 363 exact retained cases and pinned hashes;
1,518 SR comparisons; 74 atomic replays and independent evidence verification;
normal/optimized schema and geometry gates; launch/batch isolation; zero cost
failures; all 36 complete prediction reports keep their existing hash. No new
hardware observation, instruction value, calibration or timing rebaseline.

Next: cost::GroupCtx input guards and lazy coordinate calculations, resource-width
replay, controlled profiling of partial-mask/data-dependent controls. M2 hardware
and CPU profilers remain unavailable; SR113/hardware-only timing stay unvalidated.

## 2026-10-17: cost-tracer special registers and retained width/resource probes

Cache cost get_sr operands; resolve uniform values once per instruction and
coordinate axes/global origins outside the lane loop. Support half-width writes
with the other half preserved; SR56..59 now use actual execution-mask ranks/
counts, not physical position/full-group size. SR16 stays physical. Unknown
state retains skip/taint fallback; context-free/reference behavior is unchanged.
Do not treat odd destinations as 64-bit writes.

Playground adds `replay-simulator-cost-sr`: 6,171 reconstructed group probes
(3,036 divergent SR, 3,042 recorded width, 93 resource sweep), with injective
lane/value address encoding. These are not original shader executions. Odd-width
recordings check only the observed adjacent register. Placement/time-varying
exclusions are explicit. Four failure injections pass with normal/optimized
Python; baseline correctly fails divergent ranks. Independent resource verifier
passes its nine corruption checks. Retained data and manifests are unchanged.

Final five-pair cost-SR medians improve 1.572-1.962x, including global-X 1.655x.
Four-worker cost coalesced memory is 7.1% slower, functional scattered memory
7.8% slower, partial-mask functional SR57 7.9% slower, ALU CLI 1.6% slower.
No overall claim or resolved-prior-regression claim. All initial/axis/selected
samples, A/A controls, hashes, rejection evidence and repros are retained in
`results/2026-10-17-cost-special-registers/README.md`.

Validation: 44 Rust tests/doc checks, 363 exact retained cases/pinned hashes,
1,518 functional SR comparisons, 74 atomic replays plus independent evidence
checks, launch/geometry gates, zero cost-verifier failures; all 36 complete
prediction reports keep their existing hash. No new M2 validation or calibration.

Next: compact/boxed optional SR metadata and controlled profiling of memory/
partial-mask regressions; public GroupCtx guards; original resource-shader
functional replay. M2/profilers unavailable; SR113 and new timing stay unvalidated.
