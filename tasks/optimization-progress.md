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
