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

Next candidates: remove repeated environment lookups in get_sr; correct partial
workgroup masks across Y/Z; measure functional execution and tracing on retained
workloads before optimizing their hot paths. Keep cost-model calibration and
hardware evidence unchanged unless new measurements justify a change.
