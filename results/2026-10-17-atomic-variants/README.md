# Device atomic variants (2026-10-17)

Owner-approved probes of device-atomic encodings Mesa never emits, run with `scripts/probe_atomics.py`. Every variant
ran only after the same program with a known surrogate instruction passed `agx_run` (`scripts/probe_variants.py`).

- **Finding and offline check:** apple-documentation-playground `novel-results/2026-10-17-atomic-variants`,
  `make verify-atomic-variants`.
- **Evidence (all 54 records):** apple-documentation-playground `results/2026-10-17-atomic-variants/records.json.gz`.
  `python3 scripts/probe_atomics.py analyse` reads it by default.
- **Raw per-probe files:** `/scratch/apple-m2-playground/atomic-variants` (regenerable, resumable).

Summary:
- Bit 45 makes op 1 (sub) a float32 add (nearest even, denormals flushed, atomic).
- Bit 45 makes op 3 (cmpxchg) a 64-bit unsigned min; bit 46 selects max. With Rt = 1 it returns a changed flag.
- The other ops ignore bits 45-46.
- Ops 11-15 have no memory effect and end the issuing SIMD group.
