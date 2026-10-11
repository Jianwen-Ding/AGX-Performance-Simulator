# Instruction benchmark evidence — 2026-10-11

`bench.json.gz`: `scripts/run_bench.py` final run (60 variants × latency / simd_issue / gpu). Per mode it
holds firmware times, the in-dispatch clock (MHz), cycles per step, acceptance data (stable groups,
attempts, drift, previous-dispatch agreement) and, for GPU mode, the per-core group histogram and spread.
`table.md` is generated from it by `scripts/bench_report.py`.

Templates use `x` for the chain register and `c0`, `c1`, `c2` for constants (int 1/3/5, float 0.0/1.0,
half 0.0/1.0). Offline: `scripts/verify_bench.py` (`make verify-bench`).
