# Fetch and memory-wait overlap with 1-8 SIMD groups (2026-11-22, hardware)

`scripts/probe_fetch_overlap_groups.py` runs one workgroup of G groups on one core, warm (1000 back to back). The code
has K = 64 blocks, each a dependent device load to a new line, a wait, and F independent fmadd32. `hw-sweep.jsonl`
holds the measurements.

| F (code) | hw us G = 1 / 4 / 8 | model (defaults) | needed overlap |
|---|---|---|---|
| 0 (1.8 KB) | 23.0 / 22.7 / 23.0 | 23.0 / 23.1 / 23.1 | 0.1-0.3 |
| 4 (3.8 KB) | 23.2 / 23.0 / 23.3 | 24.0 / 24.2 / 24.4 | 0.0-0.1 |
| 16 (9.8 KB) | 29.8 / 30.7 / 30.3 | 27.2 / 27.8 / 29.4 | 0.4-0.56 |
| 48 (25.8 KB, beyond the 12 KB i-cache) | 49.5 / 50.3 / 51.8 | 55.1 / 56.3 / 59.9 (+11 / +12 / +16%) | (refetch path) |
| 120 (61.8 KB) | 94.1 / 95.7 / 98.3 | 102.3 / 105.3 / 114.4 (+9 / +10 / +16%) | (refetch path) |

- The number of groups barely changes the hardware time (+4% from 1 to 8 groups). Followers hit the lines the first
  group fetched.
- Inside the i-cache, straight-line code needs an overlap factor of 0-0.56, so 0.3 is right on average. The llama Q8_0
  mat-vec kernel needs about 1.0. It has 28 taken jumps (18 to another line), so its gap is probably the jump fetches,
  not the overlap factor.
- Beyond the i-cache the model over-predicts by 9-16% (more with more groups). The streamed lines cost about 60 ns, not
  the runtime profile's 70 (`AGX_FETCH_COLD_NS=60`: -2..+8%, `resim.py AGX_FETCH_COLD_NS=60`).
  `AGX_REFETCH_LINE_NS` has no effect on these first fetches.
