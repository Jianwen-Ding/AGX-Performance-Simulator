# Re-timing on the M2, 2026-10-16: real per-dispatch times

Device: Apple M2 Max (G14C B1), Mesa 25.3.6, kernel 6.19.14-400.asahi. **The machine was on battery** (AC offline,
57 %, discharging) for every run in this directory.

The 2026-10-12 recordings' times did not bracket kernel execution (median predicted / measured factor 13.6; sgemm
"65.6 TFLOP/s"). This run fixes the measurement, re-times all 177 recorded workloads without touching their functional
recordings, and answers the open checklist questions (`tasks/m2-run-checklist.md`).

```
python3 -B scripts/spirv_task.py retime all                                         # GPU, ~25 min; log in /scratch/agxsim-runs/retime-2026-10-16.log
AGX_THREADS=10 python3 -B scripts/spirv_task.py verify all --timing --json verify.json > verify.txt
python3 -B scripts/spirv_task.py probes --json probes.json > probes.txt
```

## 1. Why the old timestamps measured nothing, and the fix

Honeykrisp (`hk_query_pool.c`, `hk_CmdWriteTimestamp2`) makes a compute timestamp the **end** of the current control
stream and splits at the next timestamp write. vk_run wrote `TS(2i); dispatch i; TS(2i+1)`: `TS(2i)` became the end
of dispatch i's own stream and `TS(2i+1)` the end of an empty stream, so `TS(2i+1) - TS(2i)` was stream overhead
(11-41 µs whatever the size). The playground had found the same thing on 2026-10-10 (`command-timestamps.md`).

`tests/vk_run.c` now writes two timestamps **before** each dispatch: `TS(2i)` ends the previous (empty) stream and
`TS(2i+1)` ends the stream that receives dispatch i, so `dispatch_ns = TS(2i+1) - TS(2i)`. Check: GPU time is now
below host submit-to-idle time by 120-130 µs on every single-dispatch workload, and it scales with the work:

| Workload (large) | Old "time" | New time | Implies |
|---|---|---|---|
| sgemm | 32.7 µs | 1720 µs | 1.25 TFLOP/s |
| nbody | 27.3 µs | 1963 µs | 2.0 TFLOP/s (model over-predicts 3.3x) |
| stream (6 dispatches) | 22-35 µs each | 220-577 µs each | 298-360 GB/s per kind |

## 2. Two hazards found on the way

**Silent fallback to llvmpipe.** For a few minutes, vk_run's default device (index 0) was llvmpipe instead of the
GPU, at the same moment the display powered off (kernel log, DCP poweroff). Results were correct to 6e-7, so the run
looked plausible: 60 ms per submit, 40-2500 ns "GPU" times, 42 W system power (12 CPU cores). The output's SHA-256
equals a forced-lavapipe run of the same spec. `spirv_task.py` refuses non-GPU devices (`real_gpu_problem`), but raw
vk_run calls did not. vk_run now defaults to the first non-CPU device.

**Clock states on battery.** sgemm-large runs at ~1720 µs, ~2730-2900 µs or ~3646 µs (2.12x) depending on recent
load. On battery, **sustained** load (back-to-back submits for ≥ 0.3 s) settles in the slow state, while repeated
runs with host gaps (vk_run's buffer re-initialisation) mostly reach the fast state. The first run after a few
seconds of idle is often slow. This is the opposite of the 2026-10-10 observation (slow at start, fast after 6-9
dispatches), which may have been on AC power. The fast state matches the model's top-clock assumption (sgemm 1.19,
stream 1.10, reduce 0.84 predicted / measured).

`run_one` now keeps every run's total (`gpu_ns.runs`), and the timing fields are medians over the runs within 5 % of
the fastest (`gpu_ns.fast_runs` of `repeat`; `median_all` keeps the plain median). Recordings say
`"timing_method": "two-before"`; `retimed_utc` marks this run. A `warmup MS` directive was added to vk_run for
experiments, but it lowers the clock on battery, so `retime` does not use it.

## 3. Summary

```
{"summary": {"PASS": 155, "CLOSE": 17, "UNSIM": 5, "NODEV": 4},
 "timing": {"workloads": 172, "median_factor": 1.821, "p90_factor": 3.891, "within_10pct": 21, "within_25pct": 45, "within_2x": 98}}
```

The factor is 2^|log2(predicted / measured)|, so 1.82 means "within 1.82x either way". It was 13.6 with
the old timestamps. The functional table is unchanged from 2026-10-12. The 9 workloads whose large-size output hash
differs from the recording are all racy by design (atomics, cellsort, contour, csrbuild, cullindirect, hash_table,
lookback, mplitmus, warphash; `large.out_sha256_same: false`).

## 4. Checklist answers

- **Q2b, does store write-back cost DRAM bandwidth? Yes, about as much as reads, but neither model setting gets this
  right.** stream, measured, same bytes per dispatch: read 222 µs, fill (pure stores) 225 µs. Default model: read 212,
  fill 314. With `AGX_CHARGE_WRITEBACK=1`: fill 571, add 993 against 568 measured, and stream's total 3603 against
  2420 measured (default 2670). adamw: measured 348 µs, default 202, write-back 328. So stores take DRAM bandwidth at
  roughly the read rate. The default under-charges a load-modify-store loop like adamw. Write-back charging over-charges,
  because it stacks the written lines on top of the store misses already counted. **Keep the default.** The fix is in
  `cost.rs` (`charge_writeback`): charge each dirty line once at DRAM rate, and charge pure store misses no more than
  read misses.
- **Q3, stateful timing: confirmed; make `--stateful` the default for the frontier / converging workloads.** Where
  the two predictions differ, stateful is closer every time: bfs 0.36 → 1.15, sssp 0.26 → 0.58, cellsort 0.31 →
  0.56, boruvka 3.98 → 2.98, jpcolor 7.16 → 2.27, csrbuild 0.51 → 1.20 (predicted / measured).
- **Q1, Q2, Q4:** unchanged from 2026-10-12 (functional answers).

## 5. Measurements against the model (checklist table; predicted / measured)

| Workload | Measured µs | Predicted µs | Ratio |
|---|---|---|---|
| `blake2s` | 208515 | 13101 | 0.06 |
| `blake2sfast` | 689 | 163 | 0.24 |
| `attndec` | 2794 | 2442 | 0.87 |
| `attndecsplit` | 1820 | 634 | 0.35 |
| `spmv` | 996 | 802 | 0.80 |
| `spmvsg` | 323 | 405 | 1.25 |
| `atomcontend` | 7492 | 57 | 0.01 |
| `cellsort` | 2077 | 639 | 0.31 |
| `hough` | 687 | 239 | 0.35 |
| `rle` | 2768 | 2092 | 0.76 |
| `varint` | 4106 | 2136 | 0.52 |
| `lookback` | 612 | 55 | 0.09 |
| `q4gemv` | 921 | 200 | 0.22 |
| `q8dot` | 267 | 64 | 0.24 |
| `biquad` | 4851 | 3727 | 0.77 |
| `poly1305` | 581 | 322 | 0.55 |
| `siphash` | 1016 | 257 | 0.25 |
| `edt` | 1539 | 737 | 0.48 |
| `crc32c` | 1514 | 701 | 0.46 |
| `boruvka` | 10803 | 42943 | 3.98 |
| `adamw` | 348 | 202 | 0.58 |
| `stream` | 2420 | 2670 | 1.10 |
| `motionest` | 3311 | 4889 | 1.48 |
| `jsonidx` | 605 | 287 | 0.48 |
| `cdc` | 25668 | 5950 | 0.23 |
| `swalignpad` | 44215 | 56203 | 1.27 |
| `swalignfit` | 22685 | 32647 | 1.44 |
| `swalign` | 54398 | 71829 | 1.32 |
| `jpcolor` | 4022 | 28779 | 7.16 |
| `postings` | 4430 | 2069 | 0.47 |
| `tpchq6` | 382 | 328 | 0.86 |
| `ghash` | 1297 | 705 | 0.54 |
| `scrypt` | 22421 | 17366 | 0.77 |
| `pngunfilter` | 6987 | 5890 | 0.84 |
| `inflate` | 24912 | 20748 | 0.83 |
| `inflateshm` | 23747 | 20539 | 0.86 |
| `csrbuild` | 754 | 386 | 0.51 |
| `blackscholes` | 599 | 338 | 0.56 |
| `warphash` | 6249 | 6629 | 1.06 |
| `nms` | 999 | 77 | 0.08 |
| `sgemm` | 1742 | 2070 | 1.19 |
| `nbody` | 1987 | 6597 | 3.32 |

What these say about the hardware and the model:

- **Shared-memory occupancy cliff confirmed:** swalignpad (33280 B / workgroup) takes 1.95x swalignfit (32768 B), on
  the same compiled code. The model predicted 1.72x.
- **Dynamically indexed private arrays are catastrophic:** blake2s takes 303x blake2sfast for identical work. The
  model predicted 80x.
- **Same-address atomics serialise.** The `atomcontend` probe gives 3633 µs for one hot counter against 154-170 µs
  spread over ≥ 32768 counters, 21x apart. The model is flat (1.0x). This is the largest single gap (0.01).
- **Look-back / flag chains** (`lookback` 0.09) and the growing-work `nms` (0.08) are under-predicted, as the notes
  expected.
- **Strided access** (`coalesce` probe): stride 16 costs 5.96x stride 1. The model says 31x, so it over-penalises
  uncoalesced access.
- **Barriers** (`barriercost`): 1.57 µs per round against the model's 0.60.
- **inflate ≈ inflateshm** (24.9 / 23.7 ms): 131 KiB of select-chain code costs only 5 % over 20.5 KiB, so there is
  no large instruction-cache penalty.
- **nbody over-predicted 3.3x**, jpcolor / boruvka / pagerank 4-7x: these are the next model items.

Probe rows with real times are in `probes.txt`. The stream probe's GB/s are now physical (298-360 GB/s).
