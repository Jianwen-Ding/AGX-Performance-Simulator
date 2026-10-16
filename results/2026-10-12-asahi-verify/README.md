# Asahi verification 2026-10-12

Device: Apple M2 Max (G14C B1)
Kernel: 6.19.14-400.asahi.fc42.aarch64+16k
Driver: Mesa 25.3.6 (driver_version 104869894 = VK_MAKE_VERSION(25, 3, 6)), Vulkan 1.4.328

The recordings are the ones in `spirv/expected/` (all 181 workloads, recorded on the M2 on 2026-10-12). This report was
produced offline from them, the way `scripts/asahi_verify.sh` would produce it after recording:

```
AGX_THREADS=10 AGX_VERIFY_TIMEOUT=600 python3 -B scripts/spirv_task.py verify all --timing --json verify.json > verify.txt
python3 -B scripts/spirv_task.py probes --json probes.json        # probes.txt: the same rows, one per line
python3 -B scripts/bitfield_question.py > bitfield-question.txt   # coverage/expected/alu: the 70 recorded bitfield cases
```

The full table is in `verify.txt` (one line per workload) and `verify.json`. Summary:

```
{"summary": {"PASS": 155, "CLOSE": 17, "UNSIM": 5, "NODEV": 4}, "timing": {"workloads": 172, "median_factor": 13.59, "p90_factor": 152.176, "within_10pct": 2, "within_25pct": 3, "within_2x": 10, "most_under_predicted": [["rule110", 0.382], ["atomcontend", 0.405], ["floyd", 0.79], ["sssp", 0.796], ["lz4dec", 0.97]], "most_over_predicted": [["swalignpad", 1883.667], ["swalign", 1815.623], ["modexp", 972.798], ["swalignfit", 942.442], ["inflate", 682.508]]}, "stateful_vs_default": [...]}
```

| Status | Count | Workloads |
|---|---|---|
| PASS | 155 | simulator output identical to the GPU's, word for word |
| CLOSE | 17 | softmax, atomics, fft, trig, mlp, hash_table, attention, mplitmus, lookback, cellsort, attndec, attndecsplit, rmsrope, contour, csrbuild, blackscholes, warphash |
| FAIL | 0 | |
| UNSIM | 5 | texfetch, texfilter, imgerode, imgtonemap, cullindirect (textures / images / indirect dispatch; the GPU matches the reference for all five) |
| NODEV | 4 | argmax64, minmax64 (no buffer_int64_atomics), dsum (no float64), splat (no buffer_float32_atomic_add) |

## Probes

```
{"probe": "barriercost", "measured_us": [15.44, 12.06, 25.27, 23.35], "predicted_us": [8.54, 16.57, 45.63, 161.98], "rounds": [0, 16, 64, 256], "us_per_round_measured": 0.03347143817204305, "us_per_round_model": 0.6025123694920769}
{"probe": "atomcontend", "measured_us": [34.94, 31.25, 25.23, 25.33, 25.0], "predicted_us": [11.48, 11.48, 11.48, 11.48, 11.48], "counters": [1, 32, 1024, 32768, 1048576], "hot_over_spread_measured": 1.4, "hot_over_spread_model": 1.0}
{"probe": "smembank", "measured_us": [16.35, 24.35, 23.21, 22.6, 22.52, 22.83, 22.04], "predicted_us": [76.42, 76.45, 77.09, 78.76, 105.9, 192.03, 76.42], "strides": [1, 2, 4, 8, 16, 32, 33], "relative_measured": [1.0, 1.49, 1.42, 1.38, 1.38, 1.4, 1.35], "relative_model": [1.0, 1.0, 1.01, 1.03, 1.39, 2.51, 1.0]}
{"probe": "coalesce", "measured_us": [18.23, 11.65, 11.75, 23.02, 22.6, 21.98, 22.6], "predicted_us": [24.13, 103.51, 196.59, 382.62, 756.06, 405.46, 230.35], "strides": [1, 2, 4, 8, 16, 32, 64], "relative_measured": [1.0, 0.64, 0.64, 1.26, 1.24, 1.21, 1.24], "relative_model": [1.0, 4.29, 8.15, 15.86, 31.34, 16.8, 9.55]}
{"probe": "aluprobe", "measured_us": [25.63, 22.81, 21.9, 21.81, 22.94, 22.6], "predicted_us": [111.96, 52.35, 95.23, 57.27, 116.91, 97.27], "kinds": ["fma32 dep", "fma32 x8", "fma16 dep", "fma16 x8", "imad dep", "imad x8"], "dep_over_x8_measured": [1.12, 1.0, 1.01], "dep_over_x8_model": [2.14, 1.66, 1.2]}
{"probe": "rule110", "measured_us": {"dispatches": 1000, "median": 16.55, "min": 13.71, "max": 21.06}, "predicted_us": {"dispatches": 1000, "median": 6.36, "min": 6.36, "max": 6.36}, "us_per_dispatch_measured": 16.55, "us_per_dispatch_model": 6.36}
{"probe": "stream", "measured_us": [35.02, 26.21, 29.29, 27.38, 22.94, 21.87], "predicted_us": [376.54, 421.82, 653.82, 692.77, 212.4, 314.64], "kinds": ["copy", "scale", "add", "triad", "read", "fill"], "gbps_measured": [3832.5, 5121.2, 6873.2, 7354.4, 2925.7, 3067.9], "gbps_model": [356.5, 318.2, 307.9, 290.6, 315.9, 213.3]}
{"probe": "texfilter", "best_subtexel_bits": {"small": 10, "large": 10}, "hardware_matches_reference": true}
{"probe": "imgtonemap", "round_to_nearest_fraction": {"small": 0.99609375, "large": 0.9960896910365227}, "hardware_matches_reference": true}
{"probe": "mplitmus", "relaxed_stale_reads": {"small": 3392, "large": 4064}, "hardware_matches_reference": true}
{"probe": "f16pack", "half_mismatches": {"small": 0, "large": 0}, "hardware_matches_reference": true}
{"probe": "lookback", "mean_lookback_steps": {"small": 5.267, "large": 27.726}, "hardware_matches_reference": true}
{"probe": "viterbi", "mismatched_bits": {"small": 4923, "large": 634159}, "hardware_matches_reference": false}
{"probe": "viterbifix", "mismatched_bits": {"small": 0, "large": 0}, "hardware_matches_reference": true}
{"probe": "shiftfuse", "mismatches": {"small": {"bit_test": 1520, "byte_extract": 2922, "double_and_mask": 1944}, "large": {"bit_test": 393363, "byte_extract": 759283, "double_and_mask": 507882}}, "hardware_matches_reference": false}
{"probe": "rmsrope", "max_rel_error": {"small": 0.0007370990065600776, "large": 0.0007995462332695336}, "hardware_matches_reference": true}
{"probe": "blackscholes", "max_abs_error": {"small": 1.9854037304867234e-05, "large": 2.694422661875251e-05}, "hardware_matches_reference": true}
{"features_missing_on_device": {"argmax64": "buffer_int64_atomics", "minmax64": "buffer_int64_atomics", "dsum": "float64", "splat": "buffer_float32_atomic_add"}}
```

The probes' measured times have the timestamp problem described below; only the functional rows are usable.

## bfeil shift question (applications notes 37-40)

```
{"cases": 70, "masked_bfeil_cases": 4, "default_fails": [], "mod32_fails": ["bfeil_r14_r6_r7_r8_mask_0x1", "bfeil_r14_r6_r7_r8_mask_0x7FFFFFFF", "bfeil_r14_r6_r7_r8_mask_0xFF", "bfeil_r14_r6_r7_r8_mask_0xFFFF"], "verdict": "A: masked bfeil takes 7 bits like the full-width form; viterbi / shiftfuse / lz4dec failures are compiler bugs (fuse_ubfe)"}
```

PASS = simulator identical to the GPU word for word; CLOSE = differs, both within the reference tolerance;
FAIL = a simulator (or GPU) bug to investigate; PENDING = not recorded.

## Root-table layout of Mesa 25.3.6

The first offline verify ran the recorded code with `spirv/root_layout.mesa-26.0.json`, and the simulator read the
wrong push constants (`bitonic` looped forever, for example). The recorded preambles of all 177 workloads with code
were compared with the 26.0.3 shim compiles (`spirv/compiled/mesa-26.0.3`). Every root-table and descriptor load is
the same in both, with one exception: **push word n is root word 124 + n in 25.3.6 and 122 + n in 26.0.** The
evidence covers 149 workloads loading word 124, 90 loading 125, and so on up to `pushmat`, which loads 124..187. The
loads that match 26.0 are:

- u0_u1 holds the table's own address. u2_u3.. hold the set addresses, which the driver preloads.
- `group_count_addr` is at words 18..19. Six workloads load it: aluprobe, geom3d, hist_eq, barriercost, coalesce and smembank.
- `base_group` is at words 20..22. 173 workloads load it.
- Descriptors are 16-byte `hk_buffer_address` entries at the same set offsets. Twelve distinct u2_u3 offsets appear, identical in both.

So the draw arm of the cs/draw union in front of `push` is 8 bytes longer in 25.3. This is now in
`spirv/root_layout.mesa-25.3.json`.

Layout selection now uses the compiling driver's `driver_version`:

| Driver version | Layout |
|---|---|
| ≤ 25.2 | 25.2 |
| 25.3 | 25.3 |
| ≥ 26 | 26.0 |

The version is read from the code directory's `device.json` (written by `compile`), from the recording's `hardware.json`
next to `code/`, or from the recorded device that `record` passes to `extract`. The old preamble heuristic (`u2_u3` read → 26.0) is now only
a fallback. It had also mislabelled the texture-only shaders: imgtonemap and imgerode in 26.0.3 were labelled 25.2,
and imgtonemap, imgerode, texfetch and texfilter in 25.2.8 were labelled 26.0. These labels are corrected.

## The recorded GPU times are not kernel durations

**Fixed 2026-10-16:** vk_run now writes two timestamps before each dispatch, and every workload was re-timed. See
[2026-10-16-asahi-retime](../2026-10-16-asahi-retime/README.md), which also settles Q2b and Q3.

Per-dispatch GPU timestamps on this driver do not bracket the dispatch's execution. vk_run writes a TOP_OF_PIPE
timestamp before each dispatch and a BOTTOM_OF_PIPE timestamp after it. All 134 single-dispatch workloads measure
11-41 µs (median 31 µs), whatever their size. Examples:

| Workload | Measured | What it would imply |
|---|---|---|
| sgemm | 32.7 µs | 65.6 TFLOP/s |
| stream | 22-35 µs per dispatch | 2.9-7.4 TB/s |
| swalign | 39.8 µs | 5.4·10⁸ cells, 13.5 Tcells/s |
| nbody | 27.3 µs | 197 TFLOP/s |

50 recordings imply more than 13.6 TFLOP/s or 420 GB/s. The timestamps behave as if the end stamp is taken when the
dispatch is launched, not when it completes.

The `--timing` summary above, every `measured_us` / `dispatch_ns_median`, and the probes' measured columns are
therefore not usable as kernel times. This explains the median factor of 13.6.

What does bracket real work is a multi-dispatch recording's total `gpu_ns`, from the first start to the last end.
Barriers serialise the dispatches, so the total includes the execution of every dispatch but the last (plus barrier
gaps). On the 41 multi-dispatch workloads, predicted ÷ that total has a median of **0.71**. The values range from 0.14
(rule110, where 1000 tiny dispatches are dominated by launch overhead) and 0.01 (atomcontend) to 6.2 (jpcolor) and
8.4 (attndecsplit).

Fixing this needs a hardware change: time with host_ns per submit, or with one dispatch per submit, or check how
honeykrisp resolves compute timestamps. That cannot be done offline. `vk_run` already prints `host_ns`, but `record` does not keep it.

## FAILs

None. The five workloads where the GPU disagrees with the numpy reference (viterbi, shiftfuse, lz4dec, bitap and cdc)
are PASS: the simulator reproduces the GPU's wrong output word for word. This is Mesa's `fuse_ubfe` miscompile under
verdict A (question 1). With 26.0.3 and 25.2.8 code, `simall` gives the same reference mismatches (shiftfuse small
1520 / 2922 / 1944, the same as the GPU).

## CLOSE causes (no simulator bug identified)

The ulp figures compare the simulator with the GPU on the small size.

| Workloads | Cause |
|---|---|
| atomics, hash_table, cellsort, csrbuild, warphash, contour | Atomic / scheduling order. Integer slots, or values placed by atomic cursors, land in a different order. |
| mplitmus (3 words), lookback (1 word) | Racy litmus counters and look-back step counts, by design. |
| softmax, mlp | All differing words are within 4 ulp: `exp2` / `rcp` rounding. |
| trig | 99.9 % within 4 ulp, max 192: the `sin_pt` / `cos` path. |
| fft, attention, attndec, attndecsplit, blackscholes | Mostly 1-4 ulp, with tails up to 10⁴ ulp where results cancel. |
| rmsrope | Only 41 % within 4 ulp. Large ulp tails on near-zero rotated values, with the GPU still within 8·10⁻⁴ relative error. |

The exact recorded rcp / rsqrt tables are in use (`testsuite/transcendental`). The remaining float differences come from
`exp2` / `log2` / `sin_pt`, which are modelled as correctly rounded.

## Checklist answers (`tasks/m2-run-checklist.md`)

- **Q1, bfeil shift: verdict A.** A masked `bfeil` takes 7 bits like the full-width form. The 4 masked cases fail the
   mod-32 model, and the default model fails none. Keep the default `bfeil_shift`.
   - Mesa's `fuse_ubfe` is unsound for variable shifts. viterbi, shiftfuse (3 columns), lz4dec, bitap and cdc FAIL on
     the GPU against the reference, and the simulator matches the GPU.
   - `viterbifix` (the workaround) PASSes and the hardware matches its reference.
   - Report upstream with note 37's root cause.
- **Q2, SR58 / SR59 count active lanes: confirmed.** `bfs` PASSes word for word.
- **Q2b, write-back: not settled.** The per-dispatch times are unusable (above). There is a weak hint from stream's
   total `gpu_ns` of 2187 µs, which covers copy, scale, add, triad and read plus barriers. The read-only model predicts
   2357 µs for those five, so the GPU is not slower than the model without write-back charging. That leans against
   making it the default. Re-time stream and adamw before changing `charge_writeback`.
- **Q3, stateful timing.** The verify summary's `stateful_vs_default` uses the broken per-dispatch sums. Against the
   multi-dispatch totals, stateful is closer for every workload where the two differ, except warphash (where they are
   within 0.4 %):

   | Workload | Total `gpu_ns` | Default | Stateful |
   |---|---|---|---|
   | bfs | 3530 µs | 919 | 2958 |
   | sssp | 4681 | 858 | 1929 |
   | cellsort | 1696 | 639 | 1154 |
   | csrbuild | 1137 | 386 | 905 |
   | boruvka | 12656 | 42948 | 32228 |
   | jpcolor | 4631 | 28786 | 9095 |

   This supports making `--stateful` the default in `predictall` for the frontier / converging workloads. Confirm it
   once the timestamps are fixed.
- **Q4, transcendental differences: no, not only 1 ulp.**
   - cholesky PASSes, which is exact and better than the expected CLOSE.
   - blackscholes is CLOSE with 6028 words: 68 % within 4 ulp, max 7450 ulp. The GPU's max_abs_error is 2.7·10⁻⁵.
   - rmsrope is CLOSE with 56554 words, max_rel_error 8·10⁻⁴.
   - With the recorded rcp / rsqrt tables already in use, the remaining work is exact `exp2` / `log2` / `sin_pt`
     models. Only samples of those are recorded so far.

Results that must hold whatever the timings: rsshash, poly1305, siphash, blake2s, blake2sfast, mt19937, crc32c, sha3,
edt, bvhrays, ghash, scrypt, pngunfilter, inflate and inflateshm all PASS. csrbuild is CLOSE (atomic order), and the
GPU and simulator both match scipy. lz4dec matches the GPU but not the reference (question 1, verdict A).
