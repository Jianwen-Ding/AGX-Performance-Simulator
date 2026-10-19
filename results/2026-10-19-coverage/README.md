# Coverage report, 2026-10-19 (`tasks/agx-coverage.md` step 5)

Device: MacBook M2 Max (driver-reported G14C B1), Asahi Linux, direct DRM. Simulator: this repo
at `c43fbbf` (all Rust binaries built locally). No new GPU recordings in this report; it
summarises the retained `coverage/expected` + `coverage/timing` against the current simulator.

## Pass counts per area (`suite_check coverage/expected`)

| Area | Pass | Total | Not passing |
|---|---|---|---|
| alu | 863 | 863 | — |
| alu_masked | 213 | 213 | — |
| atomic (device 11 + threadgroup 11) | 22 | 22 | — |
| control | 75 | 75 | — |
| fmadd_b52 | 2 | 2 | — |
| fmadd_lane | 16 | 16 | — |
| fmadd_lane_masked | 8 | 8 | — |
| geometry | 623 | 634 | 11 unsupported (see below) |
| mem_load | 202 | 202 | — |
| mem_store | 194 | 194 | — |
| sync | 5 | 5 | — |
| threadgroup | 44 | 44 | — |
| undecoded | 403 | 403 | — |
| **total** | **2670** | **2681** | **11 unsupported, 0 mismatch, 0 close** |

432 words accepted from racing stores; 3,972 placement-dependent words (SR17/SR20) compared
under both placements. `make coverage-check` exits 1 solely on the 11 unsupported cases below
plus the 5 timing rows below; there are no value mismatches anywhere.

## Simulator mismatches by cause

None. Every compared word is exact (0 within-2-ulp "close" cases).

## Unsupported cases (11, all `get_sr` in geometry — launch context the checker cannot provide)

- 10 x `geometry/sr21_*`: SR21 is the VM slot, which comes from launch context
  (`GroupCtx.launch.context`); `suite_check` builds contexts with `Default::default()`
  (no VM), so `func::run` errors instead of guessing. The hardware values are correct in
  `coverage/expected`; the simulator test `cached_special_registers_match_geometry_*` covers
  SR21 with an explicit context. Not a simulator bug: without the launch's VM slot the value
  is unpredictable by design (cf. `novel-results/2026-10-16-special-registers`).
- 1 x `geometry/sr113_*`: SR113 is the time-varying firmware power-state bit, deliberately
  unmodelled (`cost.rs`: "can change mid-dispatch"). Same disposition as SR113 elsewhere.

## Faults (`coverage/faults.json`, 2 entries)

- `control/while_icmp_ugte`, `control/while_icmp_sgte`: test-design infinite loops (counter
  from 1 against a bound it never reaches), recorded during the 2026-10-12 session; the GPU
  recovered after each. Bounded `while_guard_*` replacements recorded 2026-10-17.
- 6 related `may_fault` control cases stay pending by design: `while_fcmp_{gt,gte,gtn,neq,nlt,nlte}`
  never exit for any input, so they cannot be recorded (`agx_run` refuses them at the step
  limit). No GPU time should be spent on them.

## Timing rows more than 10% off (5 of 1,290; median relative error 0.4%)

| Case | Mode | Measured | Model | Off by |
|---|---|---|---|---|
| bench/fcmp_ballot_r14_gtn_r6_r7 | simd_issue | 2.398 | 2.002 | +20% |
| bench/fcmp_ballot_r14_lte_r6_r7 | simd_issue | 2.603 | 2.002 | +30% |
| bench/fcmpsel_lt_r14_r6-abs_r7-neg_r8-cache_r9-discard | simd_issue | 4.853 | 2.097 | +131% |
| bench/icmpsel_ult_r14_r6_r7_r8-cache_r9-discard | simd_issue | 4.855 | 2.097 | +132% |
| bench/simd_min-f_r14l_r6l | latency | 25.358 | 22.021 | +15% |

Notes for the cost-model work: the two `r9-discard` rows suggest the model misses a cost of
the discard operand (result-cache eviction/writeback) in `simd_issue` mode; the two ballot rows
suggest unmodelled cross-lane-unit contention; `simd_min-f` latency is 3.3 cycles high. None
was touched here — they belong to the safe-02 cost-model gap work.

## Nondeterministic cases

None: no expected file sets `"deterministic": false`, and the two `while` faults are
deterministic test-design hangs, not flaky hardware.

## Property checks

`make coverage-properties` (playground offline verifiers standing in for the MANUAL items):
20/20 pass, 0 failures — including the 2026-10-19 additions (`verify-sb-minimal`,
`verify-fence-contention` under `memory_ordering`).
