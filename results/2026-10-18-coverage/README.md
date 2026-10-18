# Coverage report, 2026-10-18

State of the AGX coverage suite (`coverage/`, `scripts/coverage.py`) after the 2026-10-17 / 2026-10-18 sessions. All recordings are on one
device (`M2 Max G14C B1 (direct DRM)`), the oldest from 2026-10-12, the newest from 2026-10-18. Numbers come from `coverage.py status`,
`suite_check coverage/expected`, `coverage.py check --timing` and the recorded files.

## Functional cases (simulator against hardware recordings)

| Area | Recorded | Simulator exact | Unsupported | Mismatch | Not deterministic on hardware |
| --- | --- | --- | --- | --- | --- |
| alu | 863 | 863 | 0 | 0 | 0 |
| alu_masked | 213 | 213 | 0 | 0 | 0 |
| atomic | 22 | 22 | 0 | 0 | 0 |
| control | 75 | 75 | 0 | 0 | 3 |
| fmadd_lane (new) | 16 | 16 | 0 | 0 | 0 |
| fmadd_lane_masked (new) | 8 | 8 | 0 | 0 | 0 |
| fmadd_b52 (new) | 2 | 2 | 0 | 0 | 0 |
| geometry | 634 | 623 | **11** | 0 | 25 |
| mem_load | 202 | 202 | 0 | 0 | 0 |
| mem_store | 194 | 194 | 0 | 0 | 0 |
| sync | 5 | 5 | 0 | 0 | 0 |
| threadgroup | 44 | 44 | 0 | 0 | 0 |
| undecoded | 403 | 403 | 0 | 0 | 0 |
| **Total** | **2,681** | **2,670 (all exact, 0 within 2 ulp)** | **11** | **0** | **28** |

**Simulator mismatches by cause: none.** Every case it can run equals the hardware word for word (432 words taken from racing stores and
3,972 placement-dependent words, SR17 / SR20, are not compared by design).

**The 11 unsupported cases, all `get_sr`:**

* 10 x **SR21** (the hardware VM slot: `sr21_*` in 10 launch geometries): the simulator models it (it is a launch parameter), but the recorded value
  is the slot the process happened to get (1..63, least recently used), so these cases cannot be compared without that slot.
* 1 x **SR113** (`sr113_1024x1x1_wg256x1x1`): a global power-management state bit that changes during a dispatch; not modelled.

**Not deterministic on hardware** (recorded twice, outputs differ): 25 geometry cases (SR17 x 3, SR20 x 8 and SR21 x 10, which are placement / VM-slot
dependent and compared only where they are stable, plus SR1 x 2, SR49 x 1 and SR81 x 1, whose cause was not investigated) and 3 control cases
(`push_count_1`, `_2`, `_3`, an `if_icmp` followed by `pop_exec n`: the output words differ between the two dispatches; the cause was not investigated).
All other 2,653 cases were identical in both recorded dispatches.

## Faults

2 recorded faults, both **test design, not the ISA** (`coverage/faults.json`): `control/while_icmp_sgte` and `control/while_icmp_ugte` loop forever
by construction (a counter that counts up against a bound it never reaches); the bounded forms (`while_guard_*`) were recorded 2026-10-17.
No safe case faulted.

## Cases not recorded (`may_fault`, need `--risky` and the owner's approval)

837 pending: alu 292, mem_load 116, mem_store 112, sync 251 (every `memory_barrier a, b, c`), threadgroup 60, control 6. The risky atomics (11) and
40 risky control cases are recorded.

## Timing (cost model against hardware instruction-bench rows)

651 bench forms x latency / SIMD issue: **1,290 rows, median relative error 0.42%, 1,285 within 10%, 0 without a model.** The 5 rows more than
10% off:

| Row | Mode | Measured | Model |
| --- | --- | --- | --- |
| `fcmp_ballot ... gtn` | SIMD issue | 2.398 | 2.002 |
| `fcmp_ballot ... lte` | SIMD issue | 2.603 | 2.002 |
| `fcmpsel lt r6.abs r7.neg r8.cache r9.discard` | SIMD issue | 4.853 | 2.097 |
| `icmpsel ult ... r8.cache r9.discard` | SIMD issue | 4.855 | 2.097 |
| `simd_min.f r14l r6l` | latency | 25.358 | 22.021 |

(cycles per step). The two select rows with `.cache` and `.discard` sources issue in about 4.85 cycles on hardware but about 2.1 in the model: a
register-cache interaction the model does not capture. The two ballot rows are 20-30% slow in the model's favour (condition-code dependent issue). The
`simd_min.f` latency row is 15% over.

## Other regression targets (not part of the suite)

`make verify-cost-model` (output identical to the retained file), `make suite-check` (721 / 721), `cargo test` (all pass), and the playground
verifiers listed by `make coverage-properties` (17, all pass). `make verify-traces` reports 652 / 721: 69 cases where the cost model's emulator
and the functional simulator take different control-flow paths (control: 18, random: 50, kernel: 1); the check says a mismatch there means the
path depends on data, and it was already at 652 before the 2026-10-18 changes.
