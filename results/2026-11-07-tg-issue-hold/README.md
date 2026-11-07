# Threadgroup loads hold their SIMD group's issue (2026-11-07)

Measured on the M2 Max (G14C, AC power) with `scripts/probe_tg_alu.py` (agx_run_hw back to back, firmware times; slope
between 2048 and 6144 loop trips; 32 workgroups, one per core, so the busiest core has exactly NG SIMD groups).
Kernel durations 0.3-4 ms, 40 back-to-back commands after a warm-up burst: the reference loop `iiiiiiii` measures
45.0 / 46.0 core cycles per trip at 1 / 4 groups (model 45.8), i.e. the top clock. Shorter runs (256-768 trips, 100
commands) swing between two clock states (261 vs 201 "cycles" for the same loop) and are not used.

Pattern letters (see the script): `L` threadgroup load with a register base, `B` with an immediate base, `X` an `xy`
load, `i` an independent `iadd`, `f` an independent `fmadd32`, `c` an `iadd` consuming the latest load, `w` `wait 0`.
Every trip also runs the 3-instruction loop tail (`iadd`, `while_icmp`, `jmp_exec_any`).

## Findings (real core cycles per trip per group)

| pattern (per trip) | 1 group | 8 | 32 |
| --- | --- | --- | --- |
| `LLLLLLLL` | 261.0 | 260.9 | 293.4 |
| `LiLiLiLiLiLiLiLi` | 401.9 | 402.9 | 416.4 |
| `LLiiLLiiLLiiLLii` | 326.0 | 329.1 | 354.6 |
| `LLLLiiiiLLLLiiii` | 288.0 | 292.7 | 329.2 |
| `LLLLLLLLiiiiiiii` | 269.3 | 272.8 | 305.7 |
| `LwLwLwLwLwLwLwLw` | 261.0 | 261.0 | 292.3 |
| `XXXXXXXX` | 270.5 | 283.0 | 591.7 |
| `BBBBBBBB` (1 group) | 223.9 | | |
| `BwBw...`, `BiwBiw...` (1 group) | 229.0, 389.0 | | |

* A threadgroup load holds its group for ~28 cycles (register base; ~24 with an immediate base) whatever follows: an
  independent `iadd` after it issues only then (`Li` pairs cost 50 cycles, not 30).
* A threadgroup load after ALU work issues only after the ~21-cycle quiet time after the last ALU instruction (the
  quiet time the model already had, but charged only to the load's result). Each ALU run between loads costs ~20
  cycles whatever its length (1, 2, 3 or 7 instructions; `iadd` or `fmadd32`): `LLii` x4 = 4 x (2 x 28 + 2 + 21).
* A `wait 0` right after a threadgroup load is free (the result arrives within the hold).
* Up to 32 groups per core the per-group rate barely changes for `x` loads (core: 0.87 loads per cycle); `xy` loads
  at 32 groups take 2.3 core cycles each (two words per lane). The 2.2x slowdown "at 32 groups" in
  novel-results/2026-10-23-geekbench-captures came from 38 workgroups of 1024 threads: four cores get two of them
  (64 groups).

## Model change (rust/src/cost.rs)

`tg_load_hold_real` (24) + `tg_load_hold_reg_extra_real` (4, any register source) is the issue interval of a
threadgroup load, and the quiet time after ALU work delays the issue (the group waits). With the memory model the
load also waits for the core's threadgroup pipeline (bank-conflict degree x 1 cycle per access) before issuing. The
group's own serial access time (31 units + conflict degree) still delays the result, as before. `AGX_TG_ISSUE_REAL=0`
restores the earlier model.

| set | before | after |
| --- | --- | --- |
| dev (`hw-dev.jsonl`, 14 patterns x 1/8/32 groups, used to fit 24 + 4) | median 24.1%, 7 of 42 within 10% | 2.2%, 40 of 42 |
| held out (`hw-heldout.jsonl`, 8 new patterns x 2/16/24 groups) | median 30.9%, 0 of 24 within 10% | 1.8%, 22 of 24 |
| tg.json (in-shader, stride 0-32, 1-32 groups) | 2.2% (p90 10.7%) | 2.0% (p90 3.6%) |
| verify-cost-model heldout-mem24b, 4 groups | 6.5% | 4.2% |

verify-dispatch unchanged (4.8%), other memory sets unchanged. Remaining: `Liiiiiii` at 32 groups -40% and
`iiiiiiii` at 32 groups (hw 171 vs model 95 cycles per trip: independent `iadd`s issue at ~2 per cycle per core,
not ~4), `xy` loads at 24 groups +22%, `Lw` +9.5%.
