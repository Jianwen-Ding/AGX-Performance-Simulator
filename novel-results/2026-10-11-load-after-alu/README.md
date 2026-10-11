# Memory instructions issued soon after ALU work start late; ALU work in flight defers loads

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, one SIMD group, warm L1
(the same word every repetition). Known-safe encodings only; reference units (× 1.049 = real cycles).

## Claim and scope

A `device_load` + `wait 0` costs **44.8 units (47 real cycles)** when no ALU instruction issued shortly before
it, and **~63 units (66 real cycles)** when one did (see the rule below for what "shortly" means), whatever the ALU
instruction (`iadd`, `fadd16`, `mov_imm`, `get_sr`), whether or not it produces the load's address, and however
far before the load it issued (0–24 instructions) or whether its result was evicted from the result cache.

| Body per repetition | units |
| --- | --- |
| load, wait | 44.8 |
| load, wait, load, wait (independent; the dependent load→load variant, index from the first load, also gave 89.6) | 89.6 (= 2 × 44.8) |
| iadd, load, wait | 63.9 |
| fadd16 / mov_imm / get_sr, load, wait | 63.8 / 60.1 / 67.6 |
| load, iadd, wait (ALU while the load is in flight) | 44.8 |
| load, wait, iadd (the iadd precedes the next load) | 61.6 |
| store, load, wait | 52.9 |

ALU instructions issued **while the load is pending** are free for the first ~8, then push completion out:
with a = ALU instructions before the load and b after it (30-point grid, `load-alu-grid.json`), completion
is at least ~31 units (a = 0) or ~56 units (a ≥ 1) after the last ALU instruction issued before the wait.

| a \\ b | 0 | 4 | 8 | 16 | 24 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 44.8 | 45.8 | 46.1 | 53.8 | 62.2 | 69.1 |
| 1 | 63.8 | 65.7 | 69.2 | 77.1 | 84.6 | 92.4 |
| 4 | 67.6 | 71.5 | 73.9 | 80.6 | 88.8 | 95.9 |
| 8 | 75.5 | 75.8 | 76.8 | 84.6 | 92.0 | 100.7 |

**Rule (fits the 30-point grid at 1.7% median, worst 4.4%, 15 memory-state probes at 2.5% median and 7 memory-space transitions within ~1 unit).** A memory instruction cannot start earlier than a fixed quiet time after the
group's last ALU issue: **~20 units for device loads and threadgroup loads/stores, ~34 units for device
stores** (`memstate.json`, `tgmem.json`). It is not the `wait`, nor whether the pipe was idle: `iadd, wait,
load, wait` is slow (64.8) because the idle `wait` costs only ~4.9 units (< 20), while `load, iadd, wait` is fast (44.7)
because the next load issues ~40 units after that `iadd`; a loop's own counter `iadd` (33+ units before the
next iteration's first store, across the taken branch) does not slow the store. Stores then issue in 3.1
units (device) or 17.1 (threadgroup); a device store completes ~18 units after an on-time issue. Two
threadgroup loads serialize (~23 units each). Transitions between a device store and a threadgroup access
(either order) cost ~6.7 units each; device↔threadgroup loads and a threadgroup store followed by a device
load cost nothing extra (`spaces.json`); a `wait` whose slot has nothing pending costs ~4.9
units, and 32 consecutive waits cost ~11 units each (the run-length effect of the
[wait-cost finding](../2026-10-11-wait-cost/README.md)).

| Body per repetition (units) | measured |
| --- | --- |
| device store / store, store / store, wait | 3.2 / 6.2 / 21.5 |
| iadd, store / iadd, wait, store / iadd, store, store | 35.8 / 35.8 / 37.1 |
| store, 8 independent iadds (next store follows them) | 44.5 |
| threadgroup store / iadd, threadgroup store | 17.2 / 36.3 |
| threadgroup load, wait / iadd, threadgroup load, wait / 2 threadgroup loads, wait | 22.9 / 41.9 / 45.8 |

**Review follow-up (`review2.json`).** The window probe (`iadd`, k idle `wait 1`, load, `wait 0`; idle waits cost
4.9 units each) gives 63.7, 64.6, 66.8, 68.0, 71.4, 81.5, 91.1 for k = 0–8: almost flat until the waits fill
~20 units, then +4.9 per wait. That is the time rule; the alternative "slow whenever ALU issued while no
memory operation was outstanding" predicts +4.9 per wait from k = 0 (78.4 at k = 3, measured 68.0) and is
refuted for loads. Threadgroup loads are deferred by ALU work in flight to ≥ ALU issue + ~20 units
(`tg load, iadd, wait` 42.0, with 8 iadds 53.6). **Stores do not follow the rule cleanly** (open): with k idle
waits between an `iadd` and a store the store part falls from 35.9 to ~24 units after 3 waits and stays there;
`store, k iadd` costs 33.5–52.6 for k = 1–16, as the rule predicts for k ≥ 8 but less for k ≤ 4; and in
`store, 8 iadd` the first store after the loop control is slow on hardware although the same gap leaves a
plain store loop fast. `iadd, store, load, wait` (72.6) is also faster than the model (81.6): the load
overlaps the delayed store. These are fit-set observations; no held-out memory kernels exist yet.

**A device store also waits for the previous `wait` (`storeseries.json`).** `iadd`, k idle `wait`s, device store
(× 8): 35.8 / 35.7 / 36.7 / 43.6 / 63.1 / 82.5 / 101.9 units for k = 0 / 1 / 2 / 4 / 8 / 12 / 16, the same for
`wait 0` and `wait 1`. Up to two waits hide in the 34-unit store quiet time. From four on, each extra wait costs
its 4.9-unit spacing and the store starts 10.2–10.7 units after the last wait issued (the wait after the draining
wait is free, [wait-drain finding](../2026-10-11-wait-drain/README.md)). Modelled as "a device store starts
≥ 10.6 units after the previous wait issued": 14 cases within 3%, where the model had been up to 17% under.
Not explained: a store followed by 8 or 16 `iadd`s costs 44.6 / 52.6 (model 40.1 / 48.5); restarting the quiet
time at the end of the loop's taken branch would explain it but breaks the back-to-back store loop (3.3 → 7.2).

**Several loads in flight (`multiload.json`).** Back-to-back device loads overlap (two: 46.7, three: 52.4
units per repetition). But a load issued after ALU work while an earlier load is still pending does not start
until that load completes, then takes the fast latency: `iadd, load, iadd, load, wait` costs 102.7 (not ~68),
three such pairs 155.9, `load, 4 iadd, load, 4 iadd, wait` 87.9. Loads writing the same register behave the
same as different registers (103.0 vs 102.7). Modelled: 5 cases within 6%.

Interpretation (not proven): the memory pipeline needs a quiet period after ALU issue before it accepts a
request, as if ALU and memory instructions share an issue or operand path that must drain.

This resolves the discrepancy with the [load-latency finding](../2026-10-10-agx-load-latency/README.md): its
pointer chase alternates load and wait with no ALU work between, so it measured the fast case (36 ns at
1,398 MHz, ~50 cycles including 1/16 of the loop); a CPU writer on the buffer (tested) and the declared register
count (16–128 halves, tested) do not matter.

Not established: several SIMD groups, other memory types, misses beyond L1, stores in detail, and whether a
non-ALU instruction (control flow, `wait` on another slot) resets the fast state.

## Reproducer and evidence

`results/2026-10-11-cost-model/load-alu-grid.json` (grid), `memory.json`; reproduce with
`python3 scripts/cost_probes.py loadgrid` (grid), `memstate`, `tgmem` and `memory`. The rule is part of `scripts/agx_cost.py`.

## Compiler implications

* Issue a group's loads back to back: interleaving address arithmetic between loads serializes them
  (each costs a full load latency).
* Batch a group's memory instructions together and issue them before its ALU work; ALU work can then run
  under the loads (up to ~8 instructions for free). A memory instruction right after ALU work starts ~21
  cycles late (loads, threadgroup ops) or ~35 cycles late (device stores); device stores issued back to
  back cost ~3 cycles each.
* Pointer chasing and gather loops pay ~66 cycles per load in practice (address arithmetic precedes each load),
  not the ~50 of a pure chase.
