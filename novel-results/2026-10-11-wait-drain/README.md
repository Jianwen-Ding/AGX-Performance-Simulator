# `wait` drains ALU work: it completes ~17 units after the latest ALU result, on any slot

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, one SIMD group. Reference units.
Found because held-out kernels with memory instructions were under-predicted by 15% (median), with the
error growing with the number of `wait`s, including in a kernel with no memory instructions at all.

## Claim and scope

| body per repetition (× 8) | units / rep | the ALU instruction's latency + 17 |
| --- | --- | --- |
| `iadd` | 0.9 | |
| `iadd`, `wait 0` | 19.1 | 2.0 + 17 = 19.0 |
| `iadd`, `wait 1` | 19.1 | same: the slot does not matter |
| `iadd`, `wait 0`, `wait 0` | 19.1 | the second wait is free |
| `rcp`, `wait` | 21.9 | 4.8 + 17 = 21.8 |
| `imadd` / `fadd32` / `simd_shuffle`, `wait` | 22.0 / 20.1 / 20.7 | 2.9 / 2.0 / 4.8 + 17 |
| `simd_iadd`, `wait` | 37.2 | 23 + 14 |

* A `wait` whose slot has nothing pending still waits until **~17 units after the latest ALU result of its
  SIMD group completes** (the same ~17-unit figure as the register-file writeback of the
  [result-cache finding](../2026-10-11-result-cache/README.md); possibly the same mechanism).
* Consecutive waits issue at most one per **~4.9 units** (pure `wait` runs, [wait-cost finding](../2026-10-11-wait-cost/README.md));
  a wait issued after a draining wait is free because the spacing has already elapsed.

Model effect: held-out memory kernels (`heldout-mem24.json`, the set that revealed this) went from 15.0% to
6.1% median error; a second, untouched set is reported in the cost-model README.

Not established: several SIMD groups, waits with pending memory work combined with ALU drain, and
whether other synchronising instructions drain the same way.

## Reproducer and evidence

`python3 scripts/cost_probes.py waitdrain`; `results/2026-10-11-cost-model/waitdrain.json`, `waitdrain-slots.json`.

## Compiler implications

* Never place a `wait` right after ALU work unless it is needed: it costs ~19 units even with nothing to wait
  for. Mesa-style "wait just before the use" placement after ALU code pays this; hoist loads so the wait falls
  where the ALU pipeline is already drained, or after the memory result is needed anyway.
