# AGX cycle cost model (`agx_cost.py`), calibrated and validated on hardware

Status: tool + validation on one device (driver-reported M2 Max G14C B1). Predictions are in reference
units (dependent 8-byte iadd step = 2); real cycles = × 1.049 (see the instruction-bench correction).

## What it does

`python3 scripts/agx_cost.py code.bin [--groups G] [--annotate]` decodes an AGX binary with applegpu,
derives the dynamic path with applegpu's emulator (instructions it cannot execute are skipped; they do not
steer the counted loops and structured control flow used by Mesa), and simulates one core running G SIMD
groups cycle by cycle. `--annotate` prints per instruction the execution count, stall cycles and unit class;
taken-branch cost is attributed to the jump. `scripts/agx_capture.py` uses it to annotate shaders captured
from Mesa (see `docs/shader-capture.md`).

Model (every parameter measured on this GPU):

* per class latency, per-group issue interval and per-core throughput from the 60-form instruction bench,
  choosing the operand variant (immediate / `.cache` / register) that matches the instruction;
* per core: ≤ 4 issues per cycle, ≤ 4 32-bit register-operand reads per cycle (16-bit reads free), and per
  execution unit at most the best rate any of its operand variants reached;
* an issue interval blocks the whole SIMD group (the "group" rule; see accuracy below for alternatives);
* the 8-entry per-group result cache with ~17-cycle register-file writeback
  ([finding](../2026-10-11-result-cache/README.md)) and a 1-cycle 32-after-16-bit partial write hazard;
* memory instructions follow the quiet-time rule of the [load-after-ALU finding](../2026-10-11-load-after-alu/README.md)
  (start no earlier than ~20 / ~34 units after the group's last ALU issue), loads are deferred by ALU work issued
  while they are pending, device stores issue in 3.1 units, threadgroup stores in 17.1 (see Memory below;
  `--load-latency` overrides) and `wait` drains a scoreboard slot;
* an idle `wait` costs 4.9 units (the unexplained ~11-unit mode of long wait runs is not modelled;
  [wait finding](../2026-10-11-wait-cost/README.md));
* a taken backward jump costs about 33 extra real cycles, fit so an empty counted-loop trip costs the measured 35.25 units.

## Accuracy

Revised after an independent review (decode, units, method; see "Revision" below). Errors are |prediction −
measurement| / measurement, medians unless stated, from `make verify-cost-model`. "Body only" removes the
fitted empty-loop cost (35.25 units, predicted 35.27) from both sides, which is the fairer figure.

| Dataset | Role | 1 group: whole trip | 1 group: body | 4 groups/core: whole trip | 4 groups/core: body |
| --- | --- | --- | --- | --- | --- |
| 24 random 24-instr kernels | selection (issue rule chosen here) | 5.1% (max 15%) | 7.2% (max 24%) | 6.3% (max 19%) | 9.4% |
| 24 new 24-instr kernels | held out, then used to choose transcendental blocking | 5.7% (max 13%) | 8.8% (max 19%) | 8.3% (max 21%) | 11.3% |
| 12 new 48-instr kernels | held out, then used to choose transcendental blocking | 3.8% (max 8.0%) | 4.5% (max 10%) | 4.3% (max 9.5%) | 5.1% |
| **24 fresh 32-instr kernels** (seeds 9000+) | **held out: measured after every current rule** | **3.7%** (p90 9.6%, max 16%) | | **3.3%** (p90 9.4%, max 13%) | |

Latest verifier run (`results/2026-10-11-cost-model/verify-output.txt`), after the
[exec-control](../2026-10-11-exec-control-cost/README.md), [transcendental pipe](../2026-10-11-transcendental-pipe/README.md)
and store-after-wait rules. The 4-group columns improved most (48 instructions: 9.2% → 4.3%, bias −9% → −4%): the
integer multiply / bit-field / convert class was misclassified as a separate unit, and the simulator's
round-robin arbitration kept SIMD groups in lockstep. The 1-group held-out columns moved by under 0.5 points.

**Kernels with memory instructions** (`cost_hw.py --memory`: ~15% device/threadgroup loads and stores, ~6%
waits; whole-trip medians):

| Dataset | Role | 1 group | 4 groups/core |
| --- | --- | --- | --- |
| 24 kernels, seeds 7000+ | discovery: revealed the [wait drain](../2026-10-11-wait-drain/README.md) (15.0% → 5.5%) | 5.5% | 6.5% |
| **24 kernels, seeds 8000+** | **held out** (measured after the wait-drain rule) | **4.5%** (p90 8.1%, max 12.0%) | **6.0%** (max 20%) |

The held-out memory set first scored 5.6% / 10.1% (max 28.6%); its worst kernel had several loads in flight with
ALU work between them. Separate probes (`multiload.json`) then showed that a load issued after ALU work while
another load is pending waits for it to complete; with that rule (fitted on the probes only) the held-out set
scores as above.

Fitting sets (not evidence of accuracy): the 60 benchmark loops reproduce to 0.4% latency / 0.4% issue /
1.9% per-core throughput; the register rotation used to fit the result cache to 1.7% (32-bit); the memory
probes to 1.5%. The model under-predicts kernels systematically (bias −2% to −5% with one group, −9% to −11%
with four), and the 4-group figures sit near the ±6% repeatability of the 4-group measurement itself.

Issue-rule choice (made on the selection set, before the revision; whole-trip medians):

| Issue-interval rule | random, 1 group | random, 4 groups/core | leader + k independent iadds |
| --- | --- | --- | --- |
| group-blocking (default) | 4.3% | 9.8% | 7.9% (max 42%) |
| shared complex pipe | 8.1% | 14.9% | 3.9% (max 14.5%) |
| per-unit family | 14.8% | 19.0% | 3.9% (max 14.5%) |

**Revision (after review).** Registers now come from applegpu's operand objects (vector loads/stores and
16-bit pairs were previously dropped); atomics are decoded from their fields; every descriptor has an explicit
class (texture, iterator, async, stack, tile, barrier, `stack_adjust`), unknown ones raise; stores hold a
scoreboard slot; parameters are in real cycles and stay fractional (0.1-cycle simulation ticks, because 8-byte
chains really take 2.1 cycles per step); each instruction's issue interval is charged to itself and waits
to the waiting instruction; the immediate-operand variant is used only with at most one register source;
the partial-write rule applies only between 16- and 32-bit writes; the trace flags loops whose condition
depends on uniforms or on instructions the emulator cannot run (their trip count is then unknown).
Classes without hardware data (texture, iterator, threadgroup and stack memory, tile, barriers,
threadgroup atomics) reuse device-memory or 1-cycle costs and are reported as `unmeasured`.

Memory parameters: device loads follow the [load-after-ALU finding](../2026-10-11-load-after-alu/README.md)
(fast 44.8 units with no ALU work since the last wait, ~63 after ALU work, completion deferred by ALU work in
flight; fit on the 30-point `load-alu-grid.json`). That also resolves the earlier gap to the 2026-10-10 chase
(~50 cycles): the chase measured the fast case. A second independent load costs +2.6; device stores cost 3.1 units back to back (the earlier 35 included
the quiet-time delay); memory-space and wait costs follow their findings.

Residual analysis (24 random kernels, single group): the error correlates weakly with 16-bit writes
(+1.7 cycles each unexplained, r = 0.35), SIMD reductions (−2.5, r = −0.50) and transcendentals (−1.0, r = −0.24).
Targeted probes (`half.json`) rule out the obvious causes: mixed-width write-after-write costs ~0.87 per pair
in either direction, reading a fresh 16-bit result as 16 or 32 bits is an ordinary dependency, reductions
cost 23.8 cycles before an independent and 25.5 before a dependent successor (both as modelled), and a
separate 16-bit writeback time (17/22/26 cycles) leaves the random-kernel error unchanged. With 24 kernels
these correlations may be noise.

**Shared transcendental unit (sfumix.json, sfuswitch.json).** At 4 groups per core, independent `log2`,
`floor`, `rcp` contend like four `rcp`s do (148 vs 95 units per trip with one group), so the transcendental
classes share one unit per core. The model draws them from one shared bucket (each instruction costing
1 / its measured rate), and likewise imad/bitfield/convert. This moves that case from 89 to 110 predicted,
but hardware is at 148; `rcp`+`fadd32`+`imadd` (109 measured vs 84–87 predicted) is also unexplained. With
one group there is no function-switch cost (`log2`/`exp2` alternating = 4 × `log2`).

Known gaps: all rules under-predict random kernels (bias −4% with 1 group, −9% with 4), so some
contention is missing; a multi-cycle instruction followed by independent simple ALU instructions overlaps
on hardware more than the default rule allows; bfi/imadd followed by 4–8 iadds costs ~1–2 cycles more than
any rule predicts; 16-bit writeback beyond 9 registers, memory beyond L1, divergence, and more than
4 groups per core are not modelled. Clock changes are not modelled: predictions are cycles, not time.

## Reproducer and evidence

* `make verify-cost-model` (offline, re-scores every stored measurement).
* `make cost-calibrate`, `make cost-validate` (hardware) and `scripts/cost_probes.py lead|rotation|pairs|half|values`.
* Evidence: `results/2026-10-11-cost-model/` (`calibration.json`, `random-kernels.json`, `heldout24.json`,
  `heldout48.json`, `lead-plus-k.json`, `rotation.json`, `rotation16-extended.json`, `pairs.json`,
  `fetch-body-size.json`, `values.json`, `values2.json`, `memory.json`, `load-alu-grid.json`, `memstate.json`,
  `tgmem.json`, `spaces.json`, `waits.json`, `sfumix.json`, `sfuswitch.json`, `groups.json`, `half.json`).
