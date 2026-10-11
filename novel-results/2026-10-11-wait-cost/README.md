# `wait` costs ~5 units even when nothing is pending (long runs of waits: an unexplained ~11-unit mode)

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, one SIMD group, no memory
instructions in flight. Reference units (× 1.049 = real cycles).

## Claim and scope

* An idle `wait` (nothing pending on its slot) costs about **4.9 units (5.1 cycles)**, the same for slots 0
  and 1, from 1 to 30 per repetition (`waits.json`).
* Some long runs of consecutive waits cost about **11 units per wait**:

| body per repetition (× 8 per loop trip) | units / rep | loop body bytes |
| --- | --- | --- |
| 30 waits | 146.3 | 500 |
| 30 waits + iadd | 154.7 | 564 |
| 31 waits | 351.5 | 516 |
| 31 waits + iadd | 358.9 | 580 |
| 16 waits, iadd, 16 waits | 163.5 | 596 |
| 8 waits + 6 iadd | 46.1 | 532 |

  It is not a loop-body-size limit (564–596 B bodies are fast), not the total wait count, and not simply the
  run length: in `16 waits, iadd, 16 waits` the repetitions concatenate into runs of 32 consecutive waits and
  it is still fast, while runs of 31 (`31 waits + iadd`) are slow. The trigger is unknown (alignment of the
  run within the instruction stream is a candidate). An earlier version of this finding attributed the step to
  a 512-byte loop buffer; `waitruns.json` and `review2.json` refute that. The cost model charges 4.9 for every
  idle wait and does not model this mode.

Not established: the trigger of the ~11-unit mode, straight-line code, waits with pending work, and other
synchronisation instructions.

## Reproducer and evidence

`python3 scripts/cost_probes.py waits` and `review2`; `results/2026-10-11-cost-model/waits.json`,
`waitruns.json`, `review2.json`. The 4.9-unit idle cost is modelled in `scripts/agx_cost.py`.

## Compiler implications

Do not emit `wait` defensively: each costs ~5 cycles even with nothing to wait for (Mesa's
`AGX_MESA_DEBUG=wait` mode would be expensive). The long-run effect does not arise in compiled code.
