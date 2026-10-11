# Per-SIMD-group result cache: the last 8 written registers are fast; older values wait for a ~17-cycle writeback

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, compute stage, one SIMD group.
Known-safe encodings only; cycles from the in-dispatch clock (reference units, × 1.049 for real cycles).

## Claim and scope

72 independent instructions per loop trip, rotating over D destination registers (each instruction reads
the value its register received D instructions earlier):

| D (distinct destinations) | 4–8 | 9 | 10 | 12 | 16 |
| --- | --- | --- | --- | --- | --- |
| `iadd x, x, 1` cycles / instruction | 1.03 | 1.90 | 1.69 | 1.27 | 1.13 |
| `fadd32 x, x, 0.125` | 0.97 | 1.86 | 1.63 | 1.21 | 1.07 |
| `fadd16 x, x, c` (16-bit) | 1.02 | 1.89 | 1.68 | 1.64 | 1.62 |

* A value read within **8 writes** of being produced is available at the normal 2-cycle latency.
* A value read later is available only about **17 cycles after its producer issued**: solving
  stall = W / D − 1 gives W ≈ 17, 17, 15 and 18 cycles for D = 9, 10, 12 and 16 (32-bit forms).
* Interpretation: each SIMD group keeps its last 8 results in a small result/operand cache; a result
  evicted from it is readable from the register file only after a ~17-cycle writeback. 16-bit forms keep
  paying ~0.6 cycles per instruction up to D = 16, so their writeback is slower or merges halves.
* It is not a loop-buffer or fetch limit: the same 72-instruction (576 B) body with D ≤ 8 runs at full rate,
  and bodies up to 2 KiB of 8-byte instructions run at one instruction per cycle
  (`results/2026-10-11-cost-model/fetch-body-size.json`).
* Related small effects measured alongside (single group): a 32-bit write right after a 16-bit write of
  the same register costs about 0.9 cycles; writing the low and high halves independently costs nothing;
  a SIMD shuffle or reduction fed by an ALU result costs 0.7–1.8 cycles more than shuffle→shuffle chains.

16-bit forms recover only slowly beyond D = 16 (`rotation16-extended.json`: 1.49 at 20, 1.35 at 24, 1.31 at 32,
1.28 at 38 units per instruction; beyond 19 destinations the rotation mixes low and high halves). No single
writeback time fits (W = D·(1+stall) rises from ~20 to ~49), so the 16-bit rule remains open.

Not established: behaviour with several SIMD groups per core (whether the cache is per group or per
scheduler), whether loads and multi-cycle units write through the same path, and the exact 16-bit rule.

## Reproducer and evidence

`python3 scripts/cost_probes.py rotation` (also `lead`, `pairs`, `half`); evidence in
`results/2026-10-11-cost-model/` (`rotation.json`, `lead-plus-k.json`, `pairs.json`, `fetch-body-size.json`).
The rule is part of `scripts/agx_cost.py` and checked by `scripts/verify_cost_model.py`.

## Relationship to existing knowledge

Mesa sets per-operand `cache`/`discard` hints (`agx_pack.c`), which implies an operand cache, but documents
neither its size nor the cost of missing it. The 8-entry capacity and ~17-cycle writeback are measured here.

## Compiler implications

* Keep a value's consumers within 8 register writes of its producer, or more than ~17 cycles away.
  Reading a value 9–16 writes after it was produced costs up to ~0.9 extra cycles per instruction.
* Unrolled loops that rotate over 9–12 accumulators are the worst case; 8 or 16+ accumulators are fine.
* Prefer 32-bit over 16-bit temporaries when a value lives longer than 8 writes.
