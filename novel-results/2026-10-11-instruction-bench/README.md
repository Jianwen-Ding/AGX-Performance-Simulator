# Instruction benchmark suite: latency, issue rate and per-core throughput for 60 instruction forms

> **Corrections (2026-10-11, after an independent review).** (1) Cycle figures are in *reference units*:
> the 8-byte `iadd x, x, 1` reference step really takes 2.098 cycles, not 2. Every clock inferred from it is
> one of the GPU's DVFS levels (444, 612, 808, 968, 1,110, 1,236, 1,338, 1,398 MHz, Asahi `t602x-common.dtsi`)
> divided by 1.049, so **real cycles = table value × 1.049**. In real cycles the 4-byte forms come out as exact
> integers (rcp/log2/exp2/sin_pt_2 5, rsqrt 6, floor/sin_pt_1 3, fadd32 x,x,0.125 2). (2) The "≈4 instructions
> per cycle per core" figure equals the bound set by running only 4 SIMD groups per core (4 × 32 lanes ÷
> single-group issue interval); it is a lower bound on core issue, not the core's peak. (3) Earlier claims
> that `.cache` never helps and that FP16 runs at full rate with register operands were wrong (see below).
> (4) Six single-group runs ran at the 968 MHz level, not ~1,333 (ref units 923). (5) The log2 and exp2 chains
> reach NaN and +inf after a few steps, and float/int constants make several ops trivial (x+0, x·1). A
> controlled follow-up (`cost_probes.py values2`, `results/2026-10-11-cost-model/values2.json`) ran identical
> code with only the preloaded constants changed: fmul32 ×1.0 / ×1.0001 1.87 / 1.87, fmadd32 1.99 / 1.99,
> fadd32 +0.0 / +0.5 1.85 / 1.86 units per step, so timing is value-independent for these cases. (An earlier
> probe, `values`, also changed the instruction mix and is not evidence either way; log2/exp2 on NaN/inf vs
> finite values agreed within 0.3 units there.)

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, compute stage.
309 dispatches, no fault or timeout. Only instruction forms that already ran on hardware were used.

## Method

`scripts/run_bench.py` times every instruction inside one dispatch, so GPU clock changes cancel:

* Each SIMD group reads the CPU-written 24 MHz timer ([in-shader clock](../2026-10-11-in-shader-clock/README.md))
  around five counted-loop phases: REF(16), REF(64), OP(16), OP(64), REF(16), after a 2–6 ms warm-up.
* REF is the dependent `iadd x, x, 1` chain, 2 cycles per step ([loop finding](../2026-10-10-agx-loop-alu-timing/README.md)).
  The 64-minus-16 slope removes loop and timer overhead; REF's slope is ns per cycle at that moment.
* **Latency:** one SIMD group, one dependent chain. **Issue interval:** one SIMD group, 8 independent
  chains (cycles per instruction). **Per-core throughput:** 152 SIMD groups (usually 4 per core),
  8 chains each, analysed **per core** (SR20). Each phase is timed from the first of the core's groups
  entering to the last leaving, because groups sharing a busy unit are not scheduled fairly.
* Acceptance gates: single-group runs need the opening and closing REF within 2%; GPU runs need two
  independent dispatches to agree within 5% (worst case 2.7%). All 60 × 3 measurements passed.
* Accepted runs sat on DVFS levels 968, 1,110, 1,338 or 1,398 MHz (inferred 923–1,336 in reference units,
  see the correction above). One exception: the GPU-mode `imadd_cache` run infers 1,360 MHz, between two
  levels, so its cores were measured across a clock change; treat that value as unreliable.

## Claim and scope

**Core issue and lanes.** With 4 SIMD groups per core, forms with at most one 32-bit register source
(`mov_imm`, `iadd x, x, 1`, `fadd32 x, x, 0.125`, `fmadd32 x, x, 0.5, 0.125`, fp16 add/mul, `quad_shuffle`)
reach **the 4-group bound** (4 × 32 lanes ÷ their single-group issue interval, ratio 0.98–1.01): about
4 SIMD instructions per cycle per core, 128 lanes in real cycles, consistent with Apple's 128 FP32 lanes per
core. Whether a core can issue more than 4 per cycle is not established (two attempts at 8 groups per core
failed, see failed approaches). SIMD reductions, `quad_iadd` and `rsqrt` also sit on the 4-group bound, so
their per-core figures are lower bounds. The SIMD shuffles exceed the bound by 12%, which is a measurement
error of the span method for them.

**Register operands limit throughput (a genuine core limit: 0.5 of the 4-group bound).** The same
operations with 32-bit register sources fall to about **2 instructions per cycle per core with two register
sources** (`iadd`, `and`, `xor`, `fadd32`, `fmul32`: 63–65 lanes) and **1.4 with three** (`fmadd32`: 44 lanes),
consistent with about four 32-bit register-operand reads per core per cycle for these forms. The model does
not extend to 16-bit forms: `fadd16` with two 16-bit register sources reaches 123 lanes but `fmadd16` only 59.
The `.cache` source hint is mixed: it raised `fmadd32` from 44 to 63 lanes (issue 2.00 → 1.82), left `imadd`
unchanged, and halved single-group issue of `fadd32`/`fmul32`/`iadd` (1.04 → 2.03) while their core
throughput stayed at the register-read limit.

**Execution classes (latency / issue interval per SIMD group / core throughput):**

| Class | Latency | Issue interval | Lanes / cycle / core |
| --- | --- | --- | --- |
| int add/sub/logic, fp32/fp16 add/mul/fma, compare-select | 2 | 1 (fma32, select: 2) | 64 (register) to 128 (immediate) |
| integer multiply-add, bitfield/shift, popcount/bitrev/ffs, convert, floor/rint | 3 | 2 | 30–36 (one 32-lane unit) |
| 64-bit add, 32×32→64 multiply-add | 3.9–4.8 | 3.9 | 15–17 |
| rcp | 4.8 | 2.9 | 21 |
| log2, exp2, sin_pt_2 | 4.8 | 1.9 | 36–37 |
| rsqrt | 5.7 | 3.8 | 33 |
| simd_shuffle (any form), ballots | 4.8 | 2 | 46–72 |
| quad_shuffle | 3.9 | 1 | 121 |
| quad reduction (`quad_iadd`) | 11.5 | 8.7 | 15 |
| SIMD reductions (`simd_iadd/fadd/min`) / prefix scan | 22–23 / 17 | = latency (no overlap) | 5.6–7.4 |

Full per-instruction table: [table.md](../../results/2026-10-11-instruction-bench/table.md).

`mov_imm` has no source register, so its "latency" column is its issue interval, not a latency.

Not established: throughput beyond 4 SIMD groups per core. Two attempts failed (see failed
approaches), so the reduction and transcendental figures may rise with more groups. Also not
covered: data-dependent timing beyond the cases checked in the correction note, memory instructions (see the
load-latency finding), and fp16 forms with register-pressure effects.

## Reproducer and evidence

`make bench` (hardware, about 30 min), `make verify-bench` (offline, 0 failures).
`scripts/bench_report.py` regenerates the table. [Evidence](../../results/2026-10-11-instruction-bench/README.md).

## Compiler implications

* Fold constants into immediates wherever the encoding allows. A register constant halves FP32
  and integer throughput, and FMA with register operands runs at about a third of peak.
* FP16 add/mul reach the 4-group bound with register operands; fp16 FMA with three register sources does not
  (59 lanes).
* Schedule dependent ALU instructions at least 2 apart (3 for imad, bitfield, convert; 5–6 for
  transcendentals and shuffles).
* SIMD reductions cost about 23 cycles and serialize per SIMD group. Implement `__reduce_*_sync`
  and warp scans so that independent reductions come from different SIMD groups, or use
  shuffle trees (4.8-cycle latency, 2-cycle issue) when several values are reduced at once.
* 64-bit adds cost about 4× a 32-bit add; keep address arithmetic 32-bit where possible.
