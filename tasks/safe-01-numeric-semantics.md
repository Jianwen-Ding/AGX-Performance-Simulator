# Task safe-01: exact numeric semantics (conversions, fp16 transcendentals, subnormal edges)

Read [safe-probes.md](safe-probes.md) first (safety rules).

**Goal:** bit-exact models of the remaining numeric instructions ZLUDA must reproduce, recorded exhaustively the way
fp32 `exp2` / `sin_pt_2` were (`results/2026-10-16-transcendentals`). Each model goes into the functional simulator
and is checked offline.

**Tools:**
- `scripts/record_trans_ranges.py OP LO,...` records all 2^23 mantissas of a binade (~4 s each, every chunk run
  twice).
- Extend it with an `--text` option taking any safe one-source form, e.g. `convert u32_to_f, r14, r6, rte` with
  `mode=`/`round=` fields as in `coverage_catalog.py`.

## 1. `convert` (PTX `cvt`)

- **Scope:** modes 0, 1 and 4-11, both directions (round field 0 / 1, i.e. bit 26), the rounding operands Mesa
  emits (`rte`, `rtz`), and 32-bit and 16-bit sources and destinations.
- These exact forms are recorded `safe` in `coverage/expected/alu/convert_*` (80 files), but only on 1,024 random
  operands each.
- **Inputs:**
  - Every fp32 source: 256 binades x 2 signs, plus zeros, denormals, inf and NaN payloads.
  - Every 32-bit integer source in 2^23-word chunks: 512 chunks, ~35 min per form.
  - Every 16-bit source: 65,536 inputs.
- **Questions:**
  - Rounding at every tie.
  - Saturation versus wrap at overflow.
  - NaN -> integer.
  - -0.0.
  - Denormal inputs (flushed or not, per direction).
  - f32 -> f16 overflow, underflow and denormal results.
  - Do the integer -> float modes round per the operand, or always to nearest?
- **Compare with PTX `cvt` rules** (PTX ISA 9.7.9.21):
  - `.rni` / `.rzi` / `.rmi` / `.rpi`, and `.sat` behaviour.
  - NaN -> 0 for float -> integer.
  - List every case where a single AGX `convert` does not implement a PTX form, and the extra instructions needed.
- **Deliverables:**
  - An exact `fsem::convert` model.
  - `make verify-conversions` (retained samples, replayed in `agx_run`).
  - Playground `novel-results/<date>-conversions` with a PTX mapping table.

## 2. fp16 transcendentals

- **Scope:** `rcp`, `rsqrt`, `rsqrt_special`, `log2`, `exp2`, `sin_pt_1`, `sin_pt_2` with a 16-bit source and a 16-bit
  destination (forms such as `exp2 r14l, r6h` are in `coverage/expected/alu`).
- All 65,536 inputs each, in a single 2^16-lane dispatch run twice.
- The simulator assumes these are correctly rounded from samples only. Confirm or replace that with a table, and
  compare with PTX `ex2.approx.f16` / `.f16x2` bounds.

## 3. Subnormal edges of fp32 arithmetic

- **Scope:** `fadd32`, `fmul32`, `fmadd32`, `rcp`, `rsqrt`, `log2`, `exp2` on inputs and outputs at the
  normal/subnormal boundary (exponents 0-2 and results near 2^-126), with every `.abs` / `.neg` source modifier
  already recorded `safe`.
- **Question:** which operations flush inputs, outputs or both, and is the sign of a flushed result kept?
- **Deliverable:** for every PTX float instruction, whether its default (non-`.ftz`) form maps directly or needs a
  scaling fix-up, and the fix-up sequence. Verify each fix-up on the GPU on the boundary inputs.

## Acceptance

- `make test`, `make suite-check`, `make coverage-check` and `make verify-transcendentals` are unchanged or better.
- New verify targets pass in both repositories, and everything is pushed.
