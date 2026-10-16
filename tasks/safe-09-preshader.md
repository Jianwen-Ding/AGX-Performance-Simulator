# Task safe-09: the preshader (preamble) and uniform_store on direct DRM

Read [safe-probes.md](safe-probes.md) first (safety rules).

**Goal:** finish the preamble execution model. A CUDA backend can use the preamble to compute kernel-uniform values
(argument pointers, `blockDim` products, constants) once per core instead of in every thread.

**Background** (playground `novel-results/2026-10-11-preshader-execution`, `2026-10-11-sr17-sr32`):
- `tests/drm_compute.c` has `AGX_PRESHADER=1` (Preshader word, preamble at usc_words + 0x200) and
  `AGX_PRESHADER_CODE`.
- The preamble runs once per core per dispatch, on one lane, in SIMD-group slot 96, with SR24 = 5.
- The uniforms it writes are per core.
- The simulator has `uniform_store` and `func::run_preamble` (`rust/src/func.rs:555-605`).
- The finding left these open:
  - several dispatches in one command;
  - re-running the preamble within a dispatch;
  - preamble register use;
  - the "preshader-last" rule.

## Safety

- Preamble code must use only `uniform_store`, `get_sr`, ALU and device-atomic forms already run in the playground's
  `make sr-direct` preamble configurations, and must end with `stop`.
- **No barriers in the preamble.** `run_preamble` rejects them, and they were never run there.
- Run every preamble through `func::run_preamble` before the GPU.
- Several dispatches in one command depends on [safe-04](safe-04-driver-submission.md) §4 having passed for N = 2
  with the plain control kernel.

## Steps

1. **Coverage case** (`MANUAL` item `preshader`): a preamble writes 4 uniforms with `uniform_store`, and the main
   shader stores them per lane together with SR20. Expected: every lane on a core sees that core's values. Record
   it and check it in `suite_check`.
2. **Uniform file size.**
   - `uniform_store` to u0 .. u255, in steps, from the preamble. applegpu documents 256 uniforms (playground
     `docs/external-knowledge.md`). Stay within the encodable range (`isa/table.rs:465`).
   - Read every uniform back in the main shader.
   - Find which uniforms survive and whether high uniforms alias low ones.
   - The simulator errors beyond the uniform file today. Fix the bound to the measured one.
3. **Preamble register budget.**
   - Use preambles with declared register counts from 8 to 248 (the same field as the main shader's `gprs`).
   - Does the preamble's count change main-shader occupancy (the [occupancy](safe-11-occupancy-and-clock.md) counter)
     or SR19?
4. **Several dispatches per command** (after safe-04 §4):
   - Does the preamble run once per dispatch or once per command?
   - Do uniforms written by dispatch 1's preamble survive into dispatch 2 on the same core?
5. **Cost:** the firmware time of a dispatch with a 1-instruction preamble against no preamble, and against a
   preamble of 64 / 256 ALU instructions (2048 workgroups, fast-clock cluster).

## Deliverables

- A coverage case.
- Simulator fixes (uniform bound, preamble cost).
- An update to the playground finding.
- A `make verify-preshader` target. Push both repositories.
