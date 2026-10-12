# Task SPIR-V 2: dense float, SGEMM and 5×5 convolution

**Workloads:** `sgemm` (1024³ large, 128³ small), `conv2d` (2048² large, 64×48 small). The procedure is in `tasks/spirv-workloads.md`.

## Why these

These are the FLOP-bound kernels. SGEMM is the standard measure of how close the ALU pipe gets to peak. It keeps a 4×4 accumulator block in registers, so register count, occupancy and the FMA issue rate all matter. conv2d is the same pattern with a smaller arithmetic intensity, and adds clamped 2-D addressing.

## Checks

1. **Hardware vs reference.**
   - SGEMM must stay within the forward-error bound (K+2)·2⁻²⁴·(|α||A||B| + |β||C|). `max_err_over_bound` ≤ 1; lavapipe gives 0.004.
   - conv2d: relative error < 1e-6.
2. **Simulator vs hardware.** Both should be bit-identical. They are fmadd32 / fmul32 / fadd32 only, and the simulator's fp32 arithmetic is exact. Any differing word is a simulator bug or an unmodelled instruction: find it.
3. **Compiled code.**
   - Count the `fmadd32` in the inner loop: 16 per k for 4×4 blocking.
   - Note the threadgroup load forms. Are 4 consecutive floats fetched as one xyzw load?
   - Record the GPR count. `props.json` statistics, or the binary's register field, as `predict --regs` needs it.
   - Record whether the loop is unrolled.
4. **Performance.**
   - Report GFLOP/s. Peak: 38 cores × 4 SIMD × 32 lanes × 2 FLOP × clock, at the clock from `results/2026-10-11-instruction-bench`.
   - Compare with `predict`. SGEMM is the cost model's hardest test, because it mixes threadgroup loads, barriers and long FMA chains under occupancy limits.
   - If the prediction is off by more than 20%, break the time down: run `agx_cost --annotate` on `code/main.bin` for one SIMD group, and compare it with a variant compiled at `-DK_TILE=8` (add a define) to separate shared-memory cost from FMA cost.

## Deliverable

`results/<date>-spirv-sgemm-conv2d/README.md`, plus the artefacts.
