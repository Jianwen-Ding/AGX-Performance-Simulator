# Task SPIR-V 1: reduction and prefix scan

**Workloads:** `reduce`, `scan` (`spirv/shaders/reduce.comp`, `scan.comp`). The procedure is in `tasks/spirv-workloads.md`.

## Why these

They are the building blocks of most GPU algorithms, and they exercise exactly the cross-lane machinery the simulator models from the hardware findings:

- `subgroupAdd` → `simd_iadd` / `simd_fadd`, with pairwise-tree float order;
- `subgroupExclusiveAdd` → `simd_prefix_iadd` (exclusive);
- shared-memory partials across 8 SIMD groups, with barriers;
- a multi-dispatch structure: 2 dispatches for reduce, 3 for scan.

## Checks

1. **Hardware vs reference.**
   - The uint results must be exact.
   - The float partials stay within 1e-6 of the float64 sum, relative to Σ|x|.
   - Keep the hardware's float partials (out2.bin); a close match with the simulator is enough, bit-level agreement is not a goal.
2. **Simulator vs hardware.** `sim scan` and reduce's uint half must match word for word; reduce's float partials only need to be close (within the reference tolerance).
3. **Compiled code.** In `code/main.txt`:
   - find the instructions `subgroupAdd` and `subgroupExclusiveAdd` became;
   - find how `gl_SubgroupID` and `gl_NumSubgroups` are computed (special registers);
   - note whether the shared-memory partials use `threadgroup_store` with an immediate or register index.
4. **Performance.**
   - Report GB/s for `reduce` (large: 16 Mi floats + 16 Mi uints, 128 MiB read) and `scan` (1 Mi uints, about 16 MiB of traffic over the three passes), as fractions of the memory bandwidth measured in `results/2026-10-11-memory-hierarchy`.
   - Compare each dispatch's time with `predict`.
   - In scan, the single-workgroup middle pass is pure latency. Its time against `predict` is a direct test of the model's latency path.

## Deliverable

`results/<date>-spirv-reduce-scan/README.md` with these items, plus the per-workload artefacts listed in the shared procedure.
