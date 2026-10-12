# Task SPIR-V 3: atomics and sorting, histogram and bitonic sort

**Workloads:** `histogram` (16 MiB skewed bytes; a geometric distribution makes a few bins hot), `bitonic` (1 Mi keys, 66 dispatches, large; 4 Ki keys, 6 dispatches, small). The procedure is in `tasks/spirv-workloads.md`.

## Why these

These are the irregular kernels:

- histogram's cost is dominated by contention: threadgroup atomics on hot bins, then 256 device atomics per workgroup;
- bitonic sort has data-dependent swaps under `if`, barriers inside nested loops, and a long chain of small dispatches, so per-dispatch launch overhead shows up directly.

## Checks

1. **Hardware vs reference.** Both are exact:
   - histogram: every bin;
   - bitonic: sorted keys, and payload consistent with the keys.
2. **Simulator vs hardware.**
   - The final contents must match word for word. Atomics commute, and keys are unique.
   - Check that `sim` runs the `threadgroup_atomic` / `atomic` forms the compiler emits. The simulator implements the Mesa-named ops 0..10; a new op or form is a finding.
3. **Compiled code.**
   - Which atomic forms appear: threadgroup vs device, with or without return?
   - How is the `if ((ka > kb) == up)` swap compiled: exec-mask `if_icmp` / `pop_exec`, or `icmpsel` selects (branch-free)?
   - The answer decides which cost-model path applies.
4. **Performance.**
   - **histogram:** report GB/s. Then rerun with a uniform byte distribution: change `rng.geometric(0.02, ...)` to `rng.integers(0, 256, ...)` in a copy of the workload named `histogram_uniform`. The difference is the contention cost; compare it with `novel-results/2026-10-11-shared-memory-sync`.
   - **bitonic:** report the per-dispatch times (`dispatch_ns_median`). Separate the global passes (mode 1) from the shared-memory passes (modes 0 and 2). Compare the small global passes with the launch overhead from `novel-results/2026-10-11-dispatch-launch`, and run `predict` per dispatch.

## Deliverable

`results/<date>-spirv-histogram-bitonic/README.md`, plus the artefacts.
