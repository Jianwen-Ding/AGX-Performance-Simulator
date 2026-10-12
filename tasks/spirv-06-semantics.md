# Task SPIR-V 6: semantics (atomics, geometry, spills, control flow, shuffles, exact float operations)

**Workloads:** `atomics`, `geom3d`, `spill`, `control`, `shuffle`, `floatops`. The procedure is in
`tasks/spirv-workloads.md`.

## Why these

These six check **behaviour**, not throughput. Every one is exact, so any difference between the M2, the simulator
and the reference is a finding. Where the language leaves a result implementation-defined, the check allows both
answers: the sign of a zero from min/max/sign/clamp, and `round` on exact .5 ties.

| Workload | What it pins down |
|---|---|
| `atomics` | Returned old values of all eight device atomic ops on distinct words; final values on contended words; the contended returns form one chain (init → final); a threadgroup `atomicAdd` counter returns a permutation of 0..255; threadgroup `atomicMax`. Mesa aggregates the threadgroup atomics per SIMD group: the counter becomes `icmp_ballot` + `popcount` + one `threadgroup_atomic` with the lane's rank added (`ffs`, `simd_shuffle`), and the max becomes `simd_max.u` + one `threadgroup_atomic` |
| `geom3d` | `gl_GlobalInvocationID`, `gl_LocalInvocationID`, `gl_LocalInvocationIndex`, `gl_WorkGroupID`, `gl_NumWorkGroups`, `gl_WorkGroupSize` for 4×4×2 workgroups on a ragged 3-D grid |
| `spill` | Mesa spills a 96-word private array to the stack (`stack_store` / `stack_load`, scratch size 108 bytes in the 26.0.3 binary) |
| `control` | A switch (Mesa 26.0.3 turns it into a 22-instruction `icmpsel` select chain, no branches), loops with `break` / `continue` (`while_icmp`, `jmp_exec_any`, one `update_exec`), lanes returning early, a function inlined in a loop |
| `shuffle` | 37 `simd_shuffle` (the shuffle-relative and xor forms all lower to it, with computed lanes), xor butterflies, up / down with edge lanes, broadcast; the shader writes the subgroup size and the reference follows it (32 on AGX, 8 on lavapipe) |
| `floatops` | min, max, clamp, abs, sign, floor, ceil, fract, round, roundEven, trunc, step, negation, all six compares, add with ±inf, on ±0, ±inf, ties, the largest finite and smallest normal values |

## Already verified offline

- **lavapipe:** all six pass at both sizes.
- **Real driver code (Mesa 25.2.8 and 26.0.3, drm-shim):** all six decode and pass on the functional simulator: 12 of 12.
  - `spill` found a simulator gap: `stack_store` / `stack_load` were not implemented. They are now: per-lane private
    memory. The stack setup store has mask 0 and the scratch size as its immediate. Loads and stores address it
    consistently, which is what spills need. **The hardware's exact addressing (the scale of the offset) is not
    established by this.** The M2 run confirms the values, not the layout.
- **Cost model:** `predict` gives a whole-dispatch time for all six (large).

## Checks on the M2

1. **Hardware vs reference**, both sizes: all exact (the check's own allowances for zero signs and ties aside).
2. **Simulator vs hardware, word for word.** Zero differing words expected for all six. Zero signs and rounding ties
   must also match: the reference allows either answer, but the simulator should give the hardware's one.
3. **Record:**
   - `atomics`: the order the contended returns came in (one run). Is it the same between runs, between workgroups?
   - `floatops`: which zero `min(-0, +0)` / `max(+0, -0)` / `sign(-0)` give, and how `round(2.5)` / `round(-2.5)` round.
     Make the simulator match if it differs.
   - `spill`: if possible, read the scratch memory after a run (driver debug) to learn the real stack layout.
4. **Performance:** `predict` vs measured for `spill` (stack traffic), `control` (divergence) and `atomics`
   (contention on 8 words).

## Deliverable

`results/<date>-spirv-semantics/README.md`, plus the per-workload artefacts listed in the shared procedure.
