# Task SPIR-V 5: ISA coverage through the compiler

**Workloads:** `fp16`, `int64`, `bitops`, `intdiv`, `compact`, `spmv`, `transpose`, `mandelbrot`, `softmax`, `rgba8`.
The procedure is in `tasks/spirv-workloads.md`.

## Why these

The first eight workloads are about throughput. These ten are about **coverage**: each one makes Mesa's AGX compiler
emit instruction forms and lowerings that the other workloads, and the hand-written test suite, do not reach.

What Mesa 26.0.3 actually emits for each (`spirv/compiled/mesa-26.0.3/<w>/main.txt`):

| Workload | Notable instructions in the compiled main program |
|---|---|
| `fp16` | `fmadd16` (both fp16 expressions fused), `fmadd32` for the fp32 expression, 16-bit device loads and stores |
| `int64` | 64-bit multiply as 9 `imadd` with `bfeil` / `extr` / `asr` / `icmpsel` for the halves and the signed shift |
| `bitops` | `bfi`, `bfeil`, `ffs`, `bitrev`, `popcount`, `bitop`, `imadd` (wide products); the width modulo (`% (33 - off)`) as an `rcp` / `convert` / `fmadd32` division |
| `intdiv` | Division through a float reciprocal: `rcp`, `convert`, `fmadd32`, `fmul32`, then `imadd` / `isub` / `icmpsel` correction; `iadd.sat` |
| `compact` | `icmp_ballot`, `popcount`, `ffs`, `simd_shuffle` (subgroupBroadcastFirst), `else_fcmp`, `while_icmp` / `jmp_exec_any` loops |
| `spmv` | `while_icmp` loop with a memory-dependent bound, dependent `device_load` gathers |
| `transpose` | 2-D thread ids, 4 `threadgroup_store` / 4 `threadgroup_load`, 8 bounds tests (`if_icmp` / `pop_exec`) |
| `mandelbrot` | Divergent `while_icmp` / `push_exec` / `jmp_exec_any` loop, `imadd` squares, `asr` fixed-point shifts |
| `softmax` | `exp2`, `rcp`, `fcmpsel` (max), 19 `threadgroup_barrier`s in unrolled trees |
| `rgba8` | `convert` (u8norm), `fmul32.sat` + `rint` (packUnorm4x8), `bfi` / `bfeil` packing, a byte `device_store` |

The integer workloads are exact by construction (`mandelbrot` uses fixed point), so any difference between the
hardware, the simulator and the reference is a finding.

## Offline sanity checks (not hardware verification; `make spirv-verify` shows these as PENDING until recorded)

- **lavapipe:** all ten pass their references at both sizes.
- **Real driver code:** compiled by Mesa 25.2.8 and 26.0.3 through the drm-shim (`spirv/compiled/mesa-*/<workload>/`).
  All ten decode completely (main and preamble), run on the functional simulator, and pass their references: 20 of 20.
- **Cost model:** `predict` gives a whole-dispatch time for every one of them (large size).

## Checks on the M2

1. **Hardware vs reference.** `run` at both sizes. Every integer workload must be exact. `fp16`, `spmv`, `softmax` and
   `rgba8` must be within the bounds their checks print (`max_err_over_bound` ≤ 1, relative error < 1e-5, pack error
   ≤ 0.5).
2. **Simulator vs hardware, word for word.** `sim <w> --code-dir ... --hardware spirv/out/<w>-small`.
   - Integer workloads: zero differing words.
   - `fp16`: the simulator rounds exactly like the hardware's fp16 ALU, so differences here are simulator bugs.
   - `softmax`: differences come only from fp32 `exp2` (task #5's open transcendental gap); count them.
3. **Compiled code.** From `code/main.txt`, record for each workload:
   - `int64`: how many instructions one 64 × 64-bit multiply takes;
   - `intdiv`: the instruction sequence for `a / b` with a variable `b`, and for `a / 7`;
   - `compact`: which ballot form `subgroupBallot` became, and how the bit count is taken;
   - `mandelbrot`: the exec-mask instructions around the loop exit.
   - Compare each listing with the table above (from the shimmed 26.0.3 build): the M2's own Mesa may emit different
     sequences, and the cost model should be re-run on the M2's binaries, not only the committed ones.

   These are the instruction mixes the cost model sees; note any instruction it reports as unknown.
4. **Performance.** Measured `dispatch_ns_median` vs `predict` (large). The interesting ones:
   - `intdiv` and `int64`: long integer-ALU dependency chains;
   - `compact`: pass 1 is a single SIMD group walking 16 Ki counts, pure latency;
   - `mandelbrot`: divergence; compare the mean iteration count with the time;
   - `spmv` and `transpose`: memory-bound, irregular and strided.

## Deliverable

`results/<date>-spirv-isa-coverage/README.md`, plus the per-workload artefacts listed in the shared procedure.
