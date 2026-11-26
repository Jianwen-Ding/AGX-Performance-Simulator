# Task safe-13: the assumptions of the .air -> AGX pathway, on the GPU

Read [safe-probes.md](safe-probes.md) first (safety rules).

**Goal:** confirm or refute, on the M2 Max, what the `.air` / `.metallib` -> AGX pathway (`llvm-air2agx`, LLVM branch
`claude/focused-franklin-n82lij`) assumes and what its optimizations were measured with. **Every number behind them
so far comes from this repository's model** (agx_cost cycles, agx_run results), from MSL's and LLVM's semantics, and
from alive2 where it applies; nothing has run on hardware. LLVM's `llvm/utils/agx-bench/EXPERIMENTS.md` ("Experiment:
.air / .metallib -> AGX" and its follow-up) has the model results and the A/B switch of each change.

**Background.** `llvm-air2agx` reads the AIR the AIR target writes for Metal (or Xcode's), runs `agx-lower-air` and
`-O3`, and compiles for AGX. The follow-up of 2026-11-26 changed how that AIR is compiled:

| # | Change (LLVM commit) | What it assumes about the GPU | Model result (G14G) | Switch |
|---|---|---|---|---|
| H1 | MSL device-scope atomics are `syncscope("device")`; an atomic narrower than the system is not a host-coherent access (729c4a5da) | a device atomic (compare-and-swap that never changes memory for a load, exchange for a store) is coherent across cores without `f59d` / `f533` | atomic load + store 209.0 -> 117.9 cycles | revert |
| H2 | a volatile, aligned 32-bit device access is a volatile device-scope monotonic atomic (f76ee3467) | as H1, and that nothing needs a volatile device access host-coherent during a dispatch | atomic_fmax_f32 621.6 -> 151.2; a volatile load / store loop 4068.7 -> 1981.5 | revert |
| H3 | a threadgroup-only barrier with no threadgroup access since the last barrier, or none before the next, is dropped (1e36687fa, c9429a963) | the GPU needs no barrier for threadgroup memory beyond the ones between accesses (MSL's rule) | tg_tree_reduce 2555.6 -> 1680.6; mma_f32_threadgroup 568.1 -> 401.0; tg_reverse_barrier2 405.9 -> 280.9 | `-agx-air-idle-barriers=false` |
| H4 | a CAS loop of a native operation is that atomic (AGXCmpXchgLoops, 729c4a5da): G14's float add | the native float add (device atomic sub with bit 45) rounds and flushes as the loop's `fadd` | an MSL float-add CAS loop 151.2 -> 122.3 | `-agx-cmpxchg-loops=false` |
| H5 | the Metal pipeline's lock-emulated 64-bit atomics are `atomicrmw i64` again (38e77fbde) | the backend's own lock (AGXExpandWideAtomics) and G14's u64 min / max are atomic and finish under contention | REJECTED (agx_run step limit: the emulation's critical section left its loop) -> PASS on G13G and G14G; not costed (agx_cost cannot trace lock loops) | revert |

Model gaps these rest on (EXPERIMENTS.md "Model gaps"): agx_cost charges nothing for uniform registers, and a buffer
location with no parameter is an unused pointer argument (the lock table at location 30 adds 28, 112 uniform halves);
the cost of `memory_barrier` sequences and of device atomics comes from the playground's fence and atomic
measurements, not from these kernels.

## Safety

- Only code the LLVM backend already emits and the coverage suite records `safe` (device atomics cmpxchg / xchg /
  add / or, threadgroup barriers, the seven Mesa barriers); compile with `llvm-air2agx` and gate every launch on
  `agx_run` (rule 4).
- **Spin loops:** H5's kernels and the backend's wide atomics take a lock in a loop. Run them only after `agx_run`
  finishes them for the exact grid, one small dispatch first (one workgroup of 32 threads, then 64 workgroups), and
  stop at the first timeout.
- **Do not run** a kernel with a spin lock written in its own code (see step 7): `-O3` may move its critical section
  out of the lock's loop, which deadlocks a SIMD-group and wedges the queue.

## Steps

1. **H1 / H2, coherence.** Message passing across cores (the playground's `run_drm_litmus.py` style, no spin-waits):
   lane A stores data, then a flag with a device-scope atomic exchange; lane B on another core (SR20) reads the flag
   with a device-scope atomic load (the compare-and-swap form) a fixed number of times, then the data with the same
   load. Count violations without and with `f533` / `f59d`. The pathway emits no barrier around a monotonic
   device-scope atomic: zero violations of the flag's own value are required; data ordering is not promised (relaxed).
   Also: is a value written by a device atomic on one core seen by a later device atomic load on another without a
   barrier, within a bounded number of reads?
2. **H1 / H2, cost.** Time `llvm-air2agx -mcpu=agx-g14c` on `atomic_load_store_i32`, `volatile_load_store`,
   `atomic_fmax_f32` (air-gpu suite) and a volatile load / store loop, at the commit before and after, at 1000
   back-to-back commands. Does the measured ratio follow the model's (209 -> 118, 622 -> 151)?
3. **H3, barriers.** Run `tg_tree_reduce`, `tg_reverse_barrier2`, `tg_reverse_auto_barrier`, `mma_f32_threadgroup`,
   `tg_atomic_add`, `tg_v4f32_lanes` with and without `-agx-air-idle-barriers`: results bit-exact against the
   suite's references, then time. Open question the model cannot answer: does a workgroup that starts on a core while
   another workgroup's threadgroup memory there is still being written ever see those writes (the dropped leading
   barrier would not have helped, but record it)?
4. **H4, float add.** Contended and uncontended float-add CAS loops against the native atomic on G14C: results
   bit-exact (including subnormal operands, which both flush), and time.
5. **H5, 64-bit atomics.** `atomic_add_i64`, `atomic_umax_u64`, `atomic_umin_u64` (suite) on G14C under the gates
   above: results against the references; time against 32-bit atomics; with all 32 lanes on one address and on 8.
6. **Uniforms.** Launch time of a kernel whose record lists 2 arguments against the same kernel with 28 unused
   pointer arguments before a used one (location 30): does a larger uniform count cost launch time or occupancy?
   (EXPERIMENTS.md model gap 5.) If it does, the pathway should compact unused locations.
7. **Spin locks (offline only).** Write the MSL-style lock (`if (cas(lock, 0, 1)) { work; xchg(lock, 0); }` in a
   retry loop) as AIR, compile it with `llvm-air2agx`, and check the assembly: is the critical section still inside
   the loop? Run it in `agx_run` only. Record the kernels that the optimizer breaks; this decides whether the AGX
   backend needs a convergence guard for such loops.
8. **End to end.** The air-gpu suite through `llvm-air2agx` (`air_gpu_suite.py --agx-run --air2agx` gives the
   simulator's results: PASS 690, HW-FTZ 35, MISMATCH 1 on 2026-11-26) on the GPU with the hardware launcher
   (`agx_run_hw`, as `scripts/grid_sweep.py` uses it): every output bit-exact against the simulator's.

## Deliverables

- `results/<date>-air2agx-hw/README.md`: per step, the measurement, the model's number, and the verdict for H1-H5.
- Each refuted assumption: the LLVM switch turned off or the commit reverted, with the measurement in
  `llvm/utils/agx-bench/EXPERIMENTS.md`.
- Step 6: whether unused buffer locations should be compacted; step 7: the list of spin-lock shapes the optimizer
  breaks. Push both repositories.
