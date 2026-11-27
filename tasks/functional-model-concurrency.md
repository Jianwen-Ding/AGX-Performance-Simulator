# Functional model: concurrency across workgroups and out-of-range threadgroup memory

Two places where ZLUDA's CUDA-on-AGX path (ZLUDA `zluda_metal`, AGX backend) differs from CUDA because of how
the functional model (`rust/src/func.rs`) runs a dispatch. Both need answers from a real M2 before any change:
**do not change the model for either without hardware measurements** (questions 5 and 6 of
[m2-run-checklist.md](m2-run-checklist.md)).

## 1. Workgroups run one after another; SIMD groups until their next barrier

`func::run` executes each workgroup to completion before the next, and inside a workgroup runs each SIMD group
until it reaches a barrier. That is exact for programs whose workgroups communicate only through barriers
and through results read after the dispatch, which is everything the suite and the PTX-formalization
conformance corpus contain. It is not exact for programs that wait on another workgroup, or on another SIMD
group of the same workgroup without a barrier between:

- a grid-wide barrier (CUDA `cooperative_groups::this_grid().sync()`, `cuLaunchCooperativeKernel` grids of more
  than one block);
- a workgroup spinning on a flag that a later workgroup sets (decoupled look-back scans, producer / consumer
  queues; the suite's `lookback` completes in the model, its predicted time too low for want of a flag chain,
  checklist measurement table);
- a SIMD group spinning on a value another SIMD group of its workgroup stores, with no barrier in between.

On the GPU the resident workgroups run at the same time, so these programs make progress (up to occupancy:
`dispatch::co_resident_workgroups`, the B5 probe). In the model they spin until the step limit; ZLUDA reports
that as the sticky `CUDA_ERROR_LAUNCH_TIMEOUT` (702), and ZLUDA's `cuLaunchCooperativeKernel` accepts only
one-block grids for this reason.

What a fix would look like: preempt a SIMD group after a step budget and switch to another runnable group,
across the co-resident workgroups of a wave, so a waiter yields to the group it waits on. Open questions that
need the machine before writing it:

- which workgroups are co-resident and in what order they start (is a later workgroup guaranteed to run
  while an earlier one spins, for a grid within `co_resident_workgroups`?);
- the forward-progress guarantee between SIMD groups of one workgroup when one spins (does the hardware
  interleave them, or can a spinning group starve another indefinitely?);
- the memory ordering a spinning reader observes for another core's plain and atomic stores (the
  `lookback` release / acquire probe, checklist measurement table), which decides what fences the scheduler
  must model at a switch.

## 2. Threadgroup memory outside the declared size: loads read 0, stores are dropped

`func.rs` (the threadgroup load / store path, "outside the declared threadgroup memory: loads read 0, stores are
dropped") treats an access past the declared threadgroup allocation as a dropped store and a zero load. CUDA
faults such an access (`CUDA_ERROR_ILLEGAL_ADDRESS`, 700, sticky), so ZLUDA cannot report it today
(`smem_oob`, the known failure of ZLUDA's `tests/device/test_device.py`).

This is a modelling choice, not a measured behaviour. Questions for the machine:

- what an out-of-range threadgroup load returns and what an out-of-range store does on the GPU: zero / dropped
  as modelled, a wrap within the core's threadgroup memory, another workgroup's data, or a fault of the command
  buffer;
- whether that depends on the distance past the allocation (inside the core's 64 KiB, beyond it).

If the GPU does not fault, the simulator should keep (or be corrected to) the measured behaviour, and CUDA's
fault belongs in ZLUDA: a bounds check on shared accesses that records a fault in ZLUDA's `__zluda_fault`
module variable (already reported as sticky 700 by its driver), at a cost on every shared access, so likely
opt-in or strict-only. If the GPU faults, the model should report a fault and ZLUDA maps it to 700 directly.
