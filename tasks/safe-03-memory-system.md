# Task safe-03: memory-system parameters

Read [safe-probes.md](safe-probes.md) first.

**Goal:** measured, not assumed, values for the memory parameters the cost model and a CUDA runtime need. Everything
uses the proven pointer-chase and stream kernels (`agx_loop.chase_body` / `stream_body`, emulator-gated; 13.2 M
hops already run). The baseline is the playground's `experiments/memory/load-latency.md` and `novel-results/2026-10-11-memory-hierarchy`.

## 1. Cache line size

- Run pointer chases whose next element sits at offsets 4 .. 256 bytes inside and across lines. Look for the latency
  step where a neighbour stops being free.
- Do this for L1 and for L2 / SLC.

## 2. Page size and TLB reach

- Run chases with one element per page (16 KiB GPU pages are assumed) over 1 .. 100,000 pages.
- Look for the latency step that marks TLB reach (the earlier study bounded the TLB cost at about 32 ns but did not
  find its capacity).
- Repeat with BOs mapped at 16 KiB and at 2 MiB alignment, if the UAPI allows larger mappings.

## 3. SLC capacity and sharing

- Run chases with footprints 4 .. 96 MiB in 1 MiB steps, first from one core, then from all 38 cores.
- Find where latency leaves the SLC plateau, and whether the CPU sharing the SLC changes it: run once with an idle
  CPU and once with a CPU thread streaming 64 MiB.

## 4. CPU -> GPU and GPU -> CPU visibility latency (unified memory)

- Basis: the playground's in-shader-clock experiment (`make test-in-shader-clock`) showed that a CPU-written counter
  is visible inside a running shader.
- **Measure:**
  - A CPU thread writes increasing values. A bounded GPU loop (emulator-gated, fixed trip count, no data-dependent
    exit) samples them, recording sample index and value.
  - From that, derive the visibility delay and the update granularity.
  - The reverse direction: a GPU store as a CPU thread polling with `CNTVCT_EL0` sees it.
- **Never** write a loop that waits for a value (it could spin forever). Sample a fixed number of times and analyse
  offline.

## Deliverables

- A playground `novel-results/<date>-memory-parameters` with a verify target.
- Updated L1 / L2 / SLC / TLB parameters in `rust/src/cost.rs` if they differ, re-validated with
  `make verify-memory-model`.
