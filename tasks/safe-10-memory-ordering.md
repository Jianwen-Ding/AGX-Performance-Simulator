# Task safe-10: memory ordering and fences (the PTX memory model)

Read [safe-probes.md](safe-probes.md) first (safety rules).

**Goal:** a complete fence mapping for PTX `fence.*`, `membar.*`, `ld.acquire` and `st.release` at CTA and GPU scope.
Today only message passing (MP) across cores has been measured.

**Background** (playground `novel-results/2026-10-11-memory-ordering`, runner `scripts/run_drm_litmus.py`):
- The litmus tests have no loops and no spin-waits. Each lane reads a fixed number of times and the outcomes are
  counted offline.
- `f533` is the release fence and `f59d` (after `wait 0`) the acquire. `memory_barrier` (f596) alone does not order
  stores between cores. Same-core pairs never violated.
- **Open:**
  - store → load (SB) and load → store (LB) orderings;
  - threadgroup-memory ordering;
  - the cost of f533 and f59d;
  - ordering across dispatches;
  - whether Mesa's own device-scope barrier is unsafe.

## Safety

- Only the seven Mesa-emitted barrier encodings (f596, f533, f59a, f59e, f503, f599, f59d), plus loads, stores and
  atomics already recorded as `safe`.
- **No spin-waits ever.** Every read happens a fixed number of times, and the analysis is offline.
- The 251 other `memory_barrier a, b, c` encodings stay out (risky list).

## Steps

1. **Coverage case** (`MANUAL` item `memory_ordering`): record outcome counts, not words. Add a case type to the
   catalog whose check is "violations == 0" for the fenced variants and "violations > 0 allowed" for the unfenced ones.
2. **Store buffering (SB):**
   - Lane A: `x = 1; fence; r1 = y`. Lane B: `y = 1; fence; r2 = x`. A violation is r1 = r2 = 0.
   - Run each of the 7 barriers and their pairs, on the same core and on different cores (SR20 recorded).
   - This decides what `fence.sc.gpu` needs.
3. **Load buffering (LB):**
   - Lane A: `r1 = x; fence; y = 1`. Lane B: `r2 = y; fence; x = 1`. A violation is r1 = r2 = 1.
4. **Threadgroup memory:**
   - MP, SB and LB between SIMD groups of one workgroup through threadgroup memory, with no fence, `wait`, 0x78 and
     0x68.
   - **The barrier variants must be executed by every group exactly once** (as in `sync-and-end-opcodes`), so they
     cannot deadlock.
   - This decides `membar.cta` and `bar.sync` semantics.
5. **Atomics as synchronisation:** MP where the flag is set with a device atomic (add without return, and with
   return) instead of a store, with and without f533. This decides `atom.release`.
6. **Fence cost:** in the `wait-cost` style counted loop (bounded, emulator-gated), measure f533, f59d, f533 + f59d
   and Mesa's 7-barrier sequence, after nothing, after a store and after a load. Fast-clock cluster, power source
   recorded.
7. **Across dispatches:**
   - Dispatch 1 writes; dispatch 2 in the next submission reads with a normal (cached) load, without f59d.
   - Is it always fresh? This decides whether a CUDA runtime must emit a cache invalidate at kernel start.

## Deliverables

- Litmus results with a table: PTX operation → AGX sequence, scope, measured cost.
- A coverage case type.
- Fence costs in `cost.rs`.
- An update to the playground finding.
- A `make verify-memory-ordering` extension. Push both repositories.
