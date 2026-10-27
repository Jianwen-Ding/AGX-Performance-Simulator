# Follow-ups from ZLUDA / AGX backend optimization work

Areas where the simulator (functional model or `agx_cost`) could not settle a ZLUDA or AGX LLVM backend decision,
collected while optimizing without hardware (October 2026, ZLUDA and llvm-project branch
`claude/serene-tesla-fwv2f2`). Each item names the decision it blocks and what would settle it. Existing task files
are linked where they already cover part of it.

## Cost model (`agx_cost`)

1. **Memory bandwidth and coalescing.** `agx_cost` prices a load by latency and issue, not by how many cache lines
   the lanes touch. Blocked: cuSPARSE CSR SpMV with one SIMD-group per row (coalesced column/value loads) was
   rejected because the model shows the one-thread-per-row kernel issuing fewer instructions, but the row-per-thread
   kernel's lanes read 32 different rows. Needed: a per-instruction line count / bandwidth term, calibrated against
   strided vs contiguous loads on the M2 Max ([safe-03-memory-system.md](safe-03-memory-system.md)).
2. **Occupancy and spills.** `--groups G` simulates G groups regardless of the kernel's register count, and spilled
   stack traffic is priced like any memory access. Blocked: a 4x8-matrix tile GEMM (64x128 workgroup tile) is 13%
   cheaper per element in the model but needs 128 GPRs, spills 160 bytes and drops occupancy from 448 to 384
   threads. Needed: groups per core derived from GPRs / threadgroup memory, and a measured stack load/store cost
   ([safe-11-occupancy-and-clock.md](safe-11-occupancy-and-clock.md)).
3. **Instruction rows the LLVM scheduling model now uses.** The AGX `SchedMachineModel` was checked against
   `bench.json.gz`: `simd_matrix_fmadd32/16` (latency 17, issue 16), `fmadd32` and 32-bit `icmpsel` / `fcmpsel`
   (issue ~2 with three or more 32-bit sources), `rcp` (issue 3), `imadd_wide` (latency 4), `simd_prefix_iadd`
   (latency 18) differ from the generic classes. Worth confirming on G13 and Apple9 (the LLVM models reuse G14C
   numbers) and adding issue rows for cmpsel with two register sources.
4. **Launch submission and GPU idle time.** ZLUDA gained `ZLUDA_AGX_LAUNCH_BATCH=1` (ordinary launches gathered into
   one command while the queue is busy). The host submit cost (~20-30 us) and the gap between commands (~15-20 us)
   are hardware numbers; a model of submission, queue idle and per-dispatch cost inside a batched command would let
   the policy (gather only while busy, MAX_BATCH 64) be tuned offline
   ([safe-04-driver-submission.md](safe-04-driver-submission.md)). Graph batching measured on hardware traded
   generation throughput for prompt throughput in Qwen: the cause is not modelled.
5. **Memory access after ALU work (`mem_quiet_device_store_ref` 33.6, `mem_quiet_load_ref` 20).** The AGX backend
   now clusters device and threadgroup stores in the machine scheduler (`AGXMemCluster.cpp`) on the strength of
   this stall alone: ZLUDA PTX tests -0.8%, agx_libs -9.5% in `agx_cost`, much of it in the tile GEMM epilogue.
   Worth confirming on hardware: whether a store right after an ALU instruction really waits ~34 cycles while
   back-to-back stores issue every ~3, whether a `wait` or a load between them resets it, and whether the same holds
   for threadgroup stores (no kernel changed when only those were clustered). Load clustering is net worse in the
   model (+0.6% on agx_libs, loads complete later) and stays off. A post-RA scheduler was also tried: both LLVM
   post-RA schedulers spread the stores again (+1% to +3% total), so post-RA scheduling stays disabled.

## Functional model / encodings

6. **fmadd lane-offset form** ([safe-07-fmadd-lane-offset.md](safe-07-fmadd-lane-offset.md)). Not modelled, so the
   AGX backend cannot pattern-match `shuffle_up` + `fma` into it (ZLUDA task 3.2) and test the result. Needed: the
   coverage cases and functional support for bits 53 / 68 / 72 / 73 / 75-79.
7. **Atomic bit 45 on multi-die chips (ZLUDA T5).** The backend sets it for `multi-die-coherency` (G13S/C/D, G14D);
   the simulator (G14C model) decodes it on `cmpxchg` as the u64 min/max variant, so the sparse atomic SpMV kernels
   fail in the simulator on those chips. Needed: a run on an M1 Pro/Max/Ultra or M2 Ultra.
8. **Device-coherent, not host-coherent, accesses (ZLUDA T3).** AIR's device-coherent loads and stores are emitted
   as the volatile (host-coherent) form; a device-scope cache control, if one exists, would be cheaper
   ([safe-10-memory-ordering.md](safe-10-memory-ordering.md)).
9. **Software f64 cost.** The AGX backend's emulated double add / mul / fma / div / sqrt are 183 / 189 / 412 / 284 /
   297 instructions, mostly `icmpsel` chains and 64-bit shifts on 32-bit halves. A cycle (not instruction) comparison
   of variants needs the 2-cycle cmpsel issue and 64-bit iadd costs right in `agx_cost`.
