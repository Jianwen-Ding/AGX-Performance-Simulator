# Follow-ups from the PTX front end / AGX backend optimization work

Areas where the simulator (functional model or `agx_cost`) could not settle a PTX front end or AGX LLVM backend decision,
collected while optimizing without hardware (October 2026, the PTX front end and llvm-project branch
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
   threads. `agx_cost` now derives groups per core from registers and threadgroup memory (`occupancy`) and estimates a
   full core (`--core-throughput`, [rust/README.md](../rust/README.md)); that estimate is not validated on hardware.
   Still needed: a measured stack load/store cost
   ([safe-11-occupancy-and-clock.md](safe-11-occupancy-and-clock.md)).
3. **Instruction rows the LLVM scheduling model now uses.** The AGX `SchedMachineModel` was checked against
   `bench.json.gz`: `simd_matrix_fmadd32/16` (latency 17, issue 16), `fmadd32` and 32-bit `icmpsel` / `fcmpsel`
   (issue ~2 with three or more 32-bit sources), `rcp` (issue 3), `imadd_wide` (latency 4), `simd_prefix_iadd`
   (latency 18) differ from the generic classes. Worth confirming on G13 and Apple9 (the LLVM models reuse G14C
   numbers) and adding issue rows for cmpsel with two register sources.
4. **Launch submission and GPU idle time.** The PTX front end gained `AGX_RUNTIME_LAUNCH_BATCH=1` (ordinary launches gathered into
   one command while the queue is busy). The host submit cost (~20-30 us) and the gap between commands (~15-20 us)
   are hardware numbers; a model of submission, queue idle and per-dispatch cost inside a batched command would let
   the policy (gather only while busy, MAX_BATCH 64) be tuned offline
   ([safe-04-driver-submission.md](safe-04-driver-submission.md)). Graph batching measured on hardware traded
   generation throughput for prompt throughput in Qwen: the cause is not modelled.
5. **Memory access after ALU work (`mem_quiet_device_store_ref` 33.6, `mem_quiet_load_ref` 20).** The AGX backend
   now clusters device and threadgroup stores in the machine scheduler (`AGXMemCluster.cpp`) on the strength of
   this stall alone: the PTX front end's PTX tests -0.8%, agx_libs -9.5% in `agx_cost`, much of it in the tile GEMM epilogue.
   Worth confirming on hardware: whether a store right after an ALU instruction really waits ~34 cycles while
   back-to-back stores issue every ~3, whether a `wait` or a load between them resets it, and whether the same holds
   for threadgroup stores (no kernel changed when only those were clustered). Device loads are clustered too since
   agx-bench gained real-world kernels (decode attention -34%, q4_0 mat-vec -28% in the model): the same
   `mem_quiet_load_ref` question applies, and whether 8-16 loads in a row issue as the model says. Threadgroup load
   clustering stays off only because it raises tiled SGEMM from 70 to 77 GPRs, which the model cannot price (item 2). A post-RA scheduler was also tried: both LLVM
   post-RA schedulers spread the stores again (+1% to +3% total), so post-RA scheduling stays disabled.

6. **Same-address atomics from one SIMD-group.** `agx_cost` prices 32 lanes' atomics to one word like 32 to
   distinct words, so the AGX backend's atomic optimizer (`-agx-atomic-optimizer`: one atomic of the group's
   reduction) costs more in the model (agx-bench `cuda_compact` +41% cycles). Needed: a measurement of 1, 2, 8, 32
   lanes adding to one word vs. distinct words, device and threadgroup memory.

12. **Uniform register sources and preload size.** The PTX front end now compiles kernels with the AGX backend's
    `-agx-uniform-constants`, which reads wide 32-bit constants from preloaded uniforms (`.agx.kernels` extension 6)
    instead of `mov_imm` into GPRs. In the model this cuts 2-16 GPRs and gives -1..-2.4% tput geomean, because
    `agx_cost` charges nothing for a uniform source operand and nothing for more preloaded uniforms. Need to know:
    is a uniform source as fast as a GPR one for every ALU class (f16, f32, int, cmpsel)? Do more uniform words slow
    the dispatch or the start of each workgroup? Is there a uniform-file limit on resident workgroups?
    (llvm-project agx-bench EXPERIMENTS.md, entry A15.)

13. **Vector stores whose sources need copies, and bfeil's pipe.** Two AGX backend choices rest on model costs
    that need a hardware check. (a) A vector store whose lanes are loaded values in another order needs a register
    copy, and the model then charges the store-after-ALU stall (item 5): split into scalar stores, `ld_st_offset`
    is -37% cycles (AGXVectorAccesses, EXPERIMENTS.md L4-E1; threadgroup stores are not split, their split costs
    an address register). (b) `bfeil` with a mask replaces a shift and an `and` (A22), but where the shift has
    other uses the model prefers the cheaper `and` (fma_bf16x2 +10% tput otherwise): `bfeil` is in the
    WriteIMul class. Needed: whether a 2- or 4-lane device store straight after a move stalls as long as after
    arithmetic, and `bfeil`'s issue rate against `and` / `bitop`.
14. **Shared divisor reciprocals and register pressure (A20).** Computing a 32-bit divisor's reciprocal once per
    divisor (permute4d: 46 -> 60 GPRs) is -8% tput in the model because 60 GPRs still fit the same resident
    groups. On hardware the extra registers could cost occupancy on kernels nearer a register boundary; worth
    measuring permute4d-like index math with 2-4 runtime divisors.
15. **Alignment of 8- and 16-bit vector accesses (extends B33).** B33 measured i32 `xy` / `xyzw` at any 4-byte
    alignment. Are i16-format `xy` / `xyzw` device loads correct at 2-byte alignment (and i8 `xy` / `xyzw` at any
    byte), also across 64-byte lines and pages? With the AGX backend's `-agx-element-aligned-vectors` (off) the model
    gives q8_matvec -54% cycles, attn_decode -14% tput (12 -> 3 loads), f16 tile GEMMs up to -17% tput
    (llvm-project agx-bench EXPERIMENTS.md R23): the largest gain left that rests on one hardware fact.
16. **Trip counts computed by a division.** `agx_cost` skips `rcp` and `convert` (emulator_skipped), so a loop whose
    bounds come from an integer division (the backend's 32-bit division uses the f32 reciprocal) has an unknown trip
    count, and the loop is costed as not running (warning "if_icmp condition depends on ... a skipped result"). agx_libs'
    convolutions now compute their loop ranges with ceiling divisions (EXPERIMENTS.md A28): agx-bench reported -99%
    for bwd_filter, which is this, not the change. Needed: emulate `rcp` and the int/float converts (exactly as the
    hardware rounds) in the default trace, or warn louder when a costed loop's trip count is unknown. (Mapping any
    buffer, `--alloc`, switches to the functional trace, which runs them: agx-bench's lib_shapes.py does.)

## Functional model / encodings

7. **fmadd lane-offset form** ([safe-07-fmadd-lane-offset.md](safe-07-fmadd-lane-offset.md)). Not modelled, so the
   AGX backend cannot pattern-match `shuffle_up` + `fma` into it (the PTX front end's task 3.2) and test the result. Needed: the
   coverage cases and functional support for bits 53 / 68 / 72 / 73 / 75-79.
8. **Atomic bit 45 on multi-die chips (the PTX front end's T5).** The backend sets it for `multi-die-coherency` (G13S/C/D, G14D);
   the simulator (G14C model) decodes it on `cmpxchg` as the u64 min/max variant, so the sparse atomic SpMV kernels
   fail in the simulator on those chips. Needed: a run on an M1 Pro/Max/Ultra or M2 Ultra.
9. **Device-coherent, not host-coherent, accesses (the PTX front end's T3).** AIR's device-coherent loads and stores are emitted
   as the volatile (host-coherent) form; a device-scope cache control, if one exists, would be cheaper
   ([safe-10-memory-ordering.md](safe-10-memory-ordering.md)).
10. **Software f64 cost.** The AGX backend's emulated double add / mul / fma / div / sqrt are 183 / 189 / 412 / 284 /
   297 instructions, mostly `icmpsel` chains and 64-bit shifts on 32-bit halves. A cycle (not instruction) comparison
   of variants needs the 2-cycle cmpsel issue and 64-bit iadd costs right in `agx_cost`.
11. **Active-thread special registers.** In `agx_run`, `active_thread_index_in_simdgroup` (sr58) and
    `total_active_threads_in_simdgroup` (sr59) disagree with `ballot`: with one active thread sr59 read 11, with 32
    it read 1 (PTX-formalization atomics.ptx / launch-maxntid-total.ptx under the atomic optimizer). The backend
    now derives both from the ballot; the emulation (or the register numbering) needs checking against hardware.
17. **Empty-mask memory accesses (B31 against `agx_cost`).** With an if body of at most 4 instructions that holds
    device / threadgroup loads or stores run under an empty execution mask instead of jumped over, `agx_cost` reports
    every affected agx-bench kernel faster (conv3x3 tput -17%, bitonic_smem -7%, saxpy -20%; llvm-project
    EXPERIMENTS.md L15-E135, rejected). B31 measured a masked `device_load` + `wait` at ~13-20 ns and a masked
    `device_store` at ~9-12 ns (more than an active store), against ~1-4 ns for masked ALU instructions, so the
    model's cost of a masked access, or of the jump it replaces, disagrees with the hardware. Charge masked memory
    accesses as B31 measured.
