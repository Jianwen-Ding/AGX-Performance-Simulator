# Task ptx-re-blockers: hardware facts the PTX → AGX stack is blocked on

Read [safe-probes.md](safe-probes.md) first.

**Goal:** one list of every reverse-engineering question that blocks running CUDA/PTX on the M2 Max (G14C) through
ZLUDA → LLVM AGX backend → a direct-DRM runtime. Everything else in that stack can be built offline against this
simulator. Each item says who needs it, what is already known (with the task or result that covers it), and the
smallest probe that would answer the open part.

Items already answered by an existing task are listed only so the dependency is visible; do not redo them.

The consumers are:
- the LLVM AGX backend (`llvm/lib/Target/AGX`, ABI in `llvm/docs/AGXUsage.rst`) in Jianwen-Ding/llvm-project;
- ZLUDA's PTX front end and the planned AGX/Asahi runtime backend in Jianwen-Ding/ZLUDA.

## Priority order

1. **B1 scratch / stack** and **B2 a generic hardware runner**. Without them no real CUDA kernel can run.
2. **B3 64-bit atomics**, **B4 intra-SIMD-group forward progress**, **B5 cross-workgroup co-residency**,
   **B7 CPU → GPU visibility**: correctness of common CUDA idioms.
3. **B6 dispatch time limit**, **B8 code region**, **B9 kernel parameter space**, **B20 failed-job reporting**: limits
   and errors a runtime must report.
4. B10–B13: already covered by existing tasks; the consumers wait on them.

## B1. Scratch / stack memory for direct-DRM dispatches (NEW, unsafe)

- **Needed for:**
  - PTX `.local` memory and `alloca`;
  - register spills (the backend spills to scratch);
  - call stacks for non-inlined and indirect calls (`call`, `brx.idx` targets, recursion).

  Large CUDA kernels spill routinely, so without scratch most real applications cannot run.
- **Known:**
  - Mesa-compiled spill / private-array shaders replay in `agx_run` (`coverage_catalog` item `stack-via-mesa`,
    `verify-stack-mesa`).
  - The safe-probe index excludes the stack item because hand-built direct-DRM scratch programs stalled twice.
- **Open:** the exact direct-DRM setup Mesa uses for a compute dispatch with scratch on G14:
  - the scratch-related CDM / USC words and any helper program;
  - the scratch buffer(s) and their layout (per-thread size granularity, per-core / per-SIMD-group stride, maximum
    size);
  - how the shader addresses its slice.
- **Probe:**
  1. **Data trace only.** Capture the submission of a Vulkan compute shader with a known private array through Mesa
     (e.g. via `run_vulkan_compute.py`), then diff it against the same shader without the array. This is
     black-box data at the UAPI boundary.
  2. Replay the traced words through `drm_compute.c` with the Mesa-compiled code, then with a hand-built shader that
     stores and reloads one scratch word per thread, gated in `agx_run`.
- **Deliverable:**
  - the field list and buffer layout, written up in the safe-12 interface map;
  - a `drm_compute.c` option that sets up scratch of a given per-thread size;
  - a coverage case.

## B2. A generic hardware runner with agx_run's command line (NEW, safe once gated)

- **Needed for:**
  - every end-to-end hardware test of compiler output: the PTX-formalization suite, the ZLUDA tests, the Sail-derived
    AGX difftests, the MLIR integration tests;
  - the planned runtime's launch path.
- **Known:** `drm_compute.c` runs a single code blob with a 1D count, one readback buffer and an optional table, and
  `run_drm_compute.py` wraps bodies in the control's prefix / suffix.
- **Open:** a runner that takes exactly what `agx_run` takes, so that every simulator test can also run on the GPU:
  - `code.bin` (a whole kernel produced by `llc -mtriple=agx -mcpu=agx-g14c`, `.text` only);
  - `--grid X[,Y,Z] --wg x[,y,z]`;
  - several buffers (`--map ADDR=FILE`, `--alloc ADDR=SIZE`, `--dump ADDR=SIZE:FILE`);
  - arbitrary uniform values (`--uniform N=VALUE`).

  The uniform convention is the one in `AGXUsage.rst`:
  - each kernel argument is a 64-bit pointer in a uniform register pair, starting at `u0`;
  - `threadgroups_per_grid` is at the `uN` the assembly comment names;
  - the global data buffer is at `uN_uN+1`.
- **Probe:**
  - Extend `drm_compute.c` with multi-BO mapping at fixed GPU VAs, a 3D grid, and a uniform file of any size, plus a
    `scripts/run_drm_kernel.py` with `agx_run`'s flags.
  - Gate every run on `agx_run` first (the same emulator gate as today).
  - Control: one of the `llvm/test/CodeGen/AGX` kernels, e.g. a vector add, compiled by our backend.
- **Deliverable:** the runner, with one compiler-generated kernel recorded as a coverage case, so its output can be
  checked against `agx_run` and the Sail model.
- **Status 2026-10-20: a runner now exists, hardware run requested.** Jianwen-Ding/ZLUDA branch
  `claude/bold-davinci-46xi9q` has:
  - `agx_runtime`, a Rust port of `drm_compute.c`. Its unit tests pin the exact USC and CDM bytes recorded in
    Apple-M2-Playground `experiments/baseline/direct-drm-compute.md`.
  - `agx_run_hw`, which takes `agx_run`'s command line.

  It has not run on a GPU yet. To run it on the M2:

  ```sh
  git clone -b claude/bold-davinci-46xi9q https://github.com/Jianwen-Ding/ZLUDA && cd ZLUDA
  cargo build --release -p agx_runtime          # needs only Rust and libc; no LLVM, ROCm or Mesa
  RUNNER=$PWD/target/release/agx_run_hw sh /path/to/AGX-Performance-Simulator/tasks/ptx-hw-smoke/run_smoke.sh
  ```

  `tasks/ptx-hw-smoke/` holds three kernels compiled by the LLVM AGX backend (`-mcpu=agx-g14c`). Each comes as
  `.ll` source, `.s` and the `.text` `.bin`:
  - `vadd`: three pointer arguments;
  - `ids2d`: a 2D grid, thread and threadgroup positions, a divergent branch;
  - `tgrev`: 256 bytes of threadgroup memory, a barrier, and a device atomic.

  `expected/` was recorded with `agx_run` (`run_smoke.sh --record`) and checked against an independent Python
  oracle. Each kernel was run in `agx_run` first and terminates there.
  - **Report:** the PASS / FAIL lines and the JSON line of each run.
  - On a FAIL, compare the CDM and USC bytes with those `drm_compute.c` prints. `agx_runtime`'s layout differs in
    three places: code at USC offset 0x400 of its own object, uniforms from `u0`, and pipeline words at offset 0.
- **New open questions found while writing the runner** (they matter for every CUDA launch):
  - **B2a** Does the CDM global size have to be a multiple of the local size, or does the hardware run partial
    workgroups? CUDA grids are always whole blocks, but Mesa may rely on partial ones. The runner rejects
    non-multiples.
  - **B2b** How wide is the USC Uniform word's size field (bit 20)? 8 and 12 halves are measured. The runtime splits
    ranges into words of 32 halves; one word per 64 halves would halve the word count. B9 asks for the limit.
  - **B2c** With no threadgroup memory, the Shared word encodes 64 KiB ("all"). Does that limit occupancy to one
    workgroup per core for kernels that use no threadgroup memory? If it does, what is the correct "none" encoding?
    safe-11 measured `floor(65536 / declared bytes)` for kernels that do declare some.

## B3. Native 64-bit device atomics (NEW, safe)

- **Needed for:**
  - PTX `atom.{add,min,max,and,or,xor,exch,cas}.b64/u64/s64` and `red.*.64`;
  - CUDA `atomicAdd(unsigned long long*)` and 64-bit CAS loops.

  The backend currently emulates them with a 1024-entry lock table (`AGXExpandWideAtomics.cpp`), which is slow and
  serialises unrelated addresses that hash together.
- **Known:** the coverage suite has 32-bit device and threadgroup atomics (`probe_atomics.py`, `run_drm_atomic.py`,
  `run_drm_tg_atomic.py`).
- **Open:**
  - Does the device atomic encoding have a 64-bit form that executes on G14?
  - If it does, which operations, and is it atomic as a 64-bit unit (no torn halves under contention)?
  - What do the 32-bit float atomics do (`atom.add.f32`, NaN / denormal handling)?
- **Probe (safe route first):**
  - Check whether Mesa advertises `shaderBufferInt64Atomics` / `VK_KHR_shader_atomic_int64` on this device and, if
    so, record a Mesa-compiled 64-bit atomic shader.
  - Only then try perturbing the size field of a known 32-bit atomic encoding, gated in `agx_run` and counted
    against the hang budget.
- **Deliverable:**
  - the encoding table, or a recorded "no native 64-bit atomics";
  - a contention test showing atomicity.

- **Status 2026-10-20, partly answered by existing findings** (playground `2026-10-17-atomic-variants`, `atom64`):
  - On G14C, bit 45 of the device atomic turns `sub` into a float32 add and `cmpxchg` into u64 min/max without a
    returned old value.
  - There is no 64-bit add, CAS or exchange, and no threadgroup float add.
  - The LLVM backend will lower `atomicrmw fadd f32` and unused-result `atomicrmw umin/umax i64` to these. Everything
    else keeps the lock table.
  - **Still open:** the float add's rounding and NaN/denormal behaviour under contention, beyond RNE/FTZ. Also whether
    u64 min/max has a returning form.

## B4. Forward progress inside a SIMD group (NEW, safe with a bounded loop)

- **Needed for:** PTX on `sm_70+` assumes independent thread scheduling. Common CUDA code spins inside a warp, e.g. a
  per-lane lock where one lane holds the lock and another waits. On a SIMT stack machine the waiting lanes can starve
  the holder forever. The backend avoids this for its own wide-atomic emulation (the critical section is inside the
  lock loop), but user code cannot be rewritten in general.
- **Open:** When some lanes of a SIMD group spin in a loop waiting for a value that other lanes of the *same* group
  write in the other branch, does the hardware ever run the other branch before the loop exits?
- **Probe:**
  - Lane 0 loops at most N iterations reading a threadgroup or device word.
  - Lanes 1..31 take the else-branch and write the word.
  - Record the iteration at which lane 0 saw the write, or N if it never did.
  - The loop is bounded, so there is no hang. Emulator-gate the shape, and vary which side of the `if` the spinner
    is on.
- **Deliverable:** "never / sometimes / always" with the branch order rule. This decides whether ZLUDA must rewrite
  intra-warp spin patterns or report `sm_6x` semantics.

## B5. Co-residency and progress across workgroups (NEW, safe with a bounded loop)

- **Needed for:**
  - cooperative launches (`cudaLaunchCooperativeKernel`, `grid.sync()`);
  - persistent-kernel and producer / consumer patterns across blocks.

  The runtime must know how many workgroups are guaranteed to run at the same time across all cores.
- **Known:** safe-11 covers per-core occupancy limits; safe-04 found at most 4 concurrent commands GPU-wide.
- **Open:** For a single dispatch with W workgroups (W ≤ the occupancy limit × the core count), are all W resident
  together, so that a software grid barrier completes? At what W does it stop completing?
- **Probe:** each workgroup increments one device counter and then spins, with a bounded count, until it reaches W.
  Record the final counter and the spin counts, sweeping W around the occupancy limit.
- **Deliverable:** the maximum co-resident workgroup count as a function of registers and threadgroup memory, to feed
  `cuOccupancyMaxActiveBlocksPerMultiprocessor` and the cooperative-launch limit.

## B6. Longest single compute command before the firmware times out (NEW, unsafe at the limit)

- **Needed for:** CUDA compute kernels often run for seconds. The runtime must know whether long kernels are killed,
  and must report `kernelExecTimeoutEnabled`.
- **Probe:**
  - Use the bounded delay loop from `run_sr113_reps.py` with increasing iteration counts.
  - Binary-search the duration at which the submission reports a timeout. Each probe at the limit costs one recovered
    timeout, so keep it to a few.
- **Deliverable:** the limit in ms, whether it depends on the queue or priority, and how the timeout is reported to
  userspace (which DRM error or result field).

## B7. CPU → GPU visibility during a dispatch (NEW, safe with a bounded loop)

- **Needed for:**
  - host-mapped pinned memory (`cudaHostAlloc` with `cudaHostAllocMapped`);
  - managed memory;
  - device code that polls a flag written by the host while the kernel runs (and the PTX `.sys` scope).
- **Known:** safe-03 step 4 measured the GPU → CPU direction: a coherent store plus `f533` gives a round trip under
  1 µs (`cpu-gpu-visibility`).
- **Open:** the CPU → GPU direction:
  - Does a GPU load loop that is bounded and uses the coherent / uncached load form see a CPU store made while the
    dispatch runs?
  - With which load form and fence?
  - With what latency?
- **Deliverable:** the load and fence sequence for `ld.volatile.sys` / `ld.acquire.sys`, and the latency.

## B8. Code region size, placement and replacement (NEW, safe)

- **Needed for:** module load / unload (`cuModuleLoad`, `cuModuleUnload`) and very large kernels (heavily unrolled
  application code).
- **Open:**
  - the maximum size of the shader code region and of one shader;
  - the alignment needed for a shader start;
  - whether new code written at the same GPU VA as an earlier shader is seen by the next submission without an
    explicit instruction-cache flush, and which UAPI flag or command forces one.
- **Probe:**
  - Run shader A, overwrite the same VA with shader B (different constant), submit again, and read back.
  - Repeat with a fresh VA as the control.
- **Deliverable:** the limits, and the rule a runtime must follow when reusing code memory.

## B9. Kernel parameter space and uniform limits (PARTLY COVERED by safe-09)

- **Needed for:** CUDA passes up to 4 KiB of kernel parameters (32 KiB on recent architectures). The backend's ABI
  puts each argument pointer in a uniform register pair, and by-value structs need more uniforms or a parameter
  buffer.
- **Known:** safe-09 measures the uniform file size and the preamble register budget.
- **Open (if safe-09 does not already answer it):**
  - the largest uniform range one dispatch can preload through the USC words;
  - whether preloading many uniforms costs anything at launch;
  - the fastest way to read a parameter buffer instead (uniform load from memory).
- **Deliverable:** the limit and the cost curve, so the backend can choose between uniforms and a parameter buffer.

## B14. Hardware calls for non-inlined functions and recursion (DEPENDS ON unsafe-01)

- **Needed for:**
  - PTX `call` to non-inlined `.func`s and recursion;
  - `brx.idx` and function pointers without the if-chain expansion;
  - keeping very large kernels below code-size limits.

  Today the LLVM AGX backend inlines every function, turns indirect calls into a test against each address-taken
  function, and rejects recursion.
- **Known:** unsafe-01 measured uniform, full-32-lane direct, register-indirect and software-saved-link nested calls.
  Return targets are shader-buffer-relative and the continuation is `pop_exec r0l, 2`.
- **Open (unsafe-01 items 2-6):** what code generation needs before it can emit calls:
  - partial-mask and empty-mask calls;
  - divergent register-call targets;
  - divergent returns;
  - nesting depth.

  Calls also need a stack (B1) for saved links and spilled registers.
- **Deliverable:** the call/return rules for divergent SIMD-groups and their cost, so the backend can emit calls
  where they pay off.

## B15. `simd_matrix_fmadd16/32` on G14C: operand map and cost (NEW, safe once gated)

- **Needed for:** every GEMM, convolution and attention kernel. The 8x8 SIMD-group matrix multiply-accumulate is
  emulated today with 24 shuffles + 16 FMAs per tile, because the native form's operands are unmapped on G14C.
  - The form is 0x6f with bit 27 set (applegpu `simd_matrix_fmadd16/32`, paired D/A/B/C).
  - The playground reached it (`2026-10-11-subop-spaces`) but neither mapped its operands nor timed it.
- **Probe:**
  - Start from a Mesa-compiled `simdgroup_matrix` shader (Vulkan cooperative matrix, if Mesa exposes it on G14).
    Otherwise use applegpu's encoding with A = identity and B = a known matrix.
  - Record which lane holds which element of A, B, C and D, and the accumulation order (fused, and in which k order).
  - Then measure latency and throughput against the 24-shuffle emulation.
- **Deliverable:** the per-lane layout and exact arithmetic. The `agx.mma` op then gets a native G14 lowering; its
  fixed layout is `r = 4*(t/16) + (t/2)%4`, `c = 4*((t/8)%2) + 2*(t%2)`.

## B16. The 0x78 barrier: what it orders (PARTLY MEASURED)

- **Known:** playground `barrier-0x68-vs-0x78`. 0x78 synchronizes the aligned 4-SIMD-group (128-thread) block only,
  costs 2.9 ns versus 9.3 ns for 0x68, and releases about 7 cycles after the last arrival versus about 13.
- **Open:**
  - Does it order threadgroup memory between the SIMD-groups of the block, like 0x68? A tg-litmus run with 0x78 as
    the only barrier would answer it.
  - What happens when a block is partly outside a workgroup smaller than 128 threads?
  - Is it safe in divergent control flow?
- **Needed for:** `agx.block_barrier`, which library kernels use for 2x2-SIMD-group tiles. Until this is answered it
  is lowered to 0x78 plus a threadgroup-memory fence.

## B17. `async_load` / `copy_1d` / `copy_2d` (NEW, unsafe)

- **Needed for:** prefetching operand tiles into threadgroup memory while computing, i.e. double buffering.
  - Apple's M5 guidance says such staging is unnecessary for peak GEMM there.
  - On G14 it may still pay for convolutions and FFT.
- **Known:** never run on G14C. Earlier attempts hung, and the form needs scratch/helper setup (playground
  `docs/isa-coverage.md`).
- **Open:**
  - the setup it needs (shared with B1);
  - its operand map;
  - whether it overlaps with ALU work;
  - how completion is signalled.
- **Priority:** low until B1 is done. The dialect does not expose it yet.

## B18. Lane-offset fmadd as a compiler target (FOLLOWS safe-07)

- **Known:** fmadd32/fmadd16 bit 53 gives `shuffle_up(a, k) * b + c` within 16-lane halves (bits 72/73 select
  segments of 2 or 4). The cross-half read is timing-dependent, and bit 52 is a `pow()` fixup.
- **Needed:** recorded coverage cases for the exact forms `agx.fma_lane_shift` will emit:
  - k = 1..15;
  - segment widths 16, 4 and 2;
  - fmadd32 and fmadd16;
  - partial masks.

  Plus the cost per instruction against `shuffle up` + `fma`. Once recorded, the native lowering is enabled for G14.

## B19. Bit 45 of the device atomic on two-die parts (M2 Ultra): coherency or variant? (NEW, safe)

- **Why it matters:** Mesa sets bit 45 on every device atomic of a part with more than one die (its
  `needs_g13x_coherency` also covers `num_dies > 1`), so the LLVM backend does the same for `agx-g14d`. On G14C,
  though, bit 45 makes `sub` a float32 add and `cmpxchg` a u64 min (2026-10-17-atomic-variants).
- **What the backend does meanwhile:** it never emits integer `sub` (it adds the negated value). But compare-and-swap,
  which atomic loads and every CAS loop rely on, would become u64 min on an M2 Ultra if the bit means the same there.
- **Probe (needs an M2 Ultra, G14D, under Asahi):** on G14D, run Mesa's atomic cmpxchg with and without bit 45, then
  sub, using the atomic-variants harness.
- **Deliverable:** which operations bit 45 changes on a two-die part.
  - If it changes them, the coherency bit needs another field, and both Mesa and the backend need a fix.
  - If no M2 Ultra is available, record that, and the backend will treat G14D atomics as unverified.

## B20. How a failed compute job is reported, and what survives it (NEW, unsafe: needs one fault)

- **Why it matters:**
  - CUDA must return `CUDA_ERROR_ILLEGAL_ADDRESS` / `CUDA_ERROR_LAUNCH_FAILED` instead of wrong results.
  - The uAPI has no result buffer for compute commands. ZLUDA's `agx_runtime` therefore exports the out-syncobj as a
    sync file after the wait and reads its status (`SYNC_IOC_FILE_INFO`, negative errno = failed job).
  - Which errno each case gives, and whether a later submit on the same queue or VM still works, are unknown.
- **Known:**
  - `DRM_ASAHI_FEATURE_SOFT_FAULTS` makes shader loads from unmapped memory return 0 and drops stores, so an
    out-of-bounds access may not fault at all.
  - The simulator's harnesses detect hangs with the syncobj wait timeout only.
- **Probe (one at a time; keep the AGX_CLEAN_EXIT recovery ready):**
  1. Read the soft-fault bit from `DRM_ASAHI_GET_PARAMS` (`features`) on the M2 Max.
  2. A store to an unbound VA far from every mapping, e.g. 2^39. Record the fence status (sync-file status), dmesg,
     and the read-back of a sentinel buffer written by an earlier thread.
  3. Two instruction faults: an illegal opcode and a jump outside the USC heap.
  4. After each one: submit the vadd smoke kernel on the same queue, a new queue in the same VM, and a new VM.
     Record which still succeed.
  5. The runner signals a **timeline** syncobj point per command and reads the status by exporting that point
     (`DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD` with `EXPORT_SYNC_FILE` and `point`). Check that the exported status of a
     faulted command's point is negative, and that the next command's point, which waited on it in queue order,
     reports its own status, not the earlier error.
- **Deliverable:**
  - the fence errno per case (fault, illegal instruction, firmware timeout from B6);
  - whether faults are soft on Asahi's M2 Max configuration;
  - which objects must be recreated after a failure (queue, VM, file descriptor).

  `agx_runtime/src/error.rs` maps these to CUDA errors and is updated from the result.

## B21. What an executed `trap` (0x0008) does (NEW, unsafe: may fault)

- **Why it matters:**
  - `llvm.trap`, PTX `trap` and CUDA `assert` compile to `trap`.
  - CUDA requires the kernel to stop and the launch to fail (`CUDA_ERROR_ASSERT` / `CUDA_ERROR_LAUNCH_FAILED`).
  - Mesa and the backend only use `trap` as padding after `stop`, so it is never executed in known code.
  - 2026-10-11-opcode-descriptions pairs it with `stop` (0x88) but did not run it.
- **What the runtime does meanwhile:**
  - `__assertfail` writes its message to the printf buffer, sets a flag word, does a release fence and traps.
  - The host reports the flag as `CUDA_ERROR_ASSERT` even if `trap` turns out to be harmless.
- **Probe (one SIMD group first, then a full grid):**
  1. Store a sentinel, then `trap`, then a second store.
  2. Run it under three masks:
     - every lane;
     - odd lanes only, inside Mesa's `if`;
     - one SIMD group of two workgroups.
  3. Record for each mask:
     - which stores land;
     - whether other SIMD groups and workgroups keep running;
     - the fence status of the command (B20);
     - the kernel log;
     - whether the next submission on the same queue still works.
- **Deliverable:** `trap` is a no-op, ends the SIMD group like `stop` (with or without the exec mask), or faults the
  command (with which errno). If it faults, also say whether it can be used for PTX `trap` without losing the
  queue.

## B10. Memory ordering (COVERED by safe-10)

The backend and ZLUDA lower PTX `fence.{cta,gpu,sys}`, `membar.*`, `ld.acquire`, `st.release`, `atom.*.acq_rel` from
safe-10's results. Once safe-10's litmus tests are done, please also list the exact instruction sequence each scope
needs. Add the `.sys` scope too, which combines safe-10 with B7.

## B11. Numeric semantics (COVERED by safe-01, open items there)

The backend's opt-in strict IEEE mode emulates in software whatever the hardware does not do exactly:
- denormal flushing;
- NaN canonicalization;
- `fneg` / `fabs` as bit operations;
- `fmin` / `fmax` flushing;
- `fdiv` / `rsqrt` refinement.

The open safe-01 items (setp / min non-ftz fix-ups) decide which emulation can be turned off. When they are done,
please also record:
- `fneg` / `fabs` of a NaN and a denormal (payload kept? sign only?);
- `fma` vs separate `fmul` + `fadd` rounding on the fused path;
- the ULP error of the hardware `rcp` / `rsqrt` / `div` forms over the fp32 range.

## B12. Textures and surfaces (COVERED by safe-05)

Needed by PTX `tex` / `tld4` / `txq` / `suld` / `sust` and CUDA texture / surface objects. The consumers need the
G14 texture and sampler descriptor layouts and the image memory layouts (linear and twiddled) that safe-05 records.

## B13. Occupancy, clocks, submission (COVERED by safe-04 / safe-11)

Consumed as is:
- ~55 µs warm launch;
- 64 in-order commands per SUBMIT;
- at most 4 concurrent commands GPU-wide (so CUDA streams beyond 4 serialise);
- priority has no effect;
- the in-shader clock skew and cost (`clock()`, `clock64()`, `%globaltimer`).

## Reporting

- Put the evidence in the playground as the other safe tasks do.
- Add a short "Done" note per item to this file.
- Raise new coverage cases where a probe produced a reusable kernel.
- Items B1 and B6 count against the hang budget; follow the recovery policy in `unsafe-01-call-ret.md`.
