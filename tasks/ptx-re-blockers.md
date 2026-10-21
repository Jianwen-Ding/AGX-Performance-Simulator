# Task ptx-re-blockers: hardware facts the PTX → AGX stack is blocked on

Read [safe-probes.md](safe-probes.md) first. The status of every item is tracked in
[ptx-re-blockers-tracker.md](ptx-re-blockers-tracker.md).

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
   **B7 CPU → GPU visibility**, **B30 forward progress of spinning threadgroups** (the `tcgen05` tensor memory
   pool): correctness of common CUDA idioms.
3. **B6 dispatch time limit**, **B8 code region**, **B9 kernel parameter space**, **B20 failed-job reporting**: limits
   and errors a runtime must report.
4. B10–B13: already covered by existing tasks; the consumers wait on them.
5. B29 native textures: CUDA textures already work in software, so this is a performance item.
6. B31 empty-mask instruction cost, B32 asynchronous submit cost (launch batching), B33 vector access alignment,
   B34 stack access cost and wait groups, and T1 (`agx_cost --uniform`): they decide backend tuning (whether to jump over
   short if bodies) and make agx-bench measure bounds-checked kernels.
7. Added 2026-10-21: **B41 prebuilt CUDA programs** (unblocks B23), **B37 checked mode on hardware**, **B35 a shader
   clock**, B38 VMs and queues per process, B39 allocation limits, B42 context teardown (after ZLUDA's real
   contexts), B36 IPC through dma-buf, B40 cost model against measured library kernels; and for the simulator's
   multiple contexts **B43 scheduling between VMs**, B44 isolation of faults and deadlocks, B45 shared BOs between VMs.

## Where the answers went (2026-10-21)

Branch `claude/bold-davinci-46xi9q` of Jianwen-Ding/ZLUDA and Jianwen-Ding/llvm-project:

| Item | Integrated as |
| --- | --- |
| B1 | `agx_runtime/src/scratch.rs`: Mesa's `libagx_helper` and scratch layout, the Registers word's bucket; header checked against the run's dump. Not yet run on hardware. |
| B4 | ZLUDA docs: no independent thread scheduling within a warp. |
| B5, B30 | `cuOccupancyMaxActiveBlocksPerMultiprocessor` reports min(15, floor(S/g) - 1) CTAs per core; docs on grid barriers and the tensor-memory slab wait. |
| B7 | backend: host-coherent device accesses with f59d / f533 and the coherent bits. |
| B16 | backend `block-barrier` feature (G14), `llvm.agx.block.barrier`; MLIR `agx.block_barrier` native on G14 (portable in strict mode and on G13). |
| B20 | `Queue::wait` fails a signalled command whose timestamps are both 0 (`CUDA_ERROR_LAUNCH_FAILED`). |
| B21 | ZLUDA docs: `trap` does nothing; the assert flag word reports the failure. |
| B22 | block dims [1024, 1024, 1024]; launches checked against grid * block < 2^32 per dimension. |
| B26 | ZLUDA docs: the one-threadgroup-fits rule and the 0 = 256 halves encoding are confirmed. |
| B27 | backend: `llvm.fpto[su]i.sat` to i32 / i16 is one `convert`. |
| B31 | backend: no `jmp_exec_none` over at most 4 ALU instructions (`-agx-max-unskipped-instrs`). |
| B33 | not used yet: the backend has no legal vector types to vectorize memory accesses with. |

## B1. Scratch / stack memory for direct-DRM dispatches (NEW, unsafe)

- **Done 2026-10-20:** works with Mesa's own `libagx_helper` (captured by an LD_PRELOAD trace of Mesa) and Mesa's
  scratch header. `drm_compute.c` `AGX_SCRATCH_DWORDS=N` + `AGX_HELPER_CODE`; Registers word spill bucket at bits
  18-21. 80 and 1,036 dwords per thread, up to 65,536 threads on 38 cores, 0 wrong, 0 hangs. Per-thread layout:
  `(adjust_words - 1 - offset/4) * 128 + 4 * lane`. Evidence: playground `results/2026-10-20-b1-direct-drm-scratch`.

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
  - kernel arguments are packed from `u0` (a pointer is a uniform pair); the object's `.agx.kernels` section gives
    each argument's position;
  - `threadgroups_per_grid` is at the `uN` the assembly comment names;
  - the global data buffer is at `uN_uN+1`.
- **Probe:**
  - Extend `drm_compute.c` with multi-BO mapping at fixed GPU VAs, a 3D grid, and a uniform file of any size, plus a
    `scripts/run_drm_kernel.py` with `agx_run`'s flags.
  - Gate every run on `agx_run` first (the same emulator gate as today).
  - Control: one of the `llvm/test/CodeGen/AGX` kernels, e.g. a vector add, compiled by our backend.
- **Deliverable:** the runner, with one compiler-generated kernel recorded as a coverage case, so its output can be
  checked against `agx_run` and the Sail model.
- **Done 2026-10-20 (steps 1 and 2 pass on the M2 Max):** `run_smoke.sh` with `agx_run_hw` gives PASS for vadd,
  ids2d, tgrev and tgrev-count. `agx_cuda_smoke` gives `"wrong":0` (graph, unified and copy2d also 0),
  `kernel_ms` 0.067. zluda_agx needed a 3-line aarch64 `c_char` fix to build. Evidence:
  playground `results/2026-10-20-b2-hw-runner`. B2a-B2c remain open.
- **Status 2026-10-20 (before the run): a runner now exists, hardware run requested.** Jianwen-Ding/ZLUDA branch
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
- **Step 2 (after the three kernels pass): vectorAdd through the CUDA driver API.**
  - `zluda_agx` on the same ZLUDA branch is `libnvcuda_agx.so`, the CUDA driver API on agx_runtime, with no ROCm.
  - `agx_cuda_smoke` runs vectorAdd (n = 50000, 256 threads per block) as a CUDA program would: `cuModuleLoadData`,
    `cuLaunchKernel` with `kernelParams`, events, `cuMemcpyDtoH`.
  - `tasks/ptx-hw-smoke/cuda/vectoradd_g14c.o` is zoc's output for `vectoradd.ptx`, so no LLVM is needed:

    ```sh
    cargo build --release -p zluda_agx
    target/release/agx_cuda_smoke /path/to/AGX-Performance-Simulator/tasks/ptx-hw-smoke/cuda/vectoradd_g14c.o
    ```

  - **Report:** its JSON line. `"wrong":0` is a pass, and `kernel_ms` is the firmware time between the events.
  - On an error, the CUDA error name says which call failed.
- **New open questions found while writing the runner** (they matter for every CUDA launch):
  - **Done 2026-10-21 (B2a, B2c; B2b by B9):** B2a: the hardware runs partial workgroups in x, y and z (global 100 /
    local 32 runs exactly 100 threads; 7x7x7 / 4x4x4 runs 343); the runner need not reject non-multiples. B2c: the
    "none" Shared word (bit 8 clear) is what the 2026-10-18 occupancy study ran its 0 KiB cells with, reaching 96
    SIMD groups and ~55 workgroups per core: no occupancy limit. Evidence: playground
    `results/2026-10-21-b2a-partial-workgroups`.
  - **B2a** Does the CDM global size have to be a multiple of the local size, or does the hardware run partial
    workgroups? CUDA grids are always whole blocks, but Mesa may rely on partial ones. The runner rejects
    non-multiples.
  - **B2b** How wide is the USC Uniform word's size field (bit 20)? 8 and 12 halves are measured. The runtime splits
    ranges into words of 32 halves; one word per 64 halves would halve the word count. B9 asks for the limit.
  - **B2c** With no threadgroup memory, the Shared word encodes 64 KiB ("all"). Does that limit occupancy to one
    workgroup per core for kernels that use no threadgroup memory? If it does, what is the correct "none" encoding?
    safe-11 measured `floor(65536 / declared bytes)` for kernels that do declare some.

## B3. Native 64-bit device atomics (NEW, safe)

- **Done 2026-10-21 (from existing records; no new run needed):** the float add (op 1 + bit 45) rounds to nearest
  even and flushes denormal inputs and results to sign-preserving zero on 2,048 edge / rounding pairs, and is atomic
  under contention (1,024 lanes adding 1.0 end at exactly 1024.0; mixed addends return one serial chain)
  (`novel-results/2026-10-17-atomic-variants`). NaN results are 0x7fc00000 (B24). u64 min / max has **no form that
  returns the old value**: Rt = 1 returns a changed flag (new != old, B24 item 2); ZLUDA rebuilds a returning
  `atom.min/max.u64` from the flag or keeps the lock table. Throughput of the float add under contention is
  unmeasured (performance only).

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
  - **NaN payload (from the Sail PTX difftest, 2026-10-20):** `atom.global.add.f32` with a NaN operand of payload
    0x7fffffff returns/stores 0x7fc00000 in the simulator; PTX (and the Sail model) keep the payload. Please record
    the stored word for memory = 1.0 and operand = 0x7fffffff, 0x7f800001, 0xffc00001, and for memory = 0x7fffffff,
    operand = 1.0. If the hardware canonicalizes, ZLUDA documents it; if not, the simulator needs fixing.

## B4. Forward progress inside a SIMD group (NEW, safe with a bounded loop)

- **Done 2026-10-20: never.** The `if` side of a divergent SIMD group runs to completion before the `else` side
  (whatever the lane counts): spinners in the `if` side never saw a write from `else` lanes in 200,000 trips. Intra-warp
  spins on the other branch deadlock; ZLUDA must rewrite them or report sm_6x semantics. Evidence: playground
  `results/2026-10-20-b4-simd-forward-progress`.

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

- **Done 2026-10-20:** a whole grid is reliably co-resident only up to ~12-15 workgroups per core (32-thread: 576
  workgroups in 15/15 runs, flaky from 608; 128-thread: 456 in all runs, fails at 494 = the register limit). Rule:
  W <= 38 x min(15, floor(S/g) - 1). Above it, late workgroups start only when others exit. Evidence: playground
  `results/2026-10-20-b5-coresidency`.

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

- **Done 2026-10-20:** no limit for a command that keeps executing: a single compute command ran **79.9 s** (loop
  maximum) with the correct result, no timeout at any length from 0.44 s up. A deadlocked command (B26's barrier) is
  ended after ~0.5 s with fence success and zero timestamps. `kernelExecTimeoutEnabled = 0`. `drm_compute.c`
  `AGX_WAIT_MS`. Evidence: playground `results/2026-10-20-b6-command-duration`.

- **Needed for:** CUDA compute kernels often run for seconds. The runtime must know whether long kernels are killed,
  and must report `kernelExecTimeoutEnabled`.
- **Probe:**
  - Use the bounded delay loop from `run_sr113_reps.py` with increasing iteration counts.
  - Binary-search the duration at which the submission reports a timeout. Each probe at the limit costs one recovered
    timeout, so keep it to a few.
- **Deliverable:** the limit in ms, whether it depends on the queue or priority, and how the timeout is reported to
  userspace (which DRM error or result field).

## B7. CPU → GPU visibility during a dispatch (NEW, safe with a bounded loop)

- **Done 2026-10-20:** only `f59d` + coherent `device_load` (Mesa u4 = 7) sees host writes mid-dispatch (all 64
  lanes, ~0.69 us per read); plain loads, coherent loads without the fence, `f59d` + plain loads and returning device
  atomics see nothing, with or without CPU cache cleans. `ld.volatile/acquire.sys` = f59d + coherent load. Evidence:
  playground `results/2026-10-20-b7-cpu-to-gpu` (and `novel-results/2026-10-16-cpu-gpu-visibility` for latency).

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

- **Correction 2026-10-21 (B42):** ring reuse is **not** safe without the USC invalidate once launches queue deep:
  zluda_agx's back-to-back pipeline words in the USC ring fault with 45+ queued launches and pass with the invalidate on
  every command (playground `results/2026-10-21-b42-context-teardown`). Emit it on every command, as Mesa does.

- **Done 2026-10-20:** 1,000 in-place rewrites of code and pipeline words at one USC offset: 0 wrong, 0 stale,
  with or without the barrier/USC-invalidate word (which costs ~1.8 us). Shaders of 256 MiB and a code address
  3 GiB into the USC range work. Ring reuse after completion is safe. Evidence: playground
  `results/2026-10-20-b8-code-region`.

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
- **Now load-bearing for every launch:** since ZLUDA 4ee63f1, `agx_runtime` stages each launch's pipeline words in a
  256 KiB ring in the USC heap, and its uniforms and CDM stream in a 4 MiB data ring. So the same addresses carry
  different contents as soon as an older command has finished. Mesa recycles its pools similarly, but nothing
  measured confirms that it is safe.
  - **Extra probe:** alternate two kernels with different pipeline words (different uniform counts and code
    offsets) 1,000 times on one queue, with the ring shrunk to 2 KiB (`USC_RING_BYTES`) so that every slot is
    reused within a few launches.
  - **Check:** each output matches its own kernel.
  - **If not:** say which flag or ordering makes the reuse safe. The fallback is per-launch buffer objects, the
    code before 4ee63f1.
- **Deliverable:** the limits, and the rule a runtime must follow when reusing code memory.

## B9. Kernel parameter space and uniform limits (PARTLY COVERED by safe-09)

- **Done 2026-10-20:** the whole 512-half (1 KiB) uniform file preloads correctly (u69 with 140 halves, u127,
  u128 across the 256-half 'Uniform high' boundary, u255 with 512), in Mesa's words of 64 halves; no measurable
  launch cost from 20 to 512 halves. Also answers B2b (64 halves per word works). `drm_compute.c`
  `AGX_UNIFORM_HALVES`. Evidence: playground `results/2026-10-20-b9-uniform-preload`.

- **Needed for:** CUDA passes up to 4 KiB of kernel parameters (32 KiB on recent architectures). The backend's ABI
  puts each argument pointer in a uniform register pair, and by-value structs need more uniforms or a parameter
  buffer.
- **Known:** safe-09 measures the uniform file size and the preamble register budget.
- **What the backend does now (LLVM AGX `AGXPromoteKernelArgs`):**
  - It passes up to 256 bytes per kernel of by-reference PTX parameters directly in the uniforms (128 halves),
    plus the 6 halves of `threadgroups_per_grid` and 4 for the global data buffer. Larger or dynamically indexed
    parameters stay a pointer to a parameter block.
  - ZLUDA's vectorAdd went from 2 dependent waits and 6 GPRs to 1 wait and 4 GPRs in the simulator.
  - Please check that a 140-half preload works on the M2 Max (the B2 runner can send any uniform count), and
    whether a smaller limit is needed.
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
- **What it costs today (2026-10-20):** cuBLAS's `agx_sgemm_tile64` (a 32x32 tile per SIMD group, K step 8) has
  1674 instructions on G14C. Its K loop holds 192 hardware shuffles and 46 spill accesses per iteration (81 in the
  kernel), at the 128-GPR limit. The shuffles come from the emulation, and so does most of the register pressure
  (gathered rows of A and columns of B). The cost model estimates 7017 cycles per SIMD group; the scheduler
  variants tried are no better. A native form removes both.

## B16. The 0x78 barrier: what it orders (PARTLY MEASURED)

- **Done 2026-10-20:** 0x78 orders threadgroup and device memory within its 4-group block with no fence
  (`novel-results/2026-10-20-barrier-bits`); it completes for 64- and 96-thread workgroups, when reached from two code
  addresses, and under a partial mask. Lower `agx.block_barrier` to 0x78 alone. Evidence: playground
  `results/2026-10-20-b16-block-barrier`.

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

- **Done 2026-10-21:** 28 of the 34 forms recorded on the GPU and exact in `suite_check` (fmadd_lane_b18 20/20,
  fmadd_lane_b18_masked 8/8). The 6 the simulator cannot gate (segment modes with k >= 2; fmadd32 in segment modes) ran
  through `probe_variants.py`: in-segment lanes read lane i - k exactly for every k; lanes whose source is outside the
  segment read 0 (fmadd16) or an **undefined, run-to-run varying value (fmadd32)**, so the compiler must discard or
  mask them. Cost: `fmadd32_lane` 3.9 cycles dependent, 2.48 issue, 30.9 lanes/clk/core (any k) against 6.84 / ~4.0 /
  26.1 for `shuffle up` + `fma`: enable the native lowering. Evidence: playground `results/2026-10-21-b18-fmadd-lane`.

- **In progress 2026-10-20:** the 34 missing forms are in the coverage catalog (areas `fmadd_lane_b18`,
  `fmadd_lane_b18_masked`: every k = 1..15, 4-/2-lane segments for both widths, more partial masks) and validate offline;
  **not yet recorded on the GPU** (`coverage.py record --area fmadd_lane_b18,fmadd_lane_b18_masked`). Cost not measured.

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
- **Done 2026-10-20 (steps 1-5):** faults never fail the fence. A store to VA 2^39 completes normally (soft
  faults); an undecoded opcode (0x54) and a jump into unmapped USC space complete with status **success** but
  unwritten timestamps (0 ticks, ~1 ms host). Same queue, new queue, new VM and the next in-order command all work.
  Runtimes must detect failure from zero timestamps. Evidence: playground `results/2026-10-20-b20-failed-jobs`.
- **Step 1 done 2026-10-20:** `features` = 1, so soft faults are **on** on this M2 Max (playground
  `results/2026-10-20-b25-gpu-telemetry/params.json`). Out-of-bounds loads return 0 and stores are dropped.
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

- **Done 2026-10-20: a no-op.** With all lanes, odd lanes under a partial mask, or one SIMD group of two workgroups,
  every store after `trap` lands, the command ends normally and the queue keeps working. `trap` cannot implement PTX
  `trap`; keep the flag-word path. Evidence: playground `results/2026-10-20-b21-trap`.

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

## B22. Launch limits a CUDA runtime reports (NEW, safe at Metal's limits, unsafe beyond)

- **Done 2026-10-20:** workgroups of 1,024 threads along x, y or z run; 2,048 runs nothing (fence ok, invalid
  timestamps). Global sizes are 32-bit thread counts: 2^31 and 2^32 - 32 threads along x counted exactly; y and z take
  65,536+ workgroups (4,194,304 tested). Evidence: playground `results/2026-10-20-b22-launch-limits`.

- **Why it matters:** `cuDeviceGetAttribute` must report:
  - `MAX_THREADS_PER_BLOCK`;
  - `MAX_BLOCK_DIM_X/Y/Z`;
  - `MAX_GRID_DIM_X/Y/Z`.

  `agx_runtime` reports Metal's limits for now: 1024 threads per threadgroup, [1024, 1024, 64], and 2^32 − 1 threads
  per grid dimension.
- **Probe with the B2 runner:**
  - local sizes 1024, 1024x1 and 1x1024, then (only if those pass) 2048;
  - global sizes of 2^31 and 2^32 − 32 threads along x, with a kernel that counts its threadgroups with an atomic;
  - for y and z, 65535 and 65536 threadgroups.
- **Deliverable:**
  - the largest accepted local size per dimension and in total, and what happens beyond it (rejected by the
    kernel, fault, or silent wrap);
  - whether the CDM global size words are thread counts that can exceed 2^31;
  - the per-dimension grid limits.

## B23. A real CUDA runtime program on zluda_agx (NEW, safe after B2 step 2 passes)

- **Blocked 2026-10-20:** the owner does not want nvcc / the CUDA toolkit installed on the Asahi machine.
  (Also found: nvcc 12.9 cannot parse Fedora 42's GCC 15 libstdc++ headers.)

- **Why it matters:** CUDA applications do not call the driver API directly. libcudart's start-up runs first:
  `cuGetProcAddress` for every function, `cuGetExportTable` for the private tables, fat-binary registration, then
  `cudaMalloc` / `cudaMemcpy` / `<<<>>>`. zluda_agx implements those paths, but they cannot be exercised without
  a real libcudart.
- **Steps (on the Asahi machine; nvcc runs without an NVIDIA GPU):**
  1. Install the CUDA toolkit for Linux aarch64 (SBSA), any 12.x.
  2. Build a small runtime program with PTX embedded, e.g. cuda-samples `vectorAdd`, or this one:

     ```sh
     cat > t.cu <<'CU'
     #include <cstdio>
     __global__ void k(int *o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = i * 3; if (i == 5) printf("hello from %d\n", i); }
     int main() { int *d, h[1000]; cudaMalloc(&d, sizeof h); k<<<4, 256>>>(d, 1000); cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
       int bad = 0; for (int i = 0; i < 1000; i++) bad += h[i] != i * 3; printf("%s %d bad\n", cudaGetErrorString(cudaGetLastError()), bad); }
     CU
     nvcc -gencode arch=compute_80,code=compute_80 t.cu -o t
     ```

  3. Run it against zluda_agx:

     ```sh
     mkdir -p lib && ln -sf $PWD/ZLUDA/target/release/libnvcuda_agx.so lib/libcuda.so.1
     LD_LIBRARY_PATH=$PWD/lib ZLUDA_AGX_ZOC=/path/to/zoc ./t
     ```

     `zoc` needs the LLVM AGX build. Without it, record how far start-up gets.
- **Deliverable:**
  - the program's output;
  - on failure, the last CUDA call that failed, with `ltrace -e 'cu*'` or `ZLUDA_AGX_LOG` if present;
  - any driver function it needed that returned `CUDA_ERROR_NOT_SUPPORTED`.

  Such a function goes to the zluda_agx backlog.
- **Also run `agx_cuda_smoke`** (ZLUDA `zluda_agx`, `cargo build --release -p zluda_agx`). It is the driver-API
  check, and now also covers unified memory.
  - zluda_agx maps every GPU buffer on the CPU at its GPU address, with `MAP_FIXED_NOREPLACE` above 64 GiB. That
    is what makes `cudaMallocManaged` and mapped host memory work.
  - Report its JSON line, especially `"unified"` (false means the CPU range was taken or the flag was refused) and
    `"unified_wrong"`.
  - Report the kernel version (`uname -r`) and page size (`getconf PAGESIZE`).
  - If `unified` is false, report `strace -e trace=mmap ./agx_cuda_smoke 2>&1 | head` for the failing `mmap`.
  - Run it once without an argument too, with `zoc` (`cargo build --release -p compiler --no-default-features
    --features agx`) next to it: it then links two PTX modules with `cuLinkCreate` / `cuLinkAddData` /
    `cuLinkComplete` and runs the linked kernel. Report `"link_ok"` (true expected; the same kernels pass on
    `agx_run`).
- **Also run cuBLAS SGEMM** through ZLUDA's `zluda_blas_agx`:
  - Build it with `ZLUDA_LLVM_BUILD_DIR=... cargo build --release -p zluda_blas_agx`.
  - Link `target/release/libcublas_agx.so` as `lib/libcublas.so.12`, next to the `libcuda.so.1` link.
  - Build this program with `nvcc t2.cu -lcublas -o t2` and run it as in step 3:

    ```sh
    cat > t2.cu <<'CU'
    #include <cstdio>
    #include <vector>
    #include <cublas_v2.h>
    int main() {
      for (int n : {64, 128, 100}) {           // 64 and 128 take the tiled kernel, 100 the generic one
        std::vector<float> a(n * n), b(n * n), c(n * n);
        for (int i = 0; i < n * n; i++) { a[i] = (i % 7) - 3; b[i] = (i % 5) - 2; }
        float *da, *db, *dc; cudaMalloc(&da, 4 * n * n); cudaMalloc(&db, 4 * n * n); cudaMalloc(&dc, 4 * n * n);
        cudaMemcpy(da, a.data(), 4 * n * n, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), 4 * n * n, cudaMemcpyHostToDevice);
        cublasHandle_t h; cublasCreate(&h); float one = 1, zero = 0;
        cublasStatus_t st = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, da, n, db, n, &zero, dc, n);
        cudaMemcpy(c.data(), dc, 4 * n * n, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
          float s = 0; for (int k = 0; k < n; k++) s += a[i + k * n] * b[k + j * n];
          bad += c[i + j * n] != s;
        }
        printf("n=%d status=%d bad=%d\n", n, st, bad);
        cublasDestroy(h);
      }
    }
    CU
    ```

  - Level 1 and 2 (zluda_blas_agx, `agx_libs/kernels/blas.mlir`): with the same handle, also run
    `cublasSdot`, `cublasSnrm2` (on values near 1e25, whose squares overflow f32), `cublasIsamax` (on a vector
    whose maximum appears twice: the first index must win), and `cublasSgemv` with `CUBLAS_OP_N` and
    `CUBLAS_OP_T` on a 1000 x 300 matrix, against a host loop. Run each in host and in device pointer mode.
    Report any difference above 1e-5 relative, and the time of a 2^24-element `cublasSdot`.
  - Level 3 beyond GEMM: `cublasStrsm` (all side / fill / transpose combinations on a 256 x 256 well-conditioned
    triangle, then check op(A) X against alpha B), `cublasSsyrk`, `cublasSsymm`, `cublasStrmm` in place (C = B),
    `cublasSgeam` and `cublasSgemmBatched` with 16 pointer triples (`torch.bmm` uses the strided form instead; a
    small C program is the simplest check). Also report the time of `cublasStrsm` at 1024 x 1024: it solves one
    column per thread, so this measures how far it is from a blocked solver.
  - Deliverable: the three lines, which should all read `status=0 bad=0`. The values are small integers, so the
    sums are exact. Also report the time of a 2048 x 2048 SGEMM if you can (cudaEvent around the call).
- **Then cuBLASLt and cuDNN**, through `zluda_blaslt_agx` (`libcublasLt.so.12`) and `zluda_dnn_agx`
  (`libcudnn.so.9`). Both are built like zluda_blas_agx. Link them next to the others.
  - If PyTorch for aarch64 with CUDA can be installed (it runs without an NVIDIA GPU only through these
    libraries), run:
    `python -c "import torch; a=torch.randn(256,512,device='cuda'); b=torch.randn(512,128,device='cuda');
    print((a@b - (a.cpu()@b.cpu()).cuda()).abs().max())"` (cuBLASLt / cuBLAS). Also run a 3x3
    `torch.nn.functional.conv2d` against the CPU result, in float32 and float16 (cuDNN).
  - Otherwise build cuDNN's `conv_sample` from cudnn-samples, or any small legacy-API convolution program, and run
    it.
  - Deliverable: the maximum differences (f32 should be about 1e-4, f16 about 1e-2), or the first failing call with
    `ZLUDA_AGX_LOG=1`, which names every unimplemented function the program touched.
  - The rest of cuDNN in f32 (zluda_dnn_agx on `agx_libs/kernels/dnn.mlir`): with PyTorch and
    `torch.backends.cudnn.enabled = True`, compare against the CPU, forward and backward (`.backward()` on a sum),
    for `torch.nn.BatchNorm2d(16)` in training mode on a 8 x 16 x 32 x 32 input (also check `running_mean` /
    `running_var` after three steps), `torch.nn.functional.max_pool2d(x, 3, 2, 1)`, `avg_pool2d` with
    `count_include_pad` true and false, and `torch.softmax` / `log_softmax` over dim 1. Without PyTorch, any
    legacy-API program calling `cudnnBatchNormalizationForwardTraining`, `cudnnPoolingForward` and
    `cudnnSoftmaxForward` will do.
  - Deliverable: the maximum differences (about 1e-5 expected), or the first failing call with `ZLUDA_AGX_LOG=1`.
- **Then cuFFT and cuSPARSE**, through `zluda_fft_agx` (`libcufft.so.11`) and `zluda_sparse_agx`
  (`libcusparse.so.12`), built and linked the same way.
  - With PyTorch: `torch.fft.fft(torch.randn(4096, dtype=torch.complex64, device='cuda'))` against the CPU result,
    a 2-D `torch.fft.rfft2` of 256 x 256, and `torch.sparse.mm(a.to_sparse_csr(), b)` with a 1000 x 1000 matrix of
    about 1 % non-zeros and a 1000 x 64 dense b (cuSPARSE SpMM), plus `a.to_sparse_csr() @ v` (SpMV).
  - Without PyTorch: the `simpleCUFFT` and cuSPARSE `spmv_csr` / `spmm_csr` / `spmv_coo` samples from
    CUDALibrarySamples.
  - The transposed and COO SpMV paths add with native f32 atomics (`atomic_rmw addf`, op 1 + bit 45). They also
    exercise B24 item 1 under contention: report whether results differ between repeated runs beyond f32
    reassociation (about 1e-5 relative).
  - Deliverable: the maximum differences, or the first failing call with `ZLUDA_AGX_LOG=1`.

## B24. Device-atomic and barrier corners the compiler and the Sail model refuse until measured (NEW, safe)

PTX-formalization's Sail AGX model (sail/agx, updated 2026-10-20) and the LLVM AGX backend now use the measured
G14C atomic forms. They stop with an error, or avoid the encoding, for these unmeasured corners. Each is a small
probe with the atomic-variants harness (`scripts/probe_atomics.py` style, surrogate gate first).

1. **`red.add.f32`:** op 1 + bit 45 with Rt = 0 (no result).
   - Does it add like the Rt = 1 form (same rounding, flushing and contention behaviour)?
   - The backend keeps a result register for float adds until this is answered. Integer atomics already drop
     theirs, as Mesa does.
2. **`umin64` / `umax64` changed flag:** with Rt = 1, what is the flag when the source equals memory, 0 or 1?
3. **`umin64` / `umax64` operands:**
   - an address ≡ 4 mod 8 (odd 32-bit index);
   - contention from 1024 lanes on one word (is the final value the true minimum?);
   - a GPR base instead of a uniform one.
4. **Ops 11-15 with bits 45-46 other than Mesa's.** Still "end the SIMD group"?
5. **`memory_barrier a, b, c` encodings other than the seven the backend emits** (f596 f533 f59a f59e f503 f599
   f59d). Which are accepted, which are no-ops, and do any hang? Run one per dispatch.
6. **The 31 six-byte low-`0b11` opcodes** with operand bytes other than those tested: still no-ops?
7. **G13X:** do `sub` / `cmpxchg` with bit 45 behave differently there? Only if a G13X device is available.

**Deliverable:** one line per item. The Sail model and the backend then enable or keep refusing each form.

**Done 2026-10-20 (items 1-4), playground `results/2026-10-20-b24-atomic-corners`:** (1) `red.add.f32` (Rt = 0) is
identical to the returning float add; (2) the u64 min/max changed flag is 0 on equality (`new != old`); (3) an
address = 4 mod 8 makes u64 min/max a silent no-op, 1,024-lane contention gives the exact min/max, a GPR base works;
(4) ops 11-15 with bits 45-46 = 0/1/3 all end the SIMD group with no memory effect. B3's NaN question: every NaN
result of the float add is 0x7fc00000 (from the 2026-10-17 records).

## B25. GPU telemetry Asahi exposes to userspace (NEW, safe, read-only)

- **Done 2026-10-20:** only `max_frequency_khz` (1,398,000) and in-process busy time (1 GHz command timestamps)
  are available without root. No current clock, GPU temperature, GPU power rail or runtime-PM counters; Heatpipe
  Power (uW, `macsmc_hwmon`) is the closest SoC proxy. Evidence: playground `results/2026-10-20-b25-gpu-telemetry`.

- **Why it matters:** NVML users (monitoring tools, PyTorch's `torch.cuda.utilization`, schedulers) query
  utilisation, clocks, temperature and power. ZLUDA's `zluda_ml_agx` returns `NOT_SUPPORTED` for all of them
  for now.
- **Probe (no GPU work needed):** on the M2 Max under Asahi, list and read:
  - `/sys/class/drm/card*/device/` and `/sys/class/devfreq/*`: current and possible GPU frequencies, and any
    busy-time counters;
  - `/sys/class/hwmon/*`: names and labels; which sensors belong to the GPU die or the GPU power rail;
  - any `debugfs` files of the `asahi` DRM driver readable without root;
  - whether `DRM_ASAHI_GET_TIME` or the firmware timestamps can give a busy fraction, e.g. summed command time
    over a wall-clock interval.
- **Deliverable:** the paths and units of each quantity that exists, and which need root. zluda_ml_agx then
  implements `nvmlDeviceGetUtilizationRates`, `GetClockInfo`, `GetTemperature` and `GetPowerUsage` from them.

## B26. Threadgroup size versus register count, and the 256-half encoding (NEW, safe except where marked)

- **Done 2026-10-20:** (1) 1,024 threads at 104 halves run correctly, with and without a barrier; (2) Registers field 0
  = 256 halves (384 threads round-trip r127); (3) 1,024 threads at 112 halves run without a barrier but **hang with a
  barrier** (killed after 0.50 s, fence status success). The one-threadgroup-fits rule is required. Evidence: playground
  `results/2026-10-20-b26-threadgroup-registers`.

- **Why it matters:** ZLUDA now reports `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` from the kernel's registers and
  refuses larger launches with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`. The LLVM AGX backend spills to honour PTX
  `.maxntid` / `.reqntid` / `.maxnreg` (function attributes `agx-max-threads-per-threadgroup`, `agx-max-gprs`).
  Both use safe-11's measured per-core budget, min(96, 2 * floor(3328 / (2 * H))) SIMD-groups, with H rounded up to
  8 halves. They also assume that **one threadgroup must fit in that budget**: 1024 threads at <= 104 halves, 384
  threads at 256.
- **Probes** (`scripts/run_drm_occupancy.py`, single threadgroup, a per-SIMD-group atomic counter plus a result
  check):
  1. 1024 threads at 104 halves: all 32 SIMD-groups run, and the result is correct. Safe: Mesa's table allows it.
  2. 384 threads at 256 halves, launch Register count field = 0. Mesa's `__gen_to_groups` encodes 256 halves as 0
     ("0 means all"), and Mesa ships it on M1/M2. This answers safe-11's "256 declared halves wraps" item. Confirm
     that 0 means 256 halves: the kernel writes r127 of every thread to memory, and the values are correct.
     zluda_agx and the backend rely on this.
  3. **UNSAFE, optional, run last:** one 1024-thread threadgroup at 112 halves, one granule over budget. Does the
     command fault, hang (watchdog), or run partially? The runtime never issues this, so it only decides how loudly
     the docs warn.
- **Deliverable:** pass or fail for 1 and 2, and the outcome of 3 if run. If 1 fails, report the real
  per-threadgroup limit for each H.

## B27. Out-of-range and NaN float-to-integer `convert` (NEW, safe)

- **Answered 2026-10-20 by existing hardware data** (playground `novel-results/2026-10-16-conversions`, 1.37 billion
  inputs, simulator exact): every float-to-integer mode **saturates** (+-inf -> max/min, negative -> 0 for unsigned),
  **NaN -> 0**, fp32 denormals read as 0, fp16 sources are full IEEE. 8/16-bit signed results are **zero-extended**
  (sign-extend when the destination is wider). So `FP_TO_[SU]INT_SAT` can be legal for 32-bit results with no clamps;
  16/8-bit ones need only the sign extension.

- **Why it matters:** PTX's float-to-integer `cvt` saturates (NaN gives 0). The LLVM AGX backend cannot assume that
  `convert f_to_s32` / `f_to_u32` / `f_to_s16` / `f_to_u16` do, so ZLUDA now lowers every such conversion to
  `llvm.fpto[su]i.sat`. That costs two or three extra `fcmpsel` per conversion (the clamps against -2^31, 2^31 - 2^7,
  2^32 - 2^8 and the NaN check). If the hardware already saturates, the backend can make `FP_TO_[SU]INT_SAT` legal
  and drop them.
- **Probes** (one kernel, `convert` of each mode, rtz and rte, from f32 and f16 sources, to s32, u32, s16 and u16):
  inputs +-0, +-1.5, 2^31 - 128, 2^31, 2^32, 1e20, -1e20, -1.0 (to unsigned), +-inf, quiet and signalling NaNs
  of both signs, and the largest denormal.
- **Deliverable:** the result bits per input and mode. For each mode: does it saturate, what does NaN give, and
  does a negative input to an unsigned mode give 0?

## B28. The threadgroup barrier when SIMD-groups have already ended (NEW, safe except where marked)

- **Done 2026-10-20:** the barrier does not wait for ended SIMD groups or lanes: probes 1-4 (group ended before /
  after one of two barriers, odd groups of 32, lanes stopped under a partial mask) all complete in ~8 us. One
  1,024-thread threadgroup can use the full 64 KiB of threadgroup memory. Evidence: playground
  `results/2026-10-20-b28-barrier-after-exit`.

- **Why it matters:** ZLUDA is adding PTX thread-block clusters (`.reqnctapercluster`, `%cluster_ctarank`, `mapa`,
  `shared::cluster`, `barrier.cluster`) by running a cluster of C CTAs as one threadgroup of C x ntid threads, with
  each CTA's shared memory in its own window of threadgroup memory. CTA barriers (`bar.sync`) and cluster barriers
  then both become the threadgroup barrier. PTX lets one CTA of a cluster exit while the others still wait at
  barriers (PTX-formalization `cluster-exit.ptx`; plain `bar.sync` after an early `ret` of whole warps is also
  common in CUDA code). That is only safe if the hardware barrier stops waiting for SIMD-groups that executed
  `stop`.
- **Probes** (one threadgroup, a watchdog on every dispatch, `scripts/run_drm_occupancy.py` style):
  1. 64 threads (2 SIMD-groups). SIMD-group 1 ends at once; SIMD-group 0 runs `threadgroup_barrier`, then
     stores a sentinel. Does the dispatch complete with the sentinel written, or hang?
  2. As 1, with SIMD-group 1 ending after one barrier and SIMD-group 0 running two.
  3. 1024 threads (32 SIMD-groups), the odd SIMD-groups ending first. Same question.
  4. Within one SIMD-group: half the lanes end (stop with a partial execution mask), the rest hit the barrier.
     Does the barrier wait for the ended lanes?
  5. **UNSAFE if 1-4 hang, run last:** what recovers the GPU after a hung barrier (the timeout path of
     B13/safe-11)?
- **Also record:** the largest threadgroup memory a single threadgroup can allocate on G14C (Mesa uses 32 KiB;
  is more accepted?), which bounds C x per-CTA shared memory.
- **Deliverable:** completes / hangs for 1-4. If the barrier waits for ended SIMD-groups, the compiler counts
  arrivals in threadgroup memory instead for kernels that can exit early, and needs to know that.

## B29. Native texture sampling against ZLUDA's software reference (NEW, safe through Mesa/Vulkan as in safe-05)

- **Why it matters:** ZLUDA can move a texture mode from software to the native `texture_sample` / `texture_load` /
  `image_write` path only where the hardware gives the result CUDA (and PTX-formalization's model) specifies.
  safe-05 records the encodings. This item asks for the comparison and for the corners safe-05 does not cover.
- **Reference:** `ptx/lib/zluda_ptx_impl_agx_tex.c` in Jianwen-Ding/ZLUDA, branch `claude/bold-davinci-46xi9q`. It
  is plain C; build it on the host and call `__zluda_ptx_impl_agx_tex` with a descriptor (layout documented in
  the file) to get the expected words for any probe.
- **Probes** (Vulkan compute shaders as in safe-05; record the result bits, not just pass/fail):
  1. **Linear filtering weights.** CUDA uses 9-bit fixed point (8 fractional bits, weights in steps of 1/256,
     rounded to nearest). safe-05's `texfilter` probe found 10 sub-texel bits on G14. For x = i + 0.5 + k/1024
     (k = 0..1023) over a 2-texel ramp (0.0, 1.0), print the result. Is there any sampler or descriptor state that
     gives 8-bit weights? If not, ZLUDA keeps linear filtering in software (or accepts the difference behind a flag).
  2. **Normalized integer reads.** u8 / s8 / u16 / s16 texels read as float: the exact result for every code
     (exhaustive). Is it v / 255 rounded once, and is s8 -128 clamped to -1.0?
  3. **Border.** CUDA's border address mode returns the border colour, and ZLUDA currently returns zeros. Which
     border colours can the G14 sampler hold (transparent black / opaque black / opaque white only, or
     arbitrary)? This decides whether `CUDA_TEXTURE_DESC.borderColor` can be honoured.
  4. **Wrap and mirror with unnormalized coordinates.** CUDA clamps them; what does G14 do?
  5. **Mip LOD.** For `textureGrad` with known gradients, which level and blend weight are chosen? ZLUDA uses
     λ = ½·log2(max(|dPdx·size|², |dPdy·size|²)), clamped, with a 1/256 blend weight. Also: is the min/max LOD clamp
     applied before or after the bias?
  6. **Cubemaps.** The face and in-face coordinates for directions on the face boundaries (|x| = |y|, |x| = |z|,
     signed zeros). Is there seam filtering across faces? ZLUDA clamps inside the face.
  7. **Depth compare.** LEQUAL against the first channel, per texel before filtering (ZLUDA's reference)? Which
     compare functions does the sampler take?
  8. **Gather (`tld4`).** The order of the four returned texels, and which address mode applies.
  9. **Pitch-linear images.** CUDA textures over pitched device memory (`CU_RESOURCE_TYPE_PITCH2D`): does the
     G14 descriptor take a linear layout with an arbitrary row stride, and with what stride alignment? CUDA
     guarantees 32-byte pitch alignment. Also: 1D linear textures larger than the maximum image width.
  10. **Formatted surface stores** (`sust.p`): rounding of float → unorm / snorm and saturation of out-of-range
      integers by `image_write`.
- **Deliverable:** for each probe, the hardware result next to the reference's, and which descriptor and sampler
  fields set the behaviour. ZLUDA then uses the native path for the modes that match and keeps software for the rest.

## B30. Forward progress of threadgroups that spin on device memory (NEW, safe except where marked)

- **Done 2026-10-20 (from the B5 probe, which spins with a device atomic per iteration):** non-resident workgroups
  start only after a spinner exits, so spinners at full residency block a later setter forever. ZLUDA must cap
  TMEM-using CTAs to the B5 co-resident count. Evidence: playground `results/2026-10-20-b5-coresidency`.

- **Why it matters:** ZLUDA emulates the `tcgen05` tensor memory (256 KiB per CTA) with a pool of four slabs in
  device memory (`ptx/lib/zluda_ptx_impl_agx_cvt.c`, ZLUDA branch `claude/bold-davinci-46xi9q`). A CTA whose first
  `tcgen05.alloc` finds no free slab spins on a device word (atomic compare-and-swap in a loop) until a CTA that
  holds one frees it. That is only correct if a spinning threadgroup cannot keep the holder from running: the
  holder may be on the same core, or waiting for a core the spinners occupy. The `mbarrier.try_wait` loops of the
  TMA and `tcgen05.commit` paths depend on the same guarantee within a threadgroup.
- **Probes** (one compute kernel each; device memory zeroed; a 10 s timeout per run):
  1. Grid of 2 threadgroups x 32 threads: threadgroup 1 spins until a device word is 1; threadgroup 0 sets it
     after 10^6 iterations of a loop. Completes?
  2. As 1 with the grid at the measured maximum residency (all cores full of spinners) plus one setter launched
     last (highest threadgroup index). Completes, and after how long?
  3. As 2 with the spin loop containing a device atomic on every iteration (as ZLUDA's) versus a plain load.
  4. **UNSAFE if 1-3 hang, run last:** what recovers the GPU (the timeout path of B13 / safe-11)?
- **Deliverable:** completes / hangs and the time for 1-3. If 2 hangs, ZLUDA bounds the TMEM-using CTAs in flight
  at launch (the runtime can split the grid) instead of spinning; record the residency limit it should use.

## B10. Memory ordering (COVERED by safe-10)

The backend and ZLUDA lower PTX `fence.{cta,gpu,sys}`, `membar.*`, `ld.acquire`, `st.release`, `atom.*.acq_rel` from
safe-10's results. Once safe-10's litmus tests are done, please also list the exact instruction sequence each scope
needs. Add the `.sys` scope too, which combines safe-10 with B7.

## B11. Numeric semantics (COVERED by safe-01, open items there)

- **Done 2026-10-21:** `.neg` / `.abs` are arithmetic: any NaN -> canonical 0x7fc00000 (0x7e00 for f16), fp32
  denormals flush to a zero with the new sign, f16 denormals kept (playground `results/2026-10-21-b11-neg-abs-modifiers`):
  non-ftz `fneg` / `fabs` must stay integer bit ops. `fma` is fused (`2026-10-11-float-semantics`). Compares read
  denormals as +-0, so -0x007fffff < +0x00400000 is false and both equal 0.0: flush min/max operands only via the
  integer-key fix-up (`2026-10-18-fcmp`, `2026-10-19-setp-extra`). rcp / rsqrt: B46; div: `2026-10-20-fdiv32`.

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
- the ULP error of the hardware `rcp` / `rsqrt` / `div` forms over the fp32 range;
- `fcmpsel lt` and `fcmpsel eq` on denormal operands under the default flushing mode: is -0x007fffff < +0x00400000,
  and does each compare equal to 0.0? The generic `minimumnum` expansion picks its operand with `lt` and then fixes
  up signed zeros by bit pattern. If the compares flush, `min.ftz.f32(-0x007fffff, +0x00400000)` gives +0 where PTX
  gives -0 (Sail PTX difftest `gen-float-3` k22). The answer decides whether the backend flushes min/max operands
  first.

## B12. Textures and surfaces (COVERED by safe-05; see B29 for what replaces the software path)

Needed by PTX `tex` / `tld4` / `txq` / `suld` / `sust` and CUDA texture / surface objects. The consumers need the
G14 texture and sampler descriptor layouts and the image memory layouts (linear and twiddled) that safe-05 records.

Status (2026-10-20): ZLUDA now samples textures **in software** (ZLUDA `ptx/lib/zluda_ptx_impl_agx_tex.c`). The
texture, sampler and surface objects are descriptors in PTX-formalization's encoding, which `cuTexObjectCreate` /
`cuSurfObjectCreate` write. CUDA arrays are dense device allocations: levels one after another, each holding all
layers. All PTX-formalization texture and surface tests pass on the simulator. The software path is slow: every
texel fetch and filter step is integer code plus software fp64. So native sampling is the goal, and B29 lists what
deciding it needs.

## B13. Occupancy, clocks, submission (COVERED by safe-04 / safe-11)

Consumed as is:
- ~55 µs warm launch;
- 64 in-order commands per SUBMIT;
- at most 4 concurrent commands GPU-wide (so CUDA streams beyond 4 serialise);
- priority has no effect;
- the in-shader clock skew and cost (`clock()`, `clock64()`, `%globaltimer`).

## B31. Cost of instructions issued with an empty execution mask (NEW, safe)

- **Done 2026-10-20:** masked ALU (`iadd`, `fmadd32`, `simd_shuffle`) costs about as little as a jump over it (~1-4 ns
  per instruction); masked `device_load`+`wait` ~13-20 ns (active ~37-41), masked `device_store` ~9-12 ns (more than
  active; writes nothing), masked `threadgroup_load`+`wait` ~13-16 ns. Drop the jump only over ALU-only bodies.
  Evidence: playground `results/2026-10-20-b31-empty-mask-cost`.

- **Why it matters:** f50bd88 measured a taken `jmp_exec_none` at 11.48 cycles per group per step against 3.78
  untaken. Mesa leaves out the jump over short blocks; the LLVM AGX backend could do the same for if/else bodies
  of at most N instructions. Under the current cost model that is a net loss: agx-bench G14C cycles total
  38579 -> 40221, `agx_sgemm_tile64` 7017 -> 7804, `saxpy` 41 -> 116. The model charges instructions issued with
  no active thread their full latency, including `device_load` + `wait`. Whether the hardware does that is not
  measured.
- **Probes (one SIMD group, all threads inactive via a false `if_icmp`, no jump):** a run of 8 and 32 of each of
  `iadd`, `fmadd32`, `device_load` + `wait 0`, `device_store`, `threadgroup_load` + `wait`, `simd_shuffle`, then
  `pop_exec`. Compare with the same bodies skipped by `jmp_exec_none`, and with all threads active.
- **Deliverable:** cycles per instruction class with an empty mask (does a masked-off load issue a memory request,
  and does its `wait` stall?). The backend then picks the largest N (by class) for which no jump is cheaper, and
  the cost model gets the same rule.
- **Safety:** ordinary ALU and memory instructions under a false condition; see `safe-probes.md`.

## B32. Asynchronous submit cost: one command per SUBMIT against batches (NEW, safe)

- **Done 2026-10-20:** ~20-27 us host cost per command either way (a 64-command SUBMIT takes ~1-1.8 ms in the
  kernel); batching cuts the firmware gap (2.6-11 vs 16-22 us) but saves at most ~20% per launch. 1,000 launches of a
  non-atomic counter on one queue are exact with every barrier policy including all-NONE: commands never overlap and
  see each other's writes. Keep one SUBMIT per launch. `drm_compute.c` `AGX_TRAIN*`. Evidence: playground
  `results/2026-10-20-b32-submit-batching`.

- **Why it matters:** zluda_agx submits one compute command per `cuLaunchKernel` (`agx_runtime` `Queue::submit_compute`,
  `cdm_barrier = 0`, a timeline point per command), without waiting. `2026-10-16-commands-per-submit` showed 64
  commands in one SUBMIT finishing in 0.55 ms, against ~7 ms as 64 *synchronous* submits. What a back-to-back
  *asynchronous* submit costs (ioctl plus firmware scheduling) is not measured. Batching the launches of a stream
  into one SUBMIT changes the fences (one point per batch) and needs a flush at every synchronization point. It is
  worth it only if the per-submit cost is large against a short kernel.
- **Probes:** the control kernel (5.6 µs) launched N = 1, 8, 64, 512 times:
  1. one SUBMIT per launch, no waits, then one wait for the last point;
  2. batches of up to 64 commands per SUBMIT (`cdm_barrier = NONE` after the first);
  3. like 2 with chained barriers.

  Record the host wall time, the firmware first-start to last-end span, and the gap between consecutive commands
  (start of k+1 minus end of k) for each. Also record the CPU time of each `DRM_IOCTL_ASAHI_SUBMIT` call.
- **Also, for correctness without barriers:** command k+1 reads what command k wrote (a dependent pair, 1,000
  times). Does `cdm_barrier = NONE` still order memory between consecutive commands of one queue?
- **Deliverable:** µs per launch for 1-3 at each N, and whether NONE keeps memory order. zluda_agx then batches
  (with NONE or a chained barrier) if case 1 costs more than a few µs per launch.
- **Safety:** the pinned control kernel and existing submit paths; see `safe-probes.md` (safe-04 used the same
  setup).

## B33. Alignment of 64- and 128-bit threadgroup and device accesses (NEW, safe)

- **Done 2026-10-20:** i32 `xy` / `xyzw` device and threadgroup loads and stores at byte offsets 0/4/8/12 mod 16 all
  match the byte-address model (32/32 lanes each), also across 64-byte lines and 4 KiB pages. 4-byte alignment is
  enough; enable the vectorizer. Evidence: playground `results/2026-10-20-b33-vector-alignment`.

- **Why it matters:** PTX's scalar `ld.shared.f32` / `ld.global.f32` promise 4-byte alignment only. LLVM's
  load/store vectorizer can merge neighbouring 32-bit accesses into one `xy` (or `xyzw`) access, but only if the
  backend knows such an access is legal at that alignment. In the CUDA samples matrixMul, each thread reads a 16-float
  row of the A tile from threadgroup memory: 16 loads now, 8 or 4 merged. The Sail model and `agx_run` take
  threadgroup and device addresses as plain byte addresses (`base + 4 * size * index + size * k`) with no
  alignment rule. The hardware's rule is not measured.
- **Probes (one threadgroup, known data):**
  - `threadgroup_load` / `threadgroup_store` with format i32 and masks `xy` and `xyzw`, at byte bases 0, 4, 8 and 12
    (mod 16), and with format i16 at bases 0 and 2;
  - the same for `device_load` / `device_store`, at addresses 0, 4, 8 and 12 mod 16, plus one access that straddles
    a 64-byte line and a 4 KiB page.
- **Deliverable:** for each case, whether the values match the byte-address model, are rounded down, or fault. The
  backend then allows 4-byte-aligned vector accesses where the hardware does and enables the vectorizer.
- **Safety:** ordinary loads and stores inside allocated buffers; see `safe-probes.md`.

## B34. Stack (scratch) load / store cost and the scoreboard wait groups (NEW, safe once B1 works)

- **Done 2026-10-20:** two scoreboard groups (one encoding bit); `wait N` waits for group N only and a register read
  before its wait returns the old value (simulator fixed to match). Dependent `stack_load`+wait ~33 ns = an L1-hit
  device load; 16 independent stack loads ~3 ns each behind one wait; `stack_store` ~2 ns. Evidence: playground
  `results/2026-10-20-b34-stack-and-waits`.

- **Why it matters:** cuBLAS's `agx_sgemm_tile64` runs at the 128-GPR limit and spills, with 46 spill accesses per
  K-loop iteration. Costed with #7's functional path (one 64x64 tile, K = 256, 32 K steps: 123,937 cycles),
  every spill reload is a `stack_load` followed at once by `wait 0`. About a dozen such pairs in a row each stall
  ~2,000 cycles, which is the cost model's guess: `stack_load` / `stack_store` are unmeasured classes. The LLVM
  AGX backend tracks one scoreboard slot (`AGXInsertWaits`: every load in group 0, at most eight pending). The
  Sail model's `wait` completes every pending load whatever its group. The cost model does track loads' `g`
  field per slot.
- **Probes (after B1's scratch setup works):**
  1. Latency and throughput of `stack_load` / `stack_store` (i32, and the vector masks), against `device_load`
     from L1 and DRAM: a dependent chain of 64 stack loads, and 64 independent ones before a single wait.
  2. Wait groups: issue a slow `device_load` (DRAM miss) with `g = 0` and a `stack_load` (or an L1-hit device
     load) with `g = 1`, then `wait 1`. Does the wait return before the slow load? How many groups exist (try
     `g` = 0..7)? Does `wait N` cover only group N, or every group up to N?
- **Deliverable:** stack access cycles, and the wait-group semantics. The backend then puts spill reloads (and
  independent loads) in their own groups and batches reloads behind one wait. The cost model and the Sail model
  get the measured rules.
- **Safety:** ordinary loads and waits; stack accesses only once B1's scratch configuration is established; see
  `safe-probes.md`.

## B35. A clock a shader can read: `%clock`, `%clock64`, `%globaltimer`, `nanosleep` (NEW, safe)

- **Done 2026-10-21: no SR is a clock.** All 256 selectors read twice 0.96 ms apart in one thread: only SR113 (the
  power-state bit) changed. A shader can read time only through the CPU-written `CNTVCT_EL0` timer (f59d + coherent
  load, 41.7 ns ticks, 0.6-1 us per read; `novel-results/2026-10-11-in-shader-clock`, `2026-10-18-clock-load`): use it
  for `%globaltimer`, a coarse `clock64()` and a bounded `nanosleep`; there is no cycle counter. Evidence: playground
  `results/2026-10-21-b35-shader-clock`.

- **Why it matters:** ZLUDA reads `%clock`/`%clock64` as 0 and `nanosleep` does not sleep, because no timer the
  shader can read is known. CUDA code uses `clock64()` for timing loops, spin-wait backoff and benchmarks
  (e.g. the CUDA samples' `clock`), and `%globaltimer` for wall time.
- **Probe:** read every special register the decoder knows (and the unknown SR numbers next to them) twice in one
  thread, with a bounded delay loop between the reads, and record which change, by how much per iteration, whether
  they agree across SIMD-groups and cores, and whether one matches the 1 GHz command timestamps (B25).
- **Deliverable:** the SR (if any) of a cycle counter and of a global timer, its rate and width; or that none
  exists, in which case ZLUDA keeps 0 and documents it.
- **Safety:** reads only; one threadgroup, bounded loop.

## B36. Sharing GPU memory across processes and with the host (NEW, safe)

- **Done 2026-10-21:** dma-buf IPC works: an object exported with PRIME_HANDLE_TO_FD, imported in a forked process with its
  own fd and VM, binds like any other; GPU and CPU accesses on both sides agree after completion. No user-pointer
  import exists (11 ioctls, no `/dev/udmabuf`), so `cuMemHostRegister` must copy. Evidence: playground
  `results/2026-10-21-b36-b45-shared-memory` (agx_runtime patch there, not pushed).

- **Why it matters:** ZLUDA has no `cuIpcGetMemHandle`/`cuIpcOpenMemHandle` (used by multi-process training and
  NCCL-style code) and `cuMemHostRegister` cannot map existing host memory into the GPU.
- **Probes:**
  1. Export an Asahi GEM object as a dma-buf (`DRM_IOCTL_PRIME_HANDLE_TO_FD`), import it in a second process
     (`PRIME_FD_TO_HANDLE`), bind it in that process's VM and run a kernel that reads what the first wrote.
  2. Whether the Asahi UAPI accepts any user-pointer import (it is not expected to); record the error.
- **Deliverable:** whether IPC can be built on dma-buf, and what binding an imported object needs.
- **Safety:** ordinary allocation and binding; two processes.

## B37. Checked mode on the hardware: cost and the early return (NEW, safe)

- **Why it matters:** ZLUDA's checked mode (`ZLUDA_AGX_CHECKED=1`, LLVM `agx-checked`) is validated only on this
  simulator, which estimates about 1.5x the cycles of an unchecked build. Once a kernel reports a bad access, later
  launches of the module must see the record's `claimed` word and return at once.
- **Probes:** `llvm/utils/agx-checked/zluda_tests.py`'s kernels built checked and unchecked through `agx_run_hw`:
  1. the same records (kind, address, thread, block) as in the simulator for every bug case;
  2. kernel time checked against unchecked for the in-bounds kernels and agx-bench's set;
  3. a second launch after a reported error, without zeroing the record, does no store.
- **Deliverable:** pass/fail per case, the time ratios, and whether (3) holds.
- **Safety:** checked kernels never touch the bad address; the unchecked builds of the bug cases do (soft faults,
  B20), so run them last.

## B38. Several VMs and queues in one process (NEW, safe)

- **Done 2026-10-21:** no limit reached: 2,048 VMs (1 queue each) or 2,048 queues on one VM in one process, all
  alive, every async launch correct. VMs share the 63 hardware slots (bound only while they have work). Priorities:
  only LOW / MEDIUM accepted, no effect (safe-04). Evidence: playground `results/2026-10-21-b38-vms-queues`.

- **Why it matters:** ZLUDA has one context; real CUDA contexts (each with its own allocations and streams) may get a
  VM each, and every stream is a queue. The limits are not known.
- **Probes:** in one process, create 2, 4, 16 VMs and 1, 16, 64, 256 queues (on one VM and spread), run a kernel on
  each, and record which creations fail and with what error; also queue priorities (LOW..REALTIME) and whether a
  higher-priority queue's short kernel finishes before a running lower-priority long one
  (`cuStreamCreateWithPriority`).
- **Deliverable:** the limits, and whether priorities change scheduling.
- **Safety:** creation and small kernels only.

## B39. Largest allocation and address range (NEW, safe)

- **Done 2026-10-21:** no limit below physical memory: one 32 GiB object and 40 GiB in total bound and were written
  by the GPU at their last page (stopped at a 10 GiB MemAvailable floor on purpose). VM: 512 GiB, ~412 GiB of data VA.
  The UAPI reports no memory figures: report free = MemAvailable, total = MemTotal. Evidence: playground
  `results/2026-10-21-b39-allocation-limits`.

- **Why it matters:** `cuMemAlloc` of tens of GiB on a 64 GiB M2 Max, and `cuMemGetInfo` (ZLUDA reports a fixed
  share of RAM).
- **Probes:** allocate and bind single BOs of 1, 4, 16, 32 GiB and a kernel that touches the last page; the total
  across many BOs until failure; record what the DRM reports as available.
- **Deliverable:** the largest single and total allocation, the error at the limit, and a free-memory figure the
  runtime can report.
- **Safety:** may push the machine into memory pressure at the limit; stop at the first failure.

## B40. The simulator's cost model against the hardware on whole library kernels (NEW, safe)

- **Why it matters:** ZLUDA's library kernels (agx_libs: sgemm, hgemm, conv, fft, sparse, dnn) and backend choices are
  tuned with the simulator's cycle estimates; f64 is emulated in software and costed only by the model.
- **Probe:** run `agx_libs/kernels/test_kernels.py`'s kernels through `agx_run_hw` with timestamps, and the same
  launches in `agx_cost`; also the f64 add/mul/fma/div kernels of `ptx/tools/agx_f64_check.py`.
- **Deliverable:** measured time against estimated cycles per kernel (the ratio, and the worst outliers).
- **Safety:** ordinary kernels.

## B41. Prebuilt CUDA programs instead of nvcc on the Asahi machine (FOLLOWS B23)

- **Why it matters:** B23 is blocked because nvcc is not wanted on the Asahi machine. CUDA binaries built elsewhere
  (x86 or aarch64 Linux with a CUDA toolkit) carry PTX in their fat binaries, which ZLUDA compiles.
- **Probe:** run aarch64 Linux builds of the CUDA samples (vectorAdd, matrixMul, reduction, simpleCUBLAS,
  simpleCUFFT, cuSolver-free ones) with `LD_LIBRARY_PATH` pointing at ZLUDA's libraries, `-gencode` with
  `compute_XX` PTX included; record pass/fail and the first failing call (`ZLUDA_AGX_LOG=1`).
- **Deliverable:** the per-sample results; this answers B23.
- **Safety:** as B23.

## B42. `cudaDeviceReset` and context teardown on the hardware (NEW, safe once ZLUDA has real contexts)

- **Done 2026-10-21 (on ZLUDA master 7a6d979, real contexts; two bugs found and fixed locally):** 1,000 create /
  load / launch / destroy cycles, 100 destroys with 200 launches queued (destroy waits, ~1.8 ms) and 100 primary resets
  all pass with no leak, after fixing (1) a kernel shmem leak that survives process exit (~5 MiB per context: the VM
  was destroyed before the objects bound in it) and (2) faults with 45+ queued launches (stale pipeline words in the
  USC ring without the USC invalidate; Mesa always emits it). Patch and evidence: playground
  `results/2026-10-21-b42-context-teardown`; ZLUDA notes 10-11.

- **Why it matters:** ZLUDA is adding real contexts (destroy frees allocations, modules and queues;
  `cuDevicePrimaryCtxReset`). Freeing queues and VMs with work recently finished, and reusing USC memory after a
  module unload, must not leave stale state.
- **Probe:** 1,000 cycles of create context, load a module, launch, synchronise, destroy; then the same with a
  launch still running at destroy (CUDA waits for it).
- **Deliverable:** no wrong results, no leaked GPU memory (DRM's reported usage stable), no hang.
- **Safety:** ordinary work; watch memory.

## B43. How commands of different VMs share the GPU (NEW, safe)

- **Done 2026-10-21:** commands of two VMs run concurrently on the same cores, interleaved by workgroup slot: two
  full-GPU 0.84 s kernels started together both took 1.49 s and ended 26 ns apart; no per-VM time slicing, no
  preemption (a short kernel runs when a slot frees), no visible switch cost. SR21: one slot per VM, 63 recycled.
  Evidence: playground `results/2026-10-21-b43-vm-scheduling`.

- **Why it matters:** the simulator is getting several contexts (VMs) that run together, and ZLUDA real CUDA contexts
  (one VM each, or one VM shared). How the firmware schedules compute commands of different VMs decides both the
  model and whether separate VMs cost throughput.
- **Probes** (B38's setup: two processes or two VMs in one process, each with its own queue):
  1. a long kernel (B6's loop, ~1 s) on VM A, then a short one on VM B: does B's finish before A's (concurrent or time
     sliced) or after (serialized)? Repeat with the two on one VM, two queues.
  2. two long kernels that each fill every core: total time against one alone (core sharing or slicing), and the
     timestamps of each.
  3. SR21 (context) read by kernels of each VM, with 2, 4, 16 VMs alive: which values appear, and whether they are
     reused after a VM is destroyed.
- **Deliverable:** overlap or not, the slice length if slicing, SR21 per VM, and the cost of switching (the gap in the
  timestamps).
- **Safety:** ordinary bounded kernels.

## B44. Isolation between VMs when one faults or deadlocks (NEW, unsafe: needs one fault and one deadlock)

- **Why it matters:** a context's fault must not disturb another's results (CUDA keeps errors per context), and the
  simulator's multi-context mode needs the hardware's behaviour as its default.
- **Probes:**
  1. VM A stores to an unmapped VA (soft fault, B20) while VM B runs a checksum kernel: B's result and timestamps,
     A's fence and timestamps.
  2. VM A deadlocks at a barrier (B26's case, ended by the firmware after ~0.5 s) while VM B runs a stream of short
     kernels: are B's delayed during those 0.5 s, and are its results right?
  3. Same as 1 and 2 with A and B on two queues of one VM.
- **Deliverable:** whether faults and deadlocks stay within their VM (or queue), and the delay others see.
- **Safety:** one soft fault and one firmware-ended deadlock, as in B20 and B26.

## B45. Memory visibility between VMs sharing one buffer object (NEW, safe; FOLLOWS B36)

- **Done 2026-10-21:** one object in two VMs: a reader on VM B sees all of VM A's writes after a host wait or a
  syncobj wait on A's point (~130 us hand-over), and none when submitted unordered (the two run concurrently). Order
  between VMs is only what syncobj waits give. Evidence: playground `results/2026-10-21-b36-b45-shared-memory`.

- **Why it matters:** contexts sharing memory (CUDA IPC, peer-style access within one process, a library context next
  to the application's) bind the same GEM object in two VMs.
- **Probes:** bind one BO in VMs A and B at different VAs; A writes, B reads in a later command (and in a concurrent
  one with B7's coherent loads); record what B sees and whether ordering between the two queues needs a fence
  (syncobj wait) or is implied.
- **Deliverable:** the ordering rule between VMs, so ZLUDA and the simulator synchronize the same way.
- **Safety:** ordinary loads and stores.

## B46. Worst-case relative error of `rcp` and `rsqrt` (NEW, safe)

- **Done 2026-10-21:** rcp max **0.7496 ulp** (rel 2^-23.42), rsqrt / rsqrt_special **0.8177 ulp** (rel 2^-23.42) over
  every significand (the recorded [1, 2) / [1, 4) tables), and the same in every binade (776,195 GPU inputs, 0
  mismatches against the scaled tables). ~19 % of results are 1 ulp off correctly rounded; the simulator already uses
  the measured tables. The backend's 2^-19.7 margin holds with ~9 bits to spare. Denormal inputs give +-inf; results
  below 2^-126 flush to +-0. Evidence: playground `results/2026-10-21-b46-rcp-rsqrt-error`.

- **Why it matters:** the backend's software `double` division and square root (llvm-project d3c8d974d) seed a
  fixed-point Newton iteration with `rcp` / `rsqrt` on a `float` of 24 significant bits in [2^23, 2^24) and stay
  bit-exact while the estimate is within about 2^-19.7 (some 10 ulp) of the exact value. The `float` division of
  AGXStrictFP also assumes "a few ulp". The simulator models both as correctly rounded.
- **Probes:** every `float` in [1, 4) (2^24 values: covers every significand and both exponent parities) through
  `rcp` and `rsqrt`; record the maximum error in ulp against an exact reference and the inputs where it occurs.
  Repeat for a sample of exponents and for denormal inputs.
- **Deliverable:** the two maximum errors (and whether they depend on the exponent), so the simulator can model the
  real results and the backend's margin is confirmed or widened.
- **Safety:** arithmetic only.

## B47. Shift amounts of 32 to 127 in `bfi`, `bfeil`, `extr` and `asr` on G14 (NEW, safe)

- **Done 2026-10-21: pass.** Every form takes the low 7 bits of a register amount (amounts 0..255 tested, bits 7+
  ignored) and matches the formulas below and the simulator word for word (18 cases x 256 amounts). The backend needs
  no masking. Evidence: playground `results/2026-10-21-b47-shift-amounts`.

- **Why it matters:** since llvm-project's "64-bit values as halves" change, 64-bit shifts by a variable amount
  rely on these instructions taking a 7-bit amount, as dougallj's applegpu documents for G13 and the simulator
  models: `bfi 0, x, s` and `bfeil 0, x, s` give 0 for s in 32..127, `extr lo, hi, s` gives `((hi:lo) >> s)` for
  s up to 63 (and 64 gives 0), `asr x, s` fills with the sign bit from 32 on. A logical `lshr i64` is two `extr`s.
- **Probe:** for s in 0..127 and the values 0x80000001, 0xffffffff, 0x12345678 (and `extr` with a second register
  0x9abcdef0), run each instruction with a register amount; compare with the formulas above.
- **Deliverable:** pass, or the actual rule (amount masked to 5 or 6 bits?), so the backend can mask.
- **Safety:** arithmetic only.

## T1. `agx_cost`: uniform values from the command line (tooling request, no hardware needed; DONE in #7)

- **Status 2026-10-20:** Jianwen-Ding/AGX-Performance-Simulator#7 adds `--uniform`, the launch geometry and
  functional paths over mapped memory (`--map` / `--alloc`) to `agx_cost`. The request below is kept for the record.

- **Why it matters:** `agx_cost` runs a kernel with every uniform register zero. A kernel guarded by `tid < n`, with
  `n` a uniform, takes `jmp_exec_none` past its body. Its estimate is then the guard alone: since f50bd88 that
  is 41.2 cycles for llvm/utils/agx-bench `saxpy`, of which 34 are the taken jump. The trace engine already
  accepts uniforms (`ctx.uniforms` in `cost.rs`), but the CLI cannot set them.
- **Request:** `agx_cost code.bin --uniform U=V` (repeatable, 32-bit value into uniform pair U), and optionally
  `--threads N` for the active lanes. agx-bench would then pass the kernel arguments it knows (`n = 32` etc.) and
  measure the bodies of `saxpy`, `stencil` and `histogram`.

## Reporting

- Put the evidence in the playground as the other safe tasks do.
- Add a short "Done" note per item to this file.
- Raise new coverage cases where a probe produced a reusable kernel.
- Items B1 and B6 count against the hang budget; follow the recovery policy in `unsafe-01-call-ret.md`.
