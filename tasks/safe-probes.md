# Safe AGX probes: task index

These tasks learn more about the AGX GPU (M2 Max, G14C) **without risking a GPU hang**. Each one uses only:
- instruction encodings that have already run cleanly on this GPU (the coverage suite's `safe` cases, Mesa-compiled
  code, or forms recorded in the playground's novel-results);
- straight-line code, or loops that are bounded and checked in the emulator first;
- the direct-DRM harness (`tests/drm_compute.c` via `scripts/record_suite.run` or the playground's runners), which
  records any timeout against the hang budget; or Vulkan through Mesa.

Tasks safe-07 .. safe-12 come from the playground's direct-DRM findings and `coverage_catalog.MANUAL`.
The stack / scratch item is excluded: hand-built direct-DRM scratch programs stalled twice.

| Task | What it learns | Why it matters for CUDA |
|---|---|---|
| [safe-01](safe-01-numeric-semantics.md) | Exact behaviour of `convert`, fp16 transcendentals, subnormal edges of fp32 math | PTX `cvt` and the default (non-`.ftz`) float ops must match bit for bit |
| [safe-02](safe-02-performance-gaps.md) | Atomic contention, strided access, barriers, write-back, the fma sin/cos cost | The largest gaps between the cost model and the GPU (`results/2026-10-16-asahi-retime`) |
| [safe-03](safe-03-memory-system.md) | Line and page size, TLB reach, SLC capacity, CPU -> GPU visibility latency | Cache-aware lowering; emulating unified memory |
| [safe-04](safe-04-driver-submission.md) | Queue priorities, concurrent queues, submission latency, command limits | Streams, events and launch overhead in a CUDA runtime. **Done 2026-10-16** (playground novel-results `submission-latency`, `commands-per-submit`, `queue-priority`, `queue-concurrency`): ~55 µs launch warm / 0.4-0.7 ms after GPU wake-up; 64 in-order commands per SUBMIT; priority has no effect; at most 4 concurrent commands GPU-wide; clock ramps 3.1x over ~15 ms |
| [safe-05](safe-05-textures-via-vulkan.md) | `texture_sample` / `texture_load` semantics from Mesa-compiled shaders | CUDA texture and surface objects |
| [safe-06](safe-06-unknown-special-registers.md) | Record the 214 coverage "may fault" special-register cases | Completes the special-register map in the coverage suite. **Done 2026-10-16:** 214/214 recorded, 0 faults; 213 match the simulator, and SR113 is unmodelled by design |
| [safe-07](safe-07-fmadd-lane-offset.md) | The 10-byte fmadd lane-offset form and the fmadd32 bit-52 operation | A lane-shifted fma can replace `shfl.up` + `fma` in scans and stencils |
| [safe-08](safe-08-undecoded-and-control-opcodes.md) | The 49 non-faulting undecoded opcodes, what 0x18 waits for, the ending opcodes | Keeps the backend away from accidental encodings; possible sync and early-exit primitives. **Done 2026-10-16:** 403 coverage cases (simulator exact); 0x18 fixed ~4.4 ns, waits for nothing; 18 ending opcodes indistinguishable |
| [safe-09](safe-09-preshader.md) | Preamble: uniform file size, register budget, multi-dispatch behaviour, cost | Computing kernel-uniform values once per core |
| [safe-10](safe-10-memory-ordering.md) | SB / LB / threadgroup litmus tests, atomics as release, fence cost, ordering across dispatches | PTX `fence` / `membar` / `ld.acquire` / `st.release` mapping |
| [safe-11](safe-11-occupancy-and-clock.md) | Small-workgroup occupancy cap, imageblock SRs, cost and skew of the in-shader clock | Occupancy API, `cudaDeviceProp`, `clock()` / `%globaltimer` |
| [safe-12](safe-12-drm-interface-map.md) | Reference for the direct-DRM interface: UAPI, command words, limits, sync objects | Building blocks of the CUDA runtime (allocation, launch, streams, events). **Steps 1-4 done 2026-10-16:** playground `docs/drm-interface.md`; cross-queue syncobj waits cost ~0.1 ms |

Before starting a task, read the playground's `docs/external-knowledge.md` (a survey of published sources against our
measurements), so a task confirms or refutes known claims instead of rediscovering them.

## Rules for every task

1. **Before anything touches the GPU:**
   - `python3 scripts/gpu_safety.py status` must show budget left.
   - Check the power source (`cat /sys/class/power_supply/macsmc-ac/online`). On battery, sustained load lowers
     the clock about 2.1x (playground `novel-results/2026-10-16-battery-clock-states`). Record the power source with
     any timing.
   - Confirm the device is the AGX: a Vulkan runner must report "Apple M2 Max", not llvmpipe.
2. **No new encodings.** If a probe needs an instruction form the coverage suite has not recorded as `safe`, stop
   and list it as a follow-up. Do not try it.
3. **Whole programs only.**
   - Submit `run_drm_compute.build_main(body)`, never a bare body. A bare body has no final `stop`; one stopped the GPU
     executing work on 2026-10-16. `record_suite.run()` now refuses a program that does not end with `stop`.
4. **Bounded code only.**
   - Loops must use `agx_loop` bodies that pass their emulator gate.
   - Simulate any new kernel in `rust/target/release/agx_run` (set `AGX_SR_DEFAULT` if it reads unmodelled special
     registers) and check that it terminates and writes what you expect before it runs on the GPU.
5. **Small first.** Run one small dispatch and check its oracle before any sweep.
6. **Stop on any fault or timeout.** Note it in `coverage/faults.json` or the task's report, and do not retry blindly.
7. **Record evidence and push.**
   - Raw bulk data goes under `/scratch/agxsim-data/` (regenerable).
   - Compact evidence plus an offline check goes into git: the simulator's `results/<date>-<topic>/` and the
     playground's `novel-results/` + `results/` + a `make verify-*` target.

## Not safe: do not run

- funop functions 2, 3, 5, 7, 11 and 15 (24 confirmed GPU hangs), and convert mode 2, 3, 12+ or round field > 1.
- `call`, `ret`, `jmp_incomplete`, `jmp_if_skipping_doorbell`.
- The 9 undecoded opcodes that discard a whole dispatch (0x06, 0x24, 0x34, 0x44, 0x46, 0x54, 0x5c, 0x64, 0x74).
- Device atomic ops 11-15 and their u-bits.
- Direct-DRM helper / scratch programs (two attempts stalled).
- `while` loops whose termination is not proven (`control/while_icmp_ugte` froze the display).
- The `0x6e` stale-data forms (on-chip data leakage, the LeftoverLocals class).
- The 251 `memory_barrier a, b, c` encodings and the remaining coverage `--risky` areas, unless the owner approves
  them separately.
