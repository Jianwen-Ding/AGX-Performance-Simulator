# ZLUDA issues found by the blocker probes

Issues in Jianwen-Ding/ZLUDA (checked against `master` 7a6d979, 2026-10-21) found while running the
[ptx-re-blockers](ptx-re-blockers.md) probes on the M2 Max. Nothing here has been pushed to ZLUDA: the owner asked that
ZLUDA not be pushed from this machine. Fixes made locally are saved as patches in the playground (paths below); the
running notes are in the playground's `notes/zluda-issues.md`.

Status: **open** (in master), **fixed upstream** (master already has it), **local fix** (patch saved, not upstream).

## High priority

| # | Issue | Evidence | Status / fix |
| --- | --- | --- | --- |
| Z1 | **Destroyed contexts leak their GPU memory in the kernel, past process exit.** Rust drops fields in declaration order: `agx_runtime::Context { device, props, vm, stream, allocs }` and `zluda_agx::Driver { ctx, ... }` destroy the VM before the objects bound in it, so their unbinds fail and the kernel keeps the shmem (~5 MiB per context lifetime, until reboot). `Queue::drop` also never unbinds its timestamp object (`GEM_BIND_OBJECT` UNBIND). | B42: playground `results/2026-10-21-b42-context-teardown` (system `Shmem` +2,020 MiB after 400 context lifetimes) | **local fix**: `zluda-teardown-fix.patch` there (VM declared last in both structs; timestamp unbind in `Queue::drop`). With it, 1,000 cycles leak nothing |
| Z2 | **45+ launches queued on a stream fault** (commands end with both timestamps 0, reported as `CUDA_ERROR_LAUNCH_FAILED`). Pipeline words are staged back to back in the 256 KiB USC ring and the USC-cache invalidate is only emitted after a module load, so a later launch can read stale pipeline words the GPU cached while earlier commands ran. | B42 (same result dir): flaky from 45 queued launches, always from 64; `ZLUDA_AGX_USC_BARRIER=1` makes 200 queued launches pass | **open**: emit the USC barrier on every command (Mesa does; ~1.8 us per launch, B8) or give each launch's words a line the GPU cannot have fetched. Corrects B8's "reuse is safe without the barrier" |

## Build (aarch64 / GCC 15)

| # | Issue | Evidence | Status / fix |
| --- | --- | --- | --- |
| Z3 | `llvm_zluda` and `ptx` (hence `zoc`, the `compiler` crate) do not build on aarch64: FFI strings typed `*const i8` / `Vec<i8>` where `c_char` is `u8` (`llvm_zluda/src/ffi.rs`, `compile_agx.rs`, `utils.rs`, `ptx/src/pass/llvm/emit.rs`). (`zluda_agx` / `format` were already fixed upstream.) | playground `notes/local-patches/zluda-aarch64-c_char.patch` | **local fix**: use `std::ffi::c_char` |
| Z4 | Building `compiler` needs `git submodule update --init ext/HiGHS`; the failure (`highs-sys` cmake) does not say so. | | **open** (documentation) |
| Z5 | The LLVM AGX fork (branch `claude/bold-davinci-46xi9q` 3736ce1) does not build with GCC 15: missing includes (`StringExtras.h`, `IRBuilder.h`, `SmallVector.h`, `<cmath>`). | playground `notes/local-patches/llvm-agx-gcc15.patch` | **local fix** (LLVM fork, not ZLUDA) |

## Runtime behaviour to change

| # | Issue | Evidence | Status / fix |
| --- | --- | --- | --- |
| Z6 | `agx_run_hw` sets up no scratch memory, so kernels that use the stack fault there (`agx_sgemm_tile64`: 17 `stack_*` instructions; 2 of 819 agx_libs cases). zluda_agx itself sets scratch up correctly (B1). | B40: `results/2026-10-21-b40-cost-model-vs-hw`; B1: `results/2026-10-21-b1-zluda-scratch` | **open**: give `agx_run_hw` the module path's scratch setup, or refuse stack-using code |
| Z7 | `cdm_dispatch` asserts that the global size is a multiple of the local size; the hardware runs partial workgroups exactly (all dimensions). | B2a: `results/2026-10-21-b2a-partial-workgroups` | **open** (harmless for CUDA, whose grids are whole blocks; drop the assert for Mesa-style launches) |
| Z8 | No IPC / host registration: `cuIpcGetMemHandle` / `cuIpcOpenMemHandle` are missing; dma-buf export / import works on the hardware. There is no user-pointer import in the UAPI, so `cuMemHostRegister` must copy. | B36 / B45: `results/2026-10-21-b36-b45-shared-memory` | **local fix** for the runtime half: `agx_runtime-dmabuf.patch` there (`Bo::export_dmabuf`, `Vm::import_dmabuf`); cross-VM order needs a syncobj wait |
| Z9 | Only LOW and MEDIUM queue priorities exist (HIGH / REALTIME give EINVAL) and they change nothing; `cuCtxGetStreamPriorityRange` should report [0, 0]. | B38: `results/2026-10-21-b38-vms-queues` | **open** (check the current range) |
| Z10 | `cuMemGetInfo`: the UAPI reports no memory figures; use MemAvailable (and the cgroup's `memory.max`) for free, MemTotal for total. New objects are CPU-zeroed (~0.12 s per GiB), which `cuMemAlloc` does not require. | B39: `results/2026-10-21-b39-allocation-limits` | **open** |
| Z11 | `%clock` / `%clock64` / `%globaltimer` / `nanosleep`: no special register counts time; a CPU-written `CNTVCT_EL0` timer read with f59d + a coherent load works (0.6-1 us per read, one line per core group). | B35: `results/2026-10-21-b35-shader-clock` | **open** (currently reads 0) |
| Z12 | Non-ftz `neg.f32` / `abs.f32` must stay integer bit operations: the `.neg` / `.abs` source modifiers canonicalize NaNs and flush fp32 denormals. | B11: `results/2026-10-21-b11-neg-abs-modifiers` | check the lowering (backend / ZLUDA) |
| Z13 | `mma.sync` / `agx.mma` can use the native `simd_matrix_fmadd32/16` on G14: same lane layout as `agx.mma`, an in-order fma chain over k, ~3x scalar fma throughput. The lane-offset fmadd works for every k inside a segment; fp32 boundary lanes are undefined. | B15: `results/2026-10-21-b15-simd-matrix`; B18: `results/2026-10-21-b18-fmadd-lane` | **open** (backend lowering) |
| Z14 | Native textures: depth compare (`ref OP texel`, PCF), textureGrad LOD, MIN/MAX_LOD, unnormalized coordinates and pitch-linear images (any 16-byte-multiple pitch) all match CUDA; image stores round / saturate as CUDA does. | B29: `results/2026-10-21-b29-texture-corners` | **open**: move these modes off the software path |
| Z15 | Earlier notes (2026-10-20, still relevant): faults and illegal instructions report fence success with zero timestamps (handled upstream); soft faults drop out-of-bounds stores silently; intra-warp spins deadlock (B4); cooperative launches must cap co-resident CTAs (B5 / B30); local sizes over 1,024 run nothing and report success (B22). | playground `notes/zluda-issues.md` items 3, 5-8 | zero-timestamp check **fixed upstream**; the rest **open** / by design |
| Z16 | Uniform words of 64 halves (B9). | `results/2026-10-20-b9-uniform-preload` | **fixed upstream** (`UNIFORM_WORD_HALVES = 64`) |
| Z17 | Scratch / helper program support (B1). | `results/2026-10-20-b1-direct-drm-scratch` | **fixed upstream**; verified on the GPU through zluda_agx (B1, 262,144 threads) |
