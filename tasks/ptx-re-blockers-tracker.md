# Task ptx-re-blockers-tracker: every blocker in one checklist

Tracks the items of [ptx-re-blockers.md](ptx-re-blockers.md) until each is done or recorded as blocked. The details,
probes and deliverables stay in that file; this one only says where each item stands and what is left. Update the
row (and the item's "Done" note in ptx-re-blockers.md) when an item moves.

Status: **done**, **open** (safe, can run now), **needs tools** (safe, but a missing build or program stops it),
**unsafe** (needs the owner's approval and the hang budget), **blocked** (needs hardware or a decision we do not have).

Rules: [safe-probes.md](safe-probes.md). Before any GPU work: `scripts/gpu_safety.py status`, the power source,
`free -g` >= 8 GB available, and an `agx_run` emulator gate. Evidence goes in the playground's `results/`.

## Environment on 2026-10-21

- Toolchain: `source /scratch/agxsim-deps/env.sh` (Rust in `/scratch/rust`, Mesa headers, venv). No root.
- ZLUDA: `/scratch/ZLUDA` (master 8357395 plus a local aarch64 `c_char` patch). The owner said **do not push ZLUDA**;
  ZLUDA issues go to the playground's `notes/zluda-issues.md`.
- Not on this machine: the LLVM AGX build (`llc`, `zoc`), nvcc (the owner does not want it installed), an M2 Ultra,
  a G13X device.
- Hang budget: 0/30 on boot 1178a36a.

## Priority 1: nothing runs without them

| Item | Status | Left to do |
| --- | --- | --- |
| B1 scratch / stack | done | drm_compute and ZLUDA's own scratch path both pass on the GPU (262,144 threads). |
| B2 hardware runner | done | Sub-items below. |
| B2a partial workgroups | done | Partial workgroups run in every dimension (exact thread counts); `results/2026-10-21-b2a-partial-workgroups`. |
| B2b Uniform word size | done (B9) | 64 halves per word work; `agx_runtime` still splits at 32 (ZLUDA note, not pushed). |
| B2c Shared "none" encoding | done | Existing occupancy data used that word: 96 groups / ~55 workgroups per core, no limit. |

## Priority 2: correctness of common CUDA idioms

| Item | Status | Left to do |
| --- | --- | --- |
| B3 64-bit atomics | done | Contention atomic, RNE + FTZ, canonical NaN; u64 min/max returns only a changed flag. |
| B4 intra-SIMD-group progress | done | |
| B5 co-residency | done | |
| B7 CPU → GPU visibility | done | |
| B30 spinning threadgroups | done | |

## Priority 3: limits and errors a runtime reports

| Item | Status | Left to do |
| --- | --- | --- |
| B6 dispatch time limit | done | |
| B8 code region | done | |
| B9 parameter space | done | |
| B20 failed-job reporting | done | |

## Covered by other tasks

| Item | Status | Left to do |
| --- | --- | --- |
| B10 memory ordering | done | Sequence table per scope written (IRIW still untested). |
| B11 numeric semantics | done | Modifiers canonicalize NaN and flush fp32 denormals; rest answered by existing findings. |
| B12 textures | covered | See B29. |
| B13 occupancy, clocks, submission | covered | |

## Compiler targets and ISA corners

| Item | Status | Left to do |
| --- | --- | --- |
| B14 calls and recursion | done | All unsafe-01 call/ret cases measured; simulator replays 570 programs exactly. |
| B15 `simd_matrix_fmadd` | done | Layout = agx.mma layout; in-order fma chain; 16 cycles; ~3x scalar fma throughput. |
| B16 0x78 barrier | done | |
| B17 `async_load` / `copy_*` | unsafe | Low priority; needs B1's helper setup. Earlier attempts hung. |
| B18 lane-offset fmadd | done | 28 recorded exact + 6 via probe_variants (fp32 segment-boundary lanes undefined); 1.75x latency vs shuffle+fma. |
| B19 bit 45 on M2 Ultra | blocked | No G14D. Record that; the backend treats G14D atomics as unverified. |
| B21 `trap` | done | |
| B22 launch limits | done | |
| B24 atomic / barrier corners | done 1-4 | 5: unsafe `memory_barrier` sweep (owner approval). 6: six-byte low-`0b11` opcodes with new operand bytes. 7: blocked (no G13X). |
| B25 telemetry | done | |
| B26 registers vs threadgroup size | done | |
| B27 float-to-int convert | done | |
| B28 barrier after exit | done | |
| B31 empty-mask cost | done | |
| B32 submit batching | done | |
| B33 vector alignment | done | |
| B34 stack cost and wait groups | done | |
| B35 shader clock | done | No SR clock; use the CPU-written timer; `results/2026-10-21-b35-shader-clock`. |
| B46 `rcp` / `rsqrt` error | done | 0.75 / 0.82 ulp max, exponent-independent; `results/2026-10-21-b46-rcp-rsqrt-error`. |
| B47 7-bit shift amounts | done | Low 7 bits of the amount, as modelled; `results/2026-10-21-b47-shift-amounts`. |

## Textures

| Item | Status | Left to do |
| --- | --- | --- |
| B29 native sampling | mostly done | Probes 4, 9: run `scripts/run_vk_b29.py` (written, built); simulator texture unit still lacks compare / LOD clamps / pixel coords. |

## Runtime, VMs and memory

| Item | Status | Left to do |
| --- | --- | --- |
| B36 dma-buf IPC | done | dma-buf IPC works; no userptr import. |
| B38 VMs and queues per process | done | No limit up to 2,048 VMs / 2,048 queues; 63 VM slots recycled; only LOW/MEDIUM priority. |
| B39 largest allocation | done | 32 GiB single / 40 GiB total fine; free = MemAvailable. |
| B42 context teardown | done | Pass after 2 fixes (teardown leak; USC barrier needed for deep queues). |
| B43 scheduling between VMs | done | Concurrent, slot-interleaved, no slicing or preemption. |
| B44 isolation of faults and deadlocks | unsafe | One soft fault and one firmware-ended deadlock (as B20 / B26). |
| B45 shared BO between VMs | done | Syncobj wait needed and sufficient; no implicit order between VMs. |

## Whole programs and tooling

| Item | Status | Left to do |
| --- | --- | --- |
| B23 CUDA runtime program | blocked | The owner does not want nvcc on this machine. |
| B37 checked mode on hardware | done | 17/17 + 7 controls on GPU; 1.56-1.59x; claimed record stops later launches. |
| B40 cost model against hardware | done | 817/819 correct; model ~2x low (1.15 short, 4.6 long) on battery; outliers listed. |
| B41 prebuilt CUDA programs | blocked | Needs aarch64 CUDA sample binaries built elsewhere. |
| T1 `agx_cost --uniform` | done (#7) | |

## Order of work

1. Safe and ready now: B2a, B2c, B47, B46, B35, B11, B3 rest, B18, B38, B43, B36, B45, B39, B42 (raw DRM), B1 on
   `agx_runtime`, B10 write-up.
2. Then B15 and B29, which need new capture paths.
3. Unsafe, only with the owner's approval: B44, B24-5, B14, B17.
4. Blocked: B19, B23, B41; and B37, B40 until an LLVM AGX build exists here.

## Simulator parity (owner, 2026-10-21: hardware functionality found here must also be in the simulator)

| Finding | In the simulator |
| --- | --- |
| B2a partial workgroups | yes (agx_run already ran exactly X x Y x Z threads) |
| B11 `.neg` / `.abs` canonicalize NaN, flush fp32 denormals | yes (identical on all 140 probe lanes) |
| B46 rcp / rsqrt tables, exponent scaling, denormal flush | yes (`transtab.rs`) |
| B47 7-bit shift amounts | yes |
| B15 `simd_matrix_fmadd32/16` layout and in-order fma chain | **added 2026-10-21** (`K::Matrix`; `make verify-simd-matrix`: 11,520 hardware words exact); cost rows added |
| B18 lane-offset fmadd, 4-lane segments k <= 3 | **added 2026-10-21** (fp16 boundary lanes read 0; fp32 boundary lanes stay refused as undefined); cost rows added |
| B18 / B15 / `simd_shuffle_up` instruction costs | **added 2026-10-21** (bench rows in `results/2026-10-11-instruction-bench/bench.json.gz`, forms in `bench_ops.EXTRA_OPS`) |
| B14 calls / returns (all masks, divergence, nesting) | yes (`results/2026-10-20-ret-stop-model`: 570 programs exact) |
| B35 SR113 power-state bit | not modelled by design (time-varying) |
| B38 / B43 / B45 several VMs: concurrent, no implicit order, shared objects | **added 2026-10-21** to `agx_device`: cross-queue / cross-VM `wait_for`, `share` (one object in two VMs); whole commands in policy order = a legal outcome for synchronized programs |
| B8 correction: the USC invalidate is needed for deep queues | todo: a runtime-level rule (ZLUDA), nothing for agx_run to execute |

## Hand-off (2026-10-21, end of session)

Prepared and gated in agx_run, **not yet run on the GPU** (run them in this order; each is safe-class):
- B10 IRIW: playground `results/2026-10-21-b10-iriw/iriw.py gpu 65536 16 3 fenced` (and `plain`).
- B24 item 6: `results/2026-10-21-b24-nop6-operands/b24n.py gpu` (248 operand variants of the 31 six-byte no-ops).
- B29 probes 4 / 9: `scripts/run_vk_b29.py` (harness `tests/vulkan_texture.c`, build to /scratch/agxsim-runs/vkt).
- B44 (one instruction fault + one firmware-ended deadlock, as B20 / B26): `/scratch/ZLUDA/agx_runtime/examples/b44_isolation.rs`
  with `/scratch/agxsim-runs/b44/{control,op54,deadlock}.bin` (`results/2026-10-21-b44-vm-isolation/build.py`); run
  `b44_isolation /scratch/agxsim-runs/b44 vms` then `queues`.
Needs the owner: B17 (async_load, hung before) and B24 item 5 (memory_barrier sweep). Blocked: B19 (M2 Ultra), B23 / B41
(CUDA binaries built elsewhere). Simulator follow-ups: texture compare / LOD clamps / pixel coordinates (B29).
