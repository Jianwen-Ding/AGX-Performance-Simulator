# AGX Performance Simulator

A cycle-level performance model for the Apple AGX GPU (G14 / M2 family), calibrated and
validated on an M2 Max (driver-reported G14C B1) under Asahi Linux.

The model is written in Rust (`rust/`, see [rust/README.md](rust/README.md)). It was ported from
`agx_cost.py` in [Apple-M2-Playground](https://github.com/Jianwen-Ding/Apple-M2-Playground)
(commit `9754f83`), first bit-identically; since 2026-10-11 it models the core's internal structure (four
schedulers, two halves with shared units and register-read ports, see [core scaling](novel-results/2026-10-11-core-scaling/README.md)),
so results with more than one SIMD group per core differ from the Python model. It decodes
an AGX binary, derives its dynamic path by emulation, and simulates one core running G SIMD groups
cycle by cycle. Predictions are in real cycles (`--units ref` gives reference units: dependent
8-byte `iadd` step = 2; real cycles = × 1.049).

```sh
make rust
rust/target/release/agx_cost code.bin [--groups G] [--annotate] [--units real|ref]     # one core, G identical groups
rust/target/release/agx_dispatch code.bin --grid X[,Y,Z] --wg x[,y,z] [--regs HALVES] \
    [--tg-bytes N] [--clock MHZ] [--uniform u5=COUNT ...] [--no-mem]                     # whole dispatch, in us
make verify-cost-model      # offline: instruction/core model against every stored measurement (~20 s)
make verify-memory-model    # offline: memory hierarchy against the stored memory sets (~2 min)
make verify-dispatch        # offline: whole-dispatch model against 25 stored dispatches
```

`agx_dispatch` computes occupancy (register file, workgroup size), places workgroups on the busiest core
(cluster split), traces every SIMD group of that core with its own thread ids, simulates the core with the
memory hierarchy, and adds launch cost; it prints a JSON report with the predicted time.

Launch inputs are validated before decoding or materializing functional buffers:
nonzero grid/core dimensions, 1–1024 workgroup threads, representable workgroup
counts, available configured occupancy/shared memory, uniform indices and finite
timing inputs. Empty grids are explicitly outside this performance model's input
domain. In `--batch`, malformed argument arrays and invalid jobs produce ordered
`{"error": ...}` entries without aborting later jobs; blank lines are ignored,
and an empty batch fails. These are software input checks, not new hardware limits
or timing validation for extreme launch dimensions.

## What is modelled (all parameters measured on this GPU)

| Layer | Model | Accuracy (median error) |
| --- | --- | --- |
| Instructions | latency, issue interval and unit per class (60-form bench), result cache, forwarding, writeback | 1-group random kernels 3.7% |
| Core | 4 schedulers (1 issue/cycle, group g on scheduler g % 4); two halves, each with 2 register reads/cycle and one transcendental/imad/bit-field/convert/64-bit unit | 1–32 groups, 738 measurements: 3.0–3.5% |
| Memory | 64-byte segments; L1 8 KiB (loads 2, stores 1 segment/cycle), L2 256 KiB/core, SLC, DRAM 410 GB/s; load-after-ALU quiet time holds back whole batches; a load after ALU work waits for pending loads to pass the L1 (not to complete) | 6 stored sets: 1.8–7.2% |
| Shared memory, sync | threadgroup memory: 32 banks x 4 bytes, 1 access/cycle/core, ~31 (load) / 24 (store) cycles per access per group; barriers 13 (0x68) / 7 (0x78) units after the last arrival, arrival after a 24-unit drain; device atomics 1.7 units per 32-byte sector (min 6.9), shared words 2 ops/cycle GPU-wide | 4 sets: 1.4-5.2% |
| Dispatch | occupancy (registers, 64 KiB threadgroup memory, 96 groups), cluster placement ceil(ceil(n/4)/9), launch max(100 ns/workgroup, 13.6 ns/group) + 5.6 us, waves; data that fits the SLC GPU-wide starts there; whole-dispatch DRAM 360 GB/s; divergent if/else runs both sides, exec-mask instructions hold their scheduler 3 cycles | 36 dispatches incl. 4 divergent and 7 tree reductions: 4.7% (p90 9%) |

Findings: [core scaling](novel-results/2026-10-11-core-scaling/README.md),
[memory hierarchy](novel-results/2026-10-11-memory-hierarchy/README.md),
[dispatch and launch](novel-results/2026-10-11-dispatch-launch/README.md),
[cost model](novel-results/2026-10-11-cost-model/README.md).

Known gaps: simple-ALU chains with 1-3 dependency chains at 9-24 groups per core (hardware up to 4.95 instructions
per cycle at 10-12 groups and 20-60% slower at 15-20; not lockstep, not writeback; cause unknown); threadgroup-memory
textures and divergence are not measured; barriers after ALU work at 8-32 groups are 13-31% low;
predictions assume the top clock (1398 MHz) unless `--clock` is given.


Held-out accuracy (median |prediction − measurement| / measurement, whole loop trip):
3.7% with 1 SIMD group and 3.3% with 4 groups per core on 24 fresh 32-instruction kernels;
4.5% / 6.0% on 24 kernels with memory instructions.
With 1–32 SIMD groups in one workgroup per core (`scripts/scale_probes.py`, 23 instruction forms,
1 and 8 dependency chains, 598 measurements) the median error is 3.0% (p90 29%); see the core-scaling finding
for the open cases (latency-bound chains at 10–20 groups, `icmpsel`, `fmadd16`, ballots). Full model description, accuracy tables and
known gaps: [novel-results/2026-10-11-cost-model](novel-results/2026-10-11-cost-model/README.md)
(written for the Python version; file and function names there refer to it).

## Layout

| Path | Contents |
| --- | --- |
| `rust/` | The model: decoder/emulator (applegpu port), simulator, CLI (`agx_cost`), `agx_predict`, verifier, benchmark |
| `scripts/cost_hw.py`, `scripts/cost_probes.py`, `scripts/fetch_probe*.py` | Hardware calibration, random-kernel validation, targeted probes (Python; predictions come from `agx_predict`) |
| `scripts/run_bench.py`, `scripts/bench_ops.py`, `scripts/verify_bench.py`, `scripts/bench_report.py` | 60-form instruction bench that supplies per-class latency / issue / throughput |
| `scripts/scale_probes.py`, `scripts/mem_probes.py`, `scripts/wg_placement.py`, `scripts/dispatch_probes.py` | 1–32 groups per core, memory hierarchy, workgroup placement, whole-dispatch validation |
| other `scripts/*.py` | Direct-DRM / Vulkan dispatch, assembly and patching helpers the hardware scripts import |
| `tests/*.c` | C harnesses the hardware runners compile (`drm_compute.c`, Vulkan, cache-layout) |
| `testsuite/`, `coverage/` | Hardware-recorded functional test suite (721 cases); coverage catalog of every known behaviour, to be recorded ([coverage/README.md](coverage/README.md)) |
| `spirv/`, `tests/vk_run.c` | Realistic SPIR-V compute workloads with numpy references, a generic Vulkan runner, and the path from the driver's compiled code to the simulator |
| `spirv/compiled/`, `scripts/build_mesa_shim.sh` | The workloads compiled by Mesa's Asahi driver (25.2.8, 26.0.3) without the GPU (no-op drm-shim reporting a G14C); root-table layouts per Mesa version ([finding](results/2026-10-12-spirv-offline/README.md)) |
| `tasks/` | Self-contained tasks for an agent on Asahi Linux: [coverage recording](tasks/agx-coverage.md), [SPIR-V workloads](tasks/spirv-workloads.md), [safe AGX probes](tasks/safe-probes.md) (safe-01 .. safe-06: numeric semantics, cost-model gaps, memory system, submission, textures via Vulkan, the 214 unknown special-register cases) |
| `results/` | Calibration data and every validation measurement |
| `novel-results/` | The cost-model finding and the hardware findings its rules come from |

## Requirements

* Offline (model, verifier): a Rust toolchain only.
* Hardware runs (`make bench`, `cost-calibrate`, `cost-validate`): Asahi Linux on Apple silicon,
  Python 3, a C compiler, Mesa headers (`PROBE_INCLUDE`), scratch space under
  `/scratch/apple-m2-playground/`, and [applegpu](https://github.com/dougallj/applegpu) at
  `/save/workspace/AGX_Probe/applegpu` (the Python scripts use it to assemble kernels).
  `cost-validate` also needs `make rust` (it calls `rust/target/release/agx_predict`; override
  with `AGX_PREDICT`).

## Make targets

```sh
make rust               # build
make verify-cost-model  # offline: model accuracy against all stored measurements
make test               # offline: Rust unit and equivalence tests
make verify-bench       # offline: retained instruction-bench evidence
make bench              # hardware: instruction bench
make cost-calibrate     # hardware: loop / get_sr calibration
make cost-validate      # hardware: random kernels measured vs predicted
make coverage-validate  # offline: every coverage case builds, assembles and decodes
make coverage-record    # hardware: record the coverage catalog (tasks/agx-coverage.md)
make coverage-check     # offline: simulator and cost model against the coverage recordings
make spirv-lavapipe     # offline: SPIR-V workloads on Mesa lavapipe (glslang, spirv-tools, Vulkan headers)
make spirv-run          # hardware: SPIR-V workloads on the GPU (tasks/spirv-*.md)
make asahi-verify       # hardware (M2): record every SPIR-V workload on the GPU and verify the simulator against it
make spirv-verify       # offline: simulator vs those recordings (PENDING until they exist)
python3 -B scripts/spirv_task.py retime all   # hardware: re-measure every recording's times only (results/2026-10-16-asahi-retime)
make spirv-sim          # offline: functional simulator on the driver-compiled workloads vs their references (16 / 16)
make spirv-predict      # offline: predicted time of every workload from the driver-compiled code
make spirv-compile      # offline: build Mesa (MESA_VERSION, default 26.0.3) with the drm-shim, recompile the workloads
make verify-traces      # offline: functional-simulator paths vs the cost model's emulation on the test suite
make verify-transcendentals  # offline: exp2 / sin_pt_2 / log2 and the sin/cos sequences vs the M2, bit for bit
make verify-sr          # offline: all special registers vs the M2 under divergence (1518 lane-checked values)
```

Other offline tools: `agx_asm` (assembler / decoder / raw field edits on stdin), `agx_run` (functional simulation of a
dispatch over arbitrary memory, including a driver preamble that fills the uniforms). `agx_dispatch` takes the same
`--preamble` / `--map` / `--alloc` options. With them, the SIMD groups' paths come from the functional simulator over
that memory instead of the cost model's emulation, which is needed for driver-compiled code and data-dependent loops.

Some findings link to experiments, notes and results that stayed in Apple-M2-Playground; those
links resolve in that repository. Shader capture (`agx_capture.py`) also stays there.


Call/return probing: `agx_run` opts into the measured experimental model with
`AGX_EXPERIMENTAL_RET=buffer-pop2`. Default execution still rejects return and
register-call instructions; target bounds checks remain enabled. The legacy
value `1` is a contradicted hypothesis for offline comparison. Measured scope,
return/stop corrections and validation of 570 original hardware cases are
in the adjacent playground's [model integration report](../apple-documentation-playground/results/2026-10-20-control-simulator/README.md).
Run its `make verify-control-model` and `make verify-call-ret` for exact replay;
call/return timing remains uncalibrated. Nonempty return maps effective mask
depths 0 -> 2 and 1 -> 0, preserving deeper lanes (measured through depth 4).
Empty-issued return falls through without changing counters. Stop terminates
issuing lanes while masked live lanes continue; later mask operations cannot
revive stopped or nonexistent lanes. See the adjacent playground
[return/stop integration report](../apple-documentation-playground/results/2026-10-20-ret-stop-model/README.md).

Threadgroup atomic ops 11–15 now reject explicitly even with an empty execution
mask. Hardware probing found that issuing them stops the group; the old empty-mask
skip was incorrect. Shared-memory side effects before termination are unmeasured,
so a complete functional model remains pending. In the adjacent playground,
`make verify-tg-atomic-followup` validates 27 retained variants, simulates every
complete surrogate, replays 12 understood variants, and checks rejection of the
15 ending-opcode cases. `make test-probe-watchdog` there tests bounded timeout
handling and partial evidence retention without submitting GPU work.

The functional `simd_shuffle` model distinguishes masking from termination:
masked living lanes contribute low index bits and remain readable; stopped
lanes do not contribute to the quad OR, and selecting one returns the issuing
lane's own source value. Both interpreter paths are checked. The adjacent
playground's [shuffle liveness report](../apple-documentation-playground/results/2026-10-20-shuffle-liveness-model/README.md)
binds exact replay of 1,115 original hardware programs, retaining the original
275 counterexamples. Other native shuffle forms remain separate follow-ups.

The same liveness rules are now independently measured and implemented for
`simd_shuffle_down`, `simd_shuffle_up`, `simd_shuffle_xor` and
`simd_shuffle_rotate_up`, preserving their quad-low and boundary behavior.
Both interpreter paths pass the measured fixtures. See the adjacent
[native shuffle integration](../apple-documentation-playground/results/2026-10-20-native-shuffle-model/README.md)
for exact replay of 1,595 original control programs and 42 passing library tests.

Ordinary r0 writes update counter storage without immediately changing execution.
A dedicated pop synchronizes execution, including pop_exec 0. Nonempty RET
conditionally changes counter 0 to masked 2 and counter 1 to active 0; deeper
counters retain their current execution bits, even if stale after an ordinary
write. See the adjacent [conditional RET integration](../apple-documentation-playground/results/2026-10-20-ret-stale-model/README.md)
for 2,277 exact original control replays and 34 corrected counterexamples.

Later unsigned-counter and stale-target measurements extend the exact control
corpus to 2,403 programs. Sampled low-r0 values through 0xffff retain unsigned
pop/RET behavior; RET selects the lowest cached issuer before counter updates,
even with deliberately stale execution and three divergent destinations.

Known functional limit under investigation: counter-increment overflow can
reactivate a masked lane whose first native shuffle retains its destination
instead of writing the expected source. The adjacent playground's
[overflow diagnostic report](../apple-documentation-playground/novel-results/2026-10-20-overflow-first-shuffle/README.md)
preserves **18 counterexamples** among 27 complete programs. All 81 hardware
dispatches completed with exact controls and zero faults. Wait0/pop0/a second
shuffle fixes the tested programs; one intervening ordinary ALU instruction
does not. These records remain separate from the exact corpus. No generic
pipeline rule or calibrated timing correction is implemented yet; use the
playground's `make verify-overflow-first-shuffle` to check the retained matches
and explicit mismatches while this execution-state effect is investigated.
