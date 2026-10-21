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
    [--uniform I=V ...] [--wg x,y,z] [--wg-id ...] [--simd-group G] [--map VA=FILE ...]   # that group's real path
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
| Dispatch | occupancy (registers, 64 KiB threadgroup memory, 96 groups), cluster placement ceil(ceil(n/4)/9), launch max(100 ns/workgroup, 13.6 ns/group) + 5.6 us, waves; data that fits the SLC GPU-wide starts there; whole-dispatch DRAM 360 GB/s; divergent if/else runs both sides, exec-mask instructions hold their scheduler 3 cycles; an instruction issued with no active lane (an if body not jumped over) costs one issue cycle (ALU) or its measured masked cost (device load + wait ~11 ns, threadgroup load + wait ~12 ns, device store ~10 ns) | 36 dispatches incl. 4 divergent and 7 tree reductions: 4.7% (p90 9%) |

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
`agx_run --lines OBJ.o [--lines-offset N]` reads the DWARF line table of the AGX object the code came from (`llc`
with debug locations, `zoc --lineinfo`); every error names the PC of the failing instruction, and with a line table
also its `"source"` (`rust/src/lines.rs`).

Strict mode: `agx_run`, `agx_dispatch` and `agx_cost` take `--agx-strict GPU` (`g13`, `g13x`, `g14`, `g14d`, or an LLVM
processor such as `agx-g14c`; `AGX_STRICT=GPU` does the same for every tool). The functional simulator then fails on
the first executed instruction that Mesa's asahi compiler would not emit for that GPU, and the cost model on any
instruction of the program, naming the closest Mesa form and the field that differs. M1 (G13) code runs in this mode
with the G14 semantics and cost model, as everywhere else in the simulator; the two generations differ in Mesa's output
only in bit 45 of device atomics (set for multi-cluster G13, `g13x`, and two-die parts, `g14d`). The forms are generated
by `llvm/utils/agx-mesa-isa/mesa_isa.py` in the LLVM AGX fork into `rust/src/isa/mesa_table.rs`;
`rust/src/isa/mesa_corpus.txt` holds instructions packed by Mesa's own `agx_pack.c`, which the tests check are accepted.
The LLVM backend's `agx-strict` feature emits only these forms.

Multiple contexts: `agx_device SPEC.json` (library: `agx_sim::device`) runs several contexts (GPU VMs) on one
simulated GPU, each with its own memory, VM slot (what SR21 reads) and queues of dispatches; commands of a queue run in
order, and a command that faults (unmapped access, step limit) is reported with both timestamps 0 while the others run
on. How the hardware interleaves commands of different VMs is not measured yet (`tasks/ptx-re-blockers.md` B43-B45), so
the order is a policy: `round-robin` (default) or `submission`. Memory shared between VMs is not modelled.

Checked mode and fault replay: kernels built with the LLVM backend's `agx-checked` feature (ZLUDA's `zoc
--agx-checked`) run on `agx_run` like any other once their hidden arguments are passed as uniforms;
`llvm/utils/agx-checked/run_checked.py` in the LLVM fork does that and decodes the error record. ZLUDA's `zluda_agx`
saves a launch that faulted on the GPU (`ZLUDA_AGX_DUMP=DIR`) as the kernel's code, its uniforms and the memory it can
reach, with a `replay.sh` that runs it here: `AGX_RUN=rust/target/release/agx_run DIR/<pid>-<command>/replay.sh`.

`agx_run --m2-core-map FILE` optionally replays supplied SR20 identities. Append
it to a complete command with the normal uniforms and memory mappings. FILE
is a JSON array with one packed M2 Max core ID per workgroup, x varying fastest,
then y and z. `func::run_with_m2_core_map` is the library equivalent; callers
running a preamble separately can preflight with `validate_m2_core_map`.
Length and enabled-core masks are checked before execution. The measured
aligned eight-workgroup cluster rule is enforced for 1-D whole-workgroup grids
at sizes 32/64/128/256, including partial final blocks. It is not imposed on
other geometries. Invalid CLI maps fail before preamble execution or dumps.
Mapped results echo `core_map: "m2"` and `mapped_workgroups`.

This supplies scheduling state, not a schedule predictor. SR17 slots retain
their selected default/synthetic behavior. Default execution and the library's
synthetic `Dispatch::placement` mode remain unchanged. Prepared functional
traces accept maps through `PreparedDispatch::with_m2_core_map` or
`func::trace_workgroups_with_m2_core_map`. The prepared object borrows a validated
immutable map; each requested workgroup still traces independently from fresh
registers and a snapshot of supplied memory. Decode metadata is shared, while
core-dependent values and loaded addresses are resolved per trace.
`agx_dispatch --m2-core-map FILE` now evaluates complete maps through a
separate exhaustive path (described below). The
[replay verification](../apple-documentation-playground/novel-results/2026-10-20-observed-core-map-replay/README.md)
matches 80 complete retained hardware buffers literally, preserves 64 default
cases against the archived model and independent oracles, and passes three
targeted library tests and five CLI failure cases. All seven ordinary
arithmetic regression suites also pass; no hardware was submitted. The
[prepared-trace verification](../apple-documentation-playground/novel-results/2026-10-20-observed-core-map-tracing/README.md)
checks all 80 maps against independent lane-address oracles across two memory
snapshots, reordered/duplicate requests and four CPU tracing workers. Five
targeted core-map tests now pass. This does not replay concurrent execution.

`agx_dispatch --m2-core-map FILE` uses functional memory mappings/uniforms and
validates the map before reading shaders/preambles or allocating mapped buffers.
It requires the 38-core M2 Max profile, rejects `--busiest-core-wgs` and an
explicit `--max-waves`, and simulates every supplied workgroup in every core's
partition. `mapped_dispatch::simulate` is the library equivalent. Per-core
linear workgroup order and occupancy-sized waves are assumptions; no waves are
extrapolated. The JSON contains `cores` with each partition and wave, actual
`active_cores`, `critical_core`, `global_device_segments`, `waves_exhaustive`
and the wave, slot, memory and cache assumptions. The top-level cycle/wave
fields describe the critical core; `busiest_core_wgs` is the maximum count
across all cores and can belong to another core.

The device footprint uses all isolated traces. `Trace::device_segments`
preserves complete atomic byte ranges, including crossing/unaligned 32/64-bit
accesses, separately from the legacy `Trace::mem` cost inputs. Sharing uses different actual core partitions.
Texture/image/stack and texture/sampler-descriptor memory traces remain incomplete: those shaders report
`device_footprint_complete: false` and do not assume SLC residency. Cache state
restarts per modeled wave. Per-core completion times are combined by their
maximum plus fixed dispatch overhead; inter-core overlap, slots and frequency
are not supplied by the map. This is a cost estimate under declared assumptions,
not a reconstruction of concurrent hardware execution. The
[cost verification](../apple-documentation-playground/novel-results/2026-10-20-observed-core-map-cost/README.md)
checks all retained maps, independent partitions/footprints, and unchanged
legacy default/scalar-peak predictions. No timing parameters were fitted.

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

The [prepared-index follow-up](../apple-documentation-playground/novel-results/2026-10-20-overflow-register-visibility/README.md)
extends this limit beyond shuffles: immediate/copy/add can also retain old
destinations in overflow-activated lanes. The readback persists after wait/drain
with addresses prepared before divergence. It adds 17 counterexamples among
27 programs; combined overflow scope has **35 counterexamples in 54 programs**,
162 clean completed dispatches. Tested move-spacing and lane-dependent behavior
are recorded, without a calibrated-cycle or general pipeline implementation.

Default execution now detects IF/push overflow that activates a previously
masked living lane. Further work is refused until the directly measured
`wait 0` or `pop_exec r0l, 0` settlement, instead of silently predicting
immediate activation. This conservative guard also refuses some sufficiently
spaced programs that hardware accepts; it does not guess a universal spacing
count. Already-issuing wraps remain supported, and dead/absent lanes cannot
create activation. The functional mask path now respects the same persistent
liveness checks as the compiled/reference executors.

`AGX_EXPERIMENTAL_EXEC_OVERFLOW=counter-sync` explicitly retains the old,
known-incomplete immediate-sync hypothesis for offline comparisons and tightly
controlled probing. It is not a hardware-accurate activation/timing mode.
The adjacent [overflow guard report](../apple-documentation-playground/results/2026-10-20-overflow-guard/README.md)
binds 54 original-program replays: default accepts 11 exact and explicitly
refuses 43 unresolved programs; the opt-in preserves all 35 counterexamples.
All 2,403 earlier exact control cases remain exact, 44 library tests pass,
and both control suites pass with default and return-opt-in settings.

An [ordinary integer-add timing follow-up](../apple-documentation-playground/novel-results/2026-10-20-integer-chain-cost/README.md)
checks 83 straight-line programs, 249 exact recovered dispatches, on AC power.
Interleaving 1/2/4/8 register chains does not remove the isolated-dispatch
shortfall: incremental firmware-time slopes are about 3–3.7 times predictions.
Actual short-dispatch operating frequency and warm-up remain unresolved, so
this is a documented calibration question, not a justified global latency
change. Complete distributions and model/source/parameter bindings are retained
in the playground's `analysis.json`.

The [consecutive-dispatch check](../apple-documentation-playground/novel-results/2026-10-20-integer-chain-train/README.md)
completed 19 trains and 1,194 dispatches with exact final buffers, passing
controls and zero faults. First/last-quarter median ratios stay within
0.991–1.022; slope residuals remain 3.42–3.76 times predictions. Intermediate
buffers were not captured and operating frequency remains unmeasured.
The immediate-value/form difference from calibration is a next question;
these results do not justify a global cost change.

The [matched immediate-value study](../apple-documentation-playground/novel-results/2026-10-20-integer-immediate-cost/README.md)
adds 26 trains / 1,596 exact recovered dispatches. Eight-chain immediate-1/3
timings differ by less than 1% in pooled late medians and retain the 3.6x
incremental gap. Identical one-chain shaders vary by over 2x between trials.
The model predicts equal cost for both immediate values; it does not model
these variable timing states. Pooled one-chain ratios are not latency estimates.

The [controlled host-gap study](../apple-documentation-playground/novel-results/2026-10-20-integer-timing-state/README.md)
completes 39 trains / 2,394 exact recovered dispatches with zero faults.
Timing can change within one process using identical shaders and buffer
objects, so fixed allocation effects alone are insufficient. Adding 1000 us
between submissions gives late medians agreeing within 1.1% across six trials
per shape, while some early quarters remain faster. Operating frequency is
still unmeasured; the model has no timing-history correction. This protocol
is a follow-up validation condition, not a global latency recalibration.

The [controlled-gap length study](../apple-documentation-playground/novel-results/2026-10-20-integer-length-cost/README.md)
adds 50 trains / 3,180 exact recovered dispatches over 600–33,066-byte shaders.
Late medians agree within 2.3% across three trials per cell. Adjacent candidate
slopes remain 2.58–3.78x predictions across the tested lengths, without an
isolated large jump identifying a cache threshold. The current shaders do not
measure frequency or per-core assignment; the latter is the next check on the
38-core launch assumption. No timing calibration changed.

The [arithmetic core-distribution study](../apple-documentation-playground/novel-results/2026-10-20-integer-core-distribution/README.md)
checks 28 trains / 1,608 recovered dispatches with zero faults: arithmetic and
guards exact, SR19 allocation tags exact, and every SR20 placement tag valid
and uniform within its workgroup. All 12 full-size SR20 maps contain all 38
cores. The busiest core receives 28–33 cumulative workgroups, versus the
default estimate of 27. This does not measure simultaneous residency.

Append `--busiest-core-wgs N` to an otherwise complete `agx_dispatch` command
to use a measured peak count (`Launch::busiest_core_wgs_override` in the API).
The count must be between ceil(workgroups / cores) and workgroups; invalid
counts fail before shader or mapped-buffer access. The output echoes
`busiest_core_wgs_input`. This adjusts representative-core waves and launch
work with distinct sampled workgroup paths. It does not reconstruct a full
placement map, core-specific paths, memory traffic, concurrency or clock state.
Using these measured peaks changes predictions by 0.35–17.35%, with a remaining
2.76–3.06x residual for the corresponding final dispatches. All eight saved
default predictions are unchanged. Three targeted `observed_peak` library
tests and the playground's offline core-distribution verifier cover the input.

The [workgroup-geometry study](../apple-documentation-playground/novel-results/2026-10-20-integer-geometry-cost/README.md)
adds 160 trains / 9,600 recovered dispatches, with exact arithmetic/guards,
valid tag planes, exact controls and zero faults. All 72 full-size SR20 maps
preserve one cluster ID within each aligned eight-workgroup block at the
tested 1-D sizes 32/64/128/256, including seven- and one-workgroup final blocks
on neighboring grids. This is not a fixed-thread-count pattern or a complete
scheduler model. Every map contains all 38 known core IDs.

Measured-peak predictions still leave 2.67–3.42x final-dispatch residuals on
the fixed-lane-count grid. A neighboring-grid cell has 2.23x trial variation
despite the 1000 us delay, so that protocol is not a general timing guarantee.
Current default and synthetic functional placement do not predict these
observed schedules. Optional supplied-core-map replay is now implemented for
functional execution, prepared tracing and explicit exhaustive cost partitions,
preserving existing modes and distinguishing historical
model provenance from the newer replay model. No timing calibration changed.
