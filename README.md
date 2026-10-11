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

## What is modelled (all parameters measured on this GPU)

| Layer | Model | Accuracy (median error) |
| --- | --- | --- |
| Instructions | latency, issue interval and unit per class (60-form bench), result cache, forwarding, writeback | 1-group random kernels 3.7% |
| Core | 4 schedulers (1 issue/cycle, group g on scheduler g % 4); two halves, each with 2 register reads/cycle and one transcendental/imad/bit-field/convert/64-bit unit | 1–32 groups, 738 measurements: 3.0–3.5% |
| Memory | 64-byte segments; L1 8 KiB (loads 2, stores 1 segment/cycle), L2 256 KiB/core, SLC, DRAM 410 GB/s; load-after-ALU quiet time holds back whole batches | 6 stored sets: 1.8–7.2% |
| Shared memory, sync | threadgroup memory: 32 banks x 4 bytes, 1 access/cycle/core, ~31 (load) / 24 (store) cycles per access per group; barriers 13 (0x68) / 7 (0x78) units after the last arrival, arrival after a 24-unit drain; device atomics 1.7 units per 32-byte sector (min 6.9), shared words 2 ops/cycle GPU-wide | 4 sets: 1.4-5.2% |
| Dispatch | occupancy (registers, 64 KiB threadgroup memory, 96 groups), cluster placement ceil(ceil(n/4)/9), launch max(100 ns/workgroup, 13.6 ns/group) + 5.6 us, waves; divergent if/else runs both sides, exec-mask instructions hold their scheduler 3 cycles | 29 dispatches (incl. 4 divergent) |

Findings: [core scaling](novel-results/2026-10-11-core-scaling/README.md),
[memory hierarchy](novel-results/2026-10-11-memory-hierarchy/README.md),
[dispatch and launch](novel-results/2026-10-11-dispatch-launch/README.md),
[cost model](novel-results/2026-10-11-cost-model/README.md).

Known gaps: latency-bound chains at 10–20 groups per core (20–60% slow on hardware, cause unknown); the
group→scheduler mapping when some groups idle; reuse of lines fetched by other cores; threadgroup-memory
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
```

Some findings link to experiments, notes and results that stayed in Apple-M2-Playground; those
links resolve in that repository. Shader capture (`agx_capture.py`) also stays there.
