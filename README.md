# AGX Performance Simulator

A cycle-level performance model for the Apple AGX GPU (G14 / M2 family), calibrated and
validated on an M2 Max (driver-reported G14C B1) under Asahi Linux.

The model is written in Rust (`rust/`, see [rust/README.md](rust/README.md)). It was ported from
`agx_cost.py` in [Apple-M2-Playground](https://github.com/Jianwen-Ding/Apple-M2-Playground)
(commit `9754f83`) and gives bit-identical results; the Python model has been removed. It decodes
an AGX binary, derives its dynamic path by emulation, and simulates one core running G SIMD groups
cycle by cycle. Predictions are in real cycles (`--units ref` gives reference units: dependent
8-byte `iadd` step = 2; real cycles = × 1.049).

```sh
make rust
rust/target/release/agx_cost code.bin [--groups G] [--annotate] [--units real|ref]
make verify-cost-model   # offline: re-score every stored measurement (~9 s)
```

Held-out accuracy (median |prediction − measurement| / measurement, whole loop trip):
3.7% with 1 SIMD group and 3.3% with 4 groups per core on 24 fresh 32-instruction kernels;
4.5% / 6.0% on 24 kernels with memory instructions. Full model description, accuracy tables and
known gaps: [novel-results/2026-10-11-cost-model](novel-results/2026-10-11-cost-model/README.md)
(written for the Python version; file and function names there refer to it).

## Layout

| Path | Contents |
| --- | --- |
| `rust/` | The model: decoder/emulator (applegpu port), simulator, CLI (`agx_cost`), `agx_predict`, verifier, benchmark |
| `scripts/cost_hw.py`, `scripts/cost_probes.py`, `scripts/fetch_probe*.py` | Hardware calibration, random-kernel validation, targeted probes (Python; predictions come from `agx_predict`) |
| `scripts/run_bench.py`, `scripts/bench_ops.py`, `scripts/verify_bench.py`, `scripts/bench_report.py` | 60-form instruction bench that supplies per-class latency / issue / throughput |
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
