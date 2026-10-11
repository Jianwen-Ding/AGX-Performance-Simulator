# AGX Performance Simulator

A cycle-level performance model for the Apple AGX GPU (G14 / M2 family), calibrated and
validated on an M2 Max (driver-reported G14C B1) under Asahi Linux.

Imported from [Apple-M2-Playground](https://github.com/Jianwen-Ding/Apple-M2-Playground) at
commit `9754f83`, keeping the same directory layout so every script, result and finding path
is unchanged.

`scripts/agx_cost.py` decodes an AGX binary, derives its dynamic path with applegpu's emulator,
and simulates one core running G SIMD groups cycle by cycle. Predictions are in reference units
(dependent 8-byte `iadd` step = 2); real cycles = × 1.049.

```sh
python3 scripts/agx_cost.py code.bin [--groups G] [--annotate]
```

Held-out accuracy (median |prediction − measurement| / measurement, whole loop trip):
3.7% with 1 SIMD group and 3.3% with 4 groups per core on 24 fresh 32-instruction kernels;
4.5% / 6.0% on 24 kernels with memory instructions. Full model description, accuracy tables and
known gaps: [novel-results/2026-10-11-cost-model](novel-results/2026-10-11-cost-model/README.md).

## Layout

| Path | Contents |
| --- | --- |
| `scripts/agx_cost.py` | The simulator (decode, trace, cycle simulation) |
| `scripts/agx_loop.py` | applegpu wrapper: assemble/decode, emulator, loop kernels |
| `scripts/verify_cost_model.py` | Offline verifier: re-scores every stored measurement |
| `scripts/cost_hw.py`, `scripts/cost_probes.py` | Hardware calibration, random-kernel validation, targeted probes |
| `scripts/run_bench.py`, `scripts/bench_ops.py`, `scripts/verify_bench.py`, `scripts/bench_report.py` | 60-form instruction bench that supplies per-class latency / issue / throughput |
| `scripts/agx_capture.py` | Capture Mesa-compiled shaders and annotate them with the model ([docs](docs/shader-capture.md)) |
| other `scripts/*.py` | Direct-DRM / Vulkan dispatch and patching helpers the above import |
| `tests/*.c` | C harnesses the hardware runners compile (`drm_compute.c`, Vulkan, cache-layout) |
| `results/` | Calibration data and every validation measurement |
| `novel-results/` | The cost-model finding and the hardware findings its rules come from |

## Requirements

* Python 3 (standard library only).
* [applegpu](https://github.com/dougallj/applegpu), expected at `/save/workspace/AGX_Probe/applegpu`
  (`APPLEGPU` in `scripts/run_agx_timestamps.py`; `--applegpu` on the patchers). Needed for decoding
  and emulation, so also for the offline verifier.
* Hardware targets only: Asahi Linux on Apple silicon, a C compiler, Mesa headers
  (`PROBE_INCLUDE`), and scratch space under `/scratch/apple-m2-playground/`.

## Make targets

```sh
make check              # Python syntax check, no GPU
make verify-cost-model  # offline: model accuracy against all stored measurements
make verify-bench       # offline: retained instruction-bench evidence
make bench              # hardware: instruction bench
make cost-calibrate     # hardware: loop / get_sr calibration
make cost-validate      # hardware: random kernels measured vs predicted
```

Some findings link to experiments, notes and results that stayed in Apple-M2-Playground; those
links resolve in that repository.
