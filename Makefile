PYTHON ?= python3
SCRATCH_ROOT ?= /scratch/apple-m2-playground
DEVICE ?= /dev/dri/renderD128
PROBE_INCLUDE ?= /save/workspace/AGX_Probe/mesa/include
export CC PROBE_INCLUDE

.PHONY: help check bench verify-bench capture-demo cost-calibrate cost-validate verify-cost-model rust rust-verify rust-test

help:
	@echo 'make check              Offline: Python syntax check of every script'
	@echo 'make verify-cost-model  Offline: agx_cost accuracy against all stored measurements'
	@echo 'make verify-bench       Offline: check the retained instruction-bench evidence'
	@echo 'make bench              Hardware: 60-form instruction bench (latency, issue, throughput)'
	@echo 'make cost-calibrate     Hardware: loop/get_sr costs for the cost model'
	@echo 'make cost-validate      Hardware: random kernels measured vs agx_cost predictions'
	@echo 'make rust               Build the Rust port (rust/)'
	@echo 'make rust-verify        Rust offline verifier, diffed against results/.../verify-output.txt'
	@echo 'make rust-test          Rust unit and equivalence tests'
	@echo 'make capture-demo       Hardware: capture a Mesa GLES shader and annotate it with agx_cost'

check:
	$(PYTHON) -B -m py_compile scripts/*.py

bench:
	$(PYTHON) -B scripts/run_bench.py --modes latency,simd_issue,gpu

verify-bench:
	$(PYTHON) -B scripts/verify_bench.py

# tests/gles_compute.c lives in Apple-M2-Playground; point GLES_COMPUTE_C at a checkout of it.
GLES_COMPUTE_C ?= ../Apple-M2-Playground/tests/gles_compute.c
capture-demo:
	mkdir -p "$(SCRATCH_ROOT)/capture/demo"
	$(CC) -std=c11 -O2 -I"$(PROBE_INCLUDE)" "$(GLES_COMPUTE_C)" -o "$(SCRATCH_ROOT)/capture/demo/gles_compute" -ldl
	$(PYTHON) -B scripts/agx_capture.py --ir --out "$(SCRATCH_ROOT)/capture/demo" -- "$(SCRATCH_ROOT)/capture/demo/gles_compute" private 64 $(DEVICE)

cost-calibrate:
	$(PYTHON) -B scripts/cost_hw.py calibrate

cost-validate:
	$(PYTHON) -B scripts/cost_hw.py validate

verify-cost-model:
	$(PYTHON) -B scripts/verify_cost_model.py

rust:
	cd rust && cargo build --release

rust-verify: rust
	rust/target/release/verify_cost_model | diff - results/2026-10-11-cost-model/verify-output.txt && echo 'identical to verify-output.txt'

rust-test:
	cd rust && cargo test --release
