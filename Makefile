PYTHON ?= python3
SCRATCH_ROOT ?= /scratch/apple-m2-playground
DEVICE ?= /dev/dri/renderD128
PROBE_INCLUDE ?= /save/workspace/AGX_Probe/mesa/include
export CC PROBE_INCLUDE

.PHONY: help check rust test verify-cost-model verify-memory-model bench verify-bench cost-calibrate cost-validate

help:
	@echo 'make rust               Build the model (rust/target/release: agx_cost, agx_predict, verify_cost_model, agx_bench)'
	@echo 'make verify-cost-model  Offline: model accuracy against all stored measurements, diffed against verify-output.txt'
	@echo 'make verify-memory-model  Offline: memory-hierarchy model against the stored memory measurements (~2 min)'
	@echo 'make verify-dispatch    Offline: whole-dispatch model against 25 stored dispatches'
	@echo 'make test               Offline: Rust unit and equivalence tests'
	@echo 'make check              Offline: Python syntax check of the hardware scripts'
	@echo 'make verify-bench       Offline: check the retained instruction-bench evidence'
	@echo 'make bench              Hardware: 60-form instruction bench (latency, issue, throughput)'
	@echo 'make cost-calibrate     Hardware: loop/get_sr costs for the cost model'
	@echo 'make cost-validate      Hardware: random kernels measured vs model predictions (needs make rust)'

rust:
	cd rust && cargo build --release

test:
	cd rust && cargo test --release

verify-cost-model: rust
	rust/target/release/verify_cost_model | diff - results/2026-10-11-cost-model/verify-output.txt && echo 'identical to verify-output.txt'

verify-memory-model: rust
	rust/target/release/mem_check --stored --limit 0.1

verify-dispatch: rust
	rust/target/release/dispatch_check -v

check:
	$(PYTHON) -B -m py_compile scripts/*.py

bench:
	$(PYTHON) -B scripts/run_bench.py --modes latency,simd_issue,gpu

verify-bench:
	$(PYTHON) -B scripts/verify_bench.py

cost-calibrate:
	$(PYTHON) -B scripts/cost_hw.py calibrate

cost-validate: rust
	$(PYTHON) -B scripts/cost_hw.py validate
