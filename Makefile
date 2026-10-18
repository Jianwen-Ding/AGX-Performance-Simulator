PYTHON ?= python3
SCRATCH_ROOT ?= /scratch/apple-m2-playground
DEVICE ?= /dev/dri/renderD128
PROBE_INCLUDE ?= /save/workspace/AGX_Probe/mesa/include
MESA_VERSION ?= 26.0.3
MESA_SHIM_DIR ?= $(CURDIR)/../mesa-src
export CC PROBE_INCLUDE

.PHONY: help verify-conversions verify-float-edges verify-transcendentals verify-sr check rust test verify-cost-model verify-memory-model bench verify-bench cost-calibrate cost-validate suite-check record-suite record-transcendentals coverage-validate coverage-record coverage-check coverage-properties coverage-status spirv-build spirv-run spirv-lavapipe spirv-sim spirv-compile spirv-predict verify-traces spirv-verify asahi-verify

help:
	@echo 'make rust               Build the model (rust/target/release: agx_cost, agx_predict, verify_cost_model, agx_bench)'
	@echo 'make verify-cost-model  Offline: model accuracy against all stored measurements, diffed against verify-output.txt'
	@echo 'make verify-memory-model  Offline: memory-hierarchy model against the stored memory measurements (~2 min)'
	@echo 'make verify-scaling     Offline: 1-32 groups per core incl. idle-slot rows (scale_check)'
	@echo 'make suite-check        Offline: functional simulator against the hardware-recorded test suite (testsuite/)'
	@echo 'make record-suite       Hardware: re-record the test suite (~15 min)'
	@echo 'make record-transcendentals Hardware: exact fp32 rcp/rsqrt tables and log2/exp2/sin samples (testsuite/transcendental)'
	@echo 'make coverage-validate  Offline: build, assemble and gate every coverage case (scripts/coverage_catalog.py)'
	@echo 'make coverage-record    Hardware: record the safe coverage cases (resumable; tasks/agx-coverage.md)'
	@echo 'make coverage-check     Offline: simulator and cost model against coverage/expected and coverage/timing'
	@echo 'make coverage-properties  Playground verifiers that stand in for the non-suite coverage items (needs the playground checkout)'
	@echo 'make coverage-status    Recorded / pending / faulted coverage cases per area'
	@echo 'make spirv-build        Compile spirv/shaders (glslangValidator, spirv-val) and the vk_run runner'
	@echo 'make spirv-run          Hardware (Vulkan): all SPIR-V workloads, both sizes, checked and timed (tasks/spirv-*.md)'
	@echo 'make spirv-lavapipe     Offline: all SPIR-V workloads on Mesa lavapipe (CPU Vulkan) against their references'
	@echo 'make spirv-sim          Offline: functional simulator on the shim-compiled workloads vs numpy references (a sanity check, not hardware verification)'
	@echo 'make spirv-verify       Offline: simulator vs the GPU recordings in spirv/expected (PENDING until recorded on the M2)'
	@echo 'make asahi-verify       Hardware (Asahi, M2): record every SPIR-V workload on the GPU, then verify the simulator against it'
	@echo 'make spirv-compile      Offline: build Mesa $$(MESA_VERSION) with a no-op drm-shim and compile the workloads with it'
	@echo 'make spirv-predict      Offline: predicted time of every workload (large) from the driver-compiled code'
	@echo 'make verify-conversions Offline: 80 convert forms (1.37e9 hardware words) replayed in agx_run / the model'
	@echo 'make verify-float-edges Offline: fp32 boundary sweeps (floor/ceil/fcmpsel/f16 conversions) + the non-ftz floor/ceil fix-up'
	@echo 'make verify-transcendentals Offline: exp2 / sin_pt_2 / log2 and the sin/cos sequences vs the M2, bit for bit'
	@echo 'make verify-sr          Offline: special registers vs the M2 under divergence (results/2026-10-16-special-registers)'
	@echo 'make verify-traces      Offline: functional-simulator paths vs the cost model emulation on the test suite'
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
	rust/target/release/verify_cost_model > /tmp/agx-verify-cost-model.txt
	$(PYTHON) -c "import sys; a = open('/tmp/agx-verify-cost-model.txt').read(); b = open('results/2026-10-11-cost-model/verify-output.txt').read(); print('identical to verify-output.txt' if a == b else 'DIFFERS from verify-output.txt'); sys.exit(a != b)"

verify-memory-model: rust
	# mem_check simulates full-size caches in every worker thread; with one thread per core it drove the host into a
	# global OOM on 2026-10-16 (other tenants held ~55 GB). Default to 2 threads; override with AGX_THREADS.
	AGX_THREADS=$${AGX_THREADS:-2} rust/target/release/mem_check --stored --limit 0.1

verify-scaling: rust
	rust/target/release/scale_check results/2026-10-11-core-scaling/grid.json results/2026-10-11-core-scaling/chains.json results/2026-10-11-core-scaling/idle.json results/2026-10-11-core-scaling/dests.json

suite-check: rust
	rust/target/release/suite_check testsuite

record-suite:
	$(PYTHON) -B scripts/record_suite.py

record-transcendentals:
	$(PYTHON) -B scripts/record_transcendentals.py

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

coverage-validate: rust
	$(PYTHON) -B scripts/coverage.py validate

coverage-record: rust
	$(PYTHON) -B scripts/coverage.py record

coverage-check: rust
	$(PYTHON) -B scripts/coverage.py check --timing

coverage-properties:
	$(PYTHON) -B scripts/coverage.py properties

coverage-status:
	$(PYTHON) -B scripts/coverage.py status

spirv-build:
	$(PYTHON) -B scripts/spirv_task.py build

spirv-run: rust spirv-build
	$(PYTHON) -B scripts/spirv_task.py run all --size small --repeat 5
	$(PYTHON) -B scripts/spirv_task.py run all --size large --repeat 20

spirv-lavapipe: spirv-build
	VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json $(PYTHON) -B scripts/spirv_task.py run all --size small --repeat 2 --out build/spirv-lavapipe

spirv-sim: rust
	@set -e; for d in spirv/compiled/mesa-*; do echo "== $$d"; $(PYTHON) -B scripts/spirv_task.py simall $$d --work build/spirv-sim/$$(basename $$d); done

spirv-compile: rust spirv-build
	scripts/build_mesa_shim.sh $(MESA_VERSION) $(MESA_SHIM_DIR)
	$(PYTHON) -B scripts/spirv_task.py compile all --mesa $(MESA_SHIM_DIR)/mesa-$(MESA_VERSION) --out spirv/compiled/mesa-$(MESA_VERSION)

spirv-predict: rust
	$(PYTHON) -B scripts/spirv_task.py predictall spirv/compiled/mesa-$(MESA_VERSION) --size large --work build/spirv-predict

verify-conversions: rust
	$(PYTHON) -B scripts/verify_conversions.py

verify-float-edges: rust
	$(PYTHON) -B scripts/verify_float_edges.py
	$(PYTHON) -B scripts/verify_rounding_fixup.py --check

verify-transcendentals: rust
	$(PYTHON) -B scripts/trans_check.py

verify-sr: rust
	$(PYTHON) -B scripts/sr_check.py

verify-traces: rust
	rust/target/release/suite_check --traces

spirv-verify: rust
	$(PYTHON) -B scripts/spirv_task.py verify

asahi-verify:
	scripts/asahi_verify.sh
