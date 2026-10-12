# Task: record the AGX coverage suite on the M2 (Asahi Linux)

**Goal:** populate `coverage/expected/` and `coverage/timing/` with hardware results for every case in
`scripts/coverage_catalog.py`. Then report where the functional simulator and cost model disagree with them.

The catalog is complete and validated offline. No expected values exist yet; producing them is this task.
See `coverage/README.md` for what the catalog covers.

## Prerequisites

- An M2-family Mac running Asahi Linux with `/dev/dri/renderD128` (the AGX render node). The scripts assume G14 (M2 Max). Record the actual device in your report.
- `cc`, Python 3 with numpy, and Rust (`cargo`).
- The scratch directory `/scratch/apple-m2-playground` must be writable. Override it with `AGX_SCRATCH`.
- The hang budget in `scripts/gpu_safety.py` (default 30 recovered GPU timeouts per boot, `AGX_HANG_BUDGET`).
- applegpu is **not** needed.

## Steps

1. **Build and validate offline.** Both must finish with zero failures before anything touches the GPU.

   ```sh
   make rust
   make coverage-validate
   ```

2. **Record the safe cases.** This is resumable: re-running skips cases that already have a file.

   ```sh
   make coverage-record                                   # all areas
   python3 -B scripts/coverage.py record --area alu       # or one area at a time
   python3 -B scripts/coverage.py status                  # progress table
   ```

   - Functional cases take two dispatches each, roughly 25 minutes for all ~2,100 safe ones.
   - Timing cases (`--area bench`, 651 forms, three modes each) take several hours. Run them separately, possibly with `--modes latency,simd_issue` first.
   - A case that faults is written to `coverage/faults.json` and skipped on later runs. Investigate each one; a safe case faulting is itself a finding.
   - Do not delete `faults.json` entries to retry blindly.

3. **Record the risky cases**, one area at a time, watching the hang budget (`python3 scripts/gpu_safety.py status`).

   ```sh
   python3 -B scripts/coverage.py record --risky --area geometry
   python3 -B scripts/coverage.py record --risky --area alu --match 'funop|b28_6'
   ```

   - The recorder stops when the budget is used up. Reboot and resume.
   - After about 44 recoveries in one boot, the device previously wedged.

4. **Check offline:**

   ```sh
   make coverage-check                                    # suite_check coverage/expected + timing vs model
   python3 -B scripts/coverage.py check --timing -v       # every timing row
   rust/target/release/suite_check coverage/expected -v --only alu
   ```

5. **Commit the recordings and report:**
   - Commit `coverage/expected`, `coverage/timing`, `coverage/faults.json` and `coverage/log.jsonl`.
   - Write a short report (`results/<date>-coverage/README.md`) covering:
     - pass counts per area;
     - every simulator mismatch, grouped by cause;
     - every fault;
     - the timing rows where the model is off by more than 10%;
     - nondeterministic cases (`"deterministic": false` in a case file).
   - Fix the simulator (`rust/src/fsem.rs`, `func.rs`, `cost.rs`) where the cause is clear. Otherwise leave the case failing and describe it.

## Not automated: hand-built cases (`coverage_catalog.MANUAL`)

Each item needs an encoding built from its finding or a Mesa capture. Add each as a `Case` in the catalog once the
encoding exists, so it is recorded and checked like the rest.

| Item | Source |
|---|---|
| fmadd 10-byte lane-offset form | fmadd-lane-offset finding (encoding table) |
| 49 non-faulting undecoded opcodes | opcode-descriptions finding; record register effects |
| 0x18 fixed-cost op; 0x08 / 0x88 ending opcodes | sync-and-end-opcodes |
| stack_store / stack_load / stack_get_ptr / stack_adjust | stack-via-mesa (capture Mesa's scratch-using shader) |
| uniform_store from a preshader (`AGX_PRESHADER=1`) | preshader-execution, sr17-sr32 |
| SR12 / SR13 / SR15 under imageblock launches | sr12-sr13-sr15-imageblock |
| cross-core message passing (f533 release vs memory_barrier) | memory-ordering (record outcome counts, not words) |
| occupancy limits per register / shared-memory budget | occupancy |
| CPU-written timer read, monotonic per lane | 2026-10-11-in-shader-clock |

## Rules

- Never run an encoding that `validate` rejects, and never bypass `gate()`. `record` runs the gate before every dispatch.
- Keep `may_fault` cases behind `--risky`. Do not reclassify a case as `safe` until it has run cleanly.
- Recordings are append-only evidence: re-record (`--redo`) only with a reason, and note it in the report.
