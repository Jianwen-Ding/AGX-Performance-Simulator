# Task safe-06: record the 214 "may fault" special-register coverage cases

Read [safe-probes.md](safe-probes.md) first.

**Goal:** record `coverage/expected/geometry/` for the 214 geometry cases the catalog marks `may_fault`, then check them
with the simulator, which now models every selector except SR113.

## What the cases are

- `scripts/coverage_catalog.py` builds `sr_case(n, geom)` for every selector n = 0..255.
- The 42 selectors in `_sr_known()` run under 10 launch geometries and are recorded.
- The other **214 selectors** get one case each, under the first direct-DRM geometry (grid 1024x1x1, workgroup
  256x1x1). Each runs one straight-line `get_sr r14, sr<n>` and a store (`geometry/sr<n>_1024x1x1_wg256x1x1`).

## Why it is safe

Every one of these selectors has already been read on this GPU with the same 32-bit `get_sr` encoding, with no
fault:
- 126+ direct-DRM full scans (all 256 selectors per dispatch; playground `experiments/registers/sr-direct-drm.md`);
- the 2026-10-16 divergence and VM-slot runs (`run_sr_divergence.py`, `run_sr_vm_slot.py`).

The `may_fault` label predates those scans. `_sr_known()` is also stale: the 2026-10-16 work identified SR3, 11, 14,
16, 24, 25, 97, 113, 114 and the rest (playground `novel-results/2026-10-16-special-registers`).

## Steps

1. Check `python3 scripts/gpu_safety.py status`, then validate offline:

   ```sh
   make rust && make coverage-validate
   ```

2. Record **one** case first and check it:

   ```sh
   python3 -B scripts/coverage.py record --risky --area geometry --match 'geometry/sr3_1024x1x1_wg256x1x1$'
   rust/target/release/suite_check coverage/expected --only geometry -v | tail -3
   ```

3. Record the rest in batches of 50 (`record` skips cases that already have a file, so repeat the same command),
   re-checking the hang budget between batches:

   ```sh
   python3 scripts/gpu_safety.py status
   python3 -B scripts/coverage.py record --risky --area geometry --limit 50
   python3 -B scripts/coverage.py status       # geometry "risky pend" goes 214 -> 0
   ```

   - Stop on the first fault or timeout. It would contradict the scan evidence and is a finding in itself.
   - SR21 (VM slot) and SR113 (power-state bit) are recorded as nondeterministic, which is expected.
   - SR17 and SR20 depend on placement.
4. **Check:** `make coverage-check`.
   - Expected: every new case PASS, except SR113 (unmodelled by design) and SR21 (nondeterministic).
   - Anything else is a finding: update the playground's special-register map and `GroupCtx::sr`.
5. **Housekeeping.**
   - Update `_sr_known()` to the full identified set, so the recorded selectors are classed `safe`.
   - Update the coverage table in `coverage/README.md`.
   - Commit `coverage/expected/geometry/*`, `coverage/log.jsonl` and the catalog change, then push.

## Acceptance

- `coverage.py status` shows geometry risky pending = 0 and faults = 0.
- `make coverage-check` passes, with only SR113 and SR21 unsupported.
