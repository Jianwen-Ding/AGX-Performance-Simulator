# Task safe-07: the 10-byte fmadd lane-offset form and the fmadd32 bit-52 operation

Read [safe-probes.md](safe-probes.md) first (safety rules).

**Goal:** turn the playground's `fmadd-lane-offset` finding into recorded coverage cases. Then answer its open
questions, and model the form in the simulator. A lane-shifted fma is a cheap stand-in for PTX `shfl.up` combined
with a multiply-add (prefix sums, stencils, convolutions), so the PTX front end may want it.

**Background** (playground `novel-results/2026-10-11-fmadd-lane-offset`, run cleanly on direct DRM):
- Bit 53 of fmadd32 (0x3a) and fmadd16 (0x36) selects a 10-byte form.
- In that form, bits 75-79 are a lane offset k for source A: lane i reads A from lane i - k. The shift stays inside
  16-lane halves by default.
- Bit 68 lets the shift cross from lane 15 to lane 16. Bit 72 limits it to groups of 2 lanes, bit 73 to groups of 4.
- Bit 52 of the 8-byte fmadd32 changes the operation to a select-like result whose purpose is unknown.
- The simulator does not model any of this today (`rust/src/isa`, `func.rs`).

## Safety

- Use only the exact bit patterns the finding ran: bits 52, 53, 68, 72, 73, the k values it tested, and bits 64-67
  and 69-71 with k = 1.
- Bit 74 alone gave an undefined value in lane 0. Record it, but mark the case nondeterministic.
- **Do not combine** bits the finding did not combine (for example 72 with 73, or 68 with 72), and do not set bits
  beyond those listed. Any new combination is a new encoding, so add it to the follow-up list instead of running it.
- Straight-line code inside `build_main()`, as in `run_drm_bitflip.py`.

## Steps

1. **Encodings.**
   - Add a `fmadd_lane(k, bits, width)` builder to `scripts/coverage_catalog.py` that emits the 10-byte form from the
     finding's encoding table.
   - Add cases for k = 0, 1, 2, 4, 8, 15, 16, 31, with and without bits 68 / 72 / 73, for both fmadd32 and fmadd16.
   - Add the 8-byte fmadd32 with bit 52 over the finding's 31 float triples.
   - Mark them `safe` only where the finding ran that pattern. Everything else goes behind `--risky` with a note.
2. **Record:** one case first, then the rest (`coverage.py record --match fmadd_lane`).
3. **Open questions** (same encodings, different data and code around them):
   - **Write-to-read distance for the 32-lane shift.** With bit 68, how many instructions must separate the write
     of A from the read before lane 16 sees lane 15? Sweep 0-8 independent `mov_imm`s between them.
   - **Inactive source lanes.** What does lane i read when lane i - k is masked off by the exec mask? Use the
     masked-case pattern from `alu_masked`.
   - **Bit 52.** Fit a rule over more triples: ±0, ±inf, NaN, B = -1.0, A = -1.0, and denormals.
4. **Simulator.**
   - Decode the 10-byte form in `rust/src/isa` and implement the lane read in `func.rs`.
   - Model bit 52 if a rule fits every recorded word. Otherwise return UNSIM for it.
   - `make suite-check` and `make coverage-check` must pass for the new cases.
5. **PTX mapping:** which `shfl.sync.up` + `fma` patterns this form can replace, and where its behaviour at the
   boundaries (0 at the start of a half) differs from `shfl` (which returns the lane's own value).

## Deliverables

- New coverage cases with recordings.
- The simulator model.
- An update to the playground finding (answers, or "still open").
- A `make verify-fmadd-lane` target. Push both repositories.
