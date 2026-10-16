# fp32 boundary sweeps (safe-01 part 3)

`scripts/record_float_edges.py` (GPU, agx_run-gated) recorded 13 sweeps, 291 M inputs: `fcmpsel` × 7 condition codes
over all edge pairs, `floor` / `ceil` / `trunc` / `rint` over edge-dense binades, f32 → f16 and f16 → f32 via `fadd32`
with −0.0. `hw_samples.json.gz` keeps every input of the small sweeps and ~24k samples of the large ones, plus the
SHA-256 of every full result array. The full arrays live in `/scratch/agxsim-data/float_edges`.

`rounding_fixup.json.gz` holds the GPU results of the non-`.ftz` `ceil` / `floor` fix-up
(`scripts/verify_rounding_fixup.py`), which equal the IEEE oracle on all 57,439 inputs.

`make verify-float-edges` replays all of it through `agx_run` and checks the semantic claims made in the playground's
`novel-results/2026-10-16-conversions`.
