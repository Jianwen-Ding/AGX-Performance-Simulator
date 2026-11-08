# Conditional LLVM shuffle uniformity analysis

LLVM commit `a6fc58f02bcc` computes divergence only when the function has a
variable-index shuffle requiring a uniformity decision. Constant indices
and functions without shuffles skip that analysis. Candidate analysis is
built after matrix lowering, so it also sees shuffles introduced there.
No instruction encoding or cost-model parameter changes are required.

The sibling LLVM regression fixtures are `shuffle-analysis-lazy.ll` and
`matrix-fma-lane-inputs.ll`. The first rejects an eagerly scheduled legacy
uniformity pass. The second constructs different A elements in every lane
and multiplies by B=1, C=0. `scripts/verify_llvm_shuffle_analysis.py` computes
independent integer row sums and verifies all 64 float outputs per case:

| Target | Matrix path | Live code bytes | Executed instructions | Outputs checked |
| --- | --- | ---: | ---: | ---: |
| G13G | Emulated | 958 | 150 | 64 |
| G14C | Emulated | 958 | 150 | 64 |
| G14C | Native | 86 | 12 | 64 |

All 192 elements match. The pre-change compiler also passes this functional
check; this is a compilation-cost improvement, not a newly claimed matrix
semantic fix. The complete LLVM AGX code-generation suite passes 77 tests.

Reproduce from this repository:

```bash
LLVM_LLC=/scratch/llvm-agx-build/bin/llc \
LLVM_OBJCOPY=/scratch/llvm-agx-build/bin/llvm-objcopy \
AGX_RUN=/scratch/agxsim-target/release/agx_run \
python3 scripts/verify_llvm_shuffle_analysis.py
```

`make verify-llvm-shuffle-analysis` builds the simulator in its default target
directory and runs the same check. LLVM tools must be on PATH or overridden.

## Blender stage profile

The existing `module.surface-only.ll` has no shuffle requiring this analysis.
On the M2 Max, compilation stopped immediately before `structurizecfg`
takes 36.865 seconds before versus 31.689 seconds after. Backend pass wall
time is 21.105 versus 15.886 seconds; the removed uniformity pass accounts
for 4.515 seconds in the before profile. This is one paired measurement,
not a full shader compilation or render speedup.

Both generated intermediate IR files are byte-identical: 1,185,361,815 bytes,
SHA-256 `18ce910b75f8c7d6a151d3aee1dd32099df11f77e45219633241a2db3b56a9b8`.
The two large regenerable dumps were removed after hashing; timing logs and
the JSON summary remain in `/scratch/agx-shared-optimization`.

The PTX front end's separate host-cache concurrency change (`ed757a8`) does not change
simulated GPU behavior: eight simultaneous misses now call the compiler
once, cache failures still compile, and failed compilation releases the
lock. Its runtime tests pass 47 cases and driver tests pass 33 cases.
