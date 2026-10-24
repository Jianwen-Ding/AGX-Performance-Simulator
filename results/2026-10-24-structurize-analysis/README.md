# Conditional legacy LLVM structurization analysis

LLVM `2f73959ea7b7` removes the unconditional legacy UniformityInfo pass
from StructurizeCFG. The pass computes uniformity once, before rewriting
its first region, only when the effective skip-uniform-regions option is
true. It retains the original-function analysis across regions and resets
it between functions. AGX uses false by default, so needs no such analysis.
The new pass-manager implementation already makes its analysis conditional.

No ISA decoder, functional model, or cost parameters changed. The shared
LLVM fixture `structurize-analysis-lazy.ll` checks the legacy pass schedule,
the library-option override, lane-dependent branches, and analysis lifetime
across three functions. The available LLVM suites pass 103 tests (78 AGX,
25 generic structurization); three AMDGPU-only tests are unsupported by this
AGX-only build.

`scripts/verify_llvm_structurize_analysis.py` compiles the three shared
kernels separately with skipping disabled/enabled on G13G and G14C. It
checks taken and untaken uniform branches against independent output values
42/24 or the unchanged sentinel 99, and the lane-zero-only divergent store
against 42. All 20 simulated output checks pass. Each pair also has identical
code, simulator execution, and predicted cost. Live code is 54 bytes for
each uniform fixture and 58 bytes for the divergent fixture on both targets.
The full numeric records are in `functional-cost.json`.

```bash
LLVM_LLC=/scratch/llvm-agx-build/bin/llc \
LLVM_OBJCOPY=/scratch/llvm-agx-build/bin/llvm-objcopy \
AGX_RUN=/scratch/agxsim-target/release/agx_run \
AGX_COST=/scratch/agxsim-target/release/agx_cost \
make verify-llvm-structurize-analysis
```

The Make target and direct script were both executed successfully with the
existing release simulator binaries (Cargo target overridden to scratch).
This is a compiler-analysis change, not an established GPU execution speedup.
The large Blender shader still exceeds the bounded structurization check;
pass-manager cache clearing is the next measured bottleneck to investigate.
