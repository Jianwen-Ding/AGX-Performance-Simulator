# Early LLVM device-function pruning

LLVM commit `07a2e4f1702f` adds GlobalDCE between AGX device-function
internalization and forced inlining. This avoids expanding unreachable call
chains. It uses existing instructions and requires no ISA or cost-model change.

`scripts/verify_llvm_pruning.py` compiles the LLVM regression fixture with
the pass enabled and disabled for `agx-g13g` and `agx-g14c`, checks machine
instructions, and runs both the functional and cost simulators. Both CPUs
produce identical 34-byte live code, identical execution statistics, the
expected stored value 42, and identical predicted cost across the variants.
The fixture also checks that externally visible global data and a live
helper call survive pruning before inlining.

Reproduce from the simulator repository:

```bash
LLVM_LLC=/scratch/llvm-agx-build/bin/llc \
LLVM_OBJCOPY=/scratch/llvm-agx-build/bin/llvm-objcopy \
AGX_RUN=/scratch/agxsim-target/release/agx_run \
AGX_COST=/scratch/agxsim-target/release/agx_cost \
python3 scripts/verify_llvm_pruning.py
```

`make verify-llvm-pruning` builds the simulator binaries in the usual target
directory and runs the same check; LLVM tools must be on PATH or specified
through the variables above. The matching LLVM fixture is
`llvm/test/CodeGen/AGX/prune-before-inline.ll` in the sibling LLVM checkout.
All 75 AGX code-generation tests and the simulator's release tests pass.

On the M2 Max, alternating six standalone compilations of Blender's existing
camera-only optimized IR gave these wall-clock seconds:

| Early pruning | Run 1 | Run 2 | Run 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Off | 0.6572 | 0.6715 | 0.6561 | 0.6572 |
| On | 0.4277 | 0.4134 | 0.4071 | 0.4134 |

That is 37.1% less compile wall time for this one kernel. All six objects
have identical 44,222-byte `.text`, SHA-256
`48252ea19c0650f3a52e3011fa461e42f44df469d7d5964a8eabb9fa17c694ba`.
This does not establish a Blender render speedup or an improvement for other
applications. Cross-application measurements remain in progress. Timing
logs and objects are regenerable under `/scratch/agx-shared-optimization`.
