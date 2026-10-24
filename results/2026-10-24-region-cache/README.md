# LLVM region-node cache invalidation

LLVM `8c350523dde0` tracks regions with nonempty node caches. Global
invalidation clears these maps immediately, without recursively visiting
all empty regions after every region-pass invocation. Local recursive
clearing, cache destruction, analysis moves and release/recalculation update
the owner set. Move assignment also releases the destination's previous tree.

The functional fixture in LLVM `46aff3aba735`, `region-cache-nested.ll`, has
nested lane-dependent diamonds inside a loop. `scripts/verify_llvm_region_cache.py`
checks all 32 output lanes against an independent integer sum for trip counts
0, 1, 4 and 9 on G13G/G14C. The simulator's grid size is in total threads;
32 grid threads and a 32-thread workgroup exercise all lanes in one workgroup.

With LLVM_LLC_BASELINE set to the pre-change compiler, all 256 current and
256 baseline output checks pass. Each target produces identical 256-byte
live code, simulator execution records, and predicted costs for both
compilers. The complete records are in `functional-cost.json`.

No simulator decoder, execution semantics, or hardware-cost parameters
changed. The optimization removes host compiler work. It does not predict a
GPU execution improvement for unchanged shader code.

```bash
LLVM_LLC=/scratch/llvm-agx-build/bin/llc \
LLVM_LLC_BASELINE=/scratch/agx-region-cache/llc-before \
LLVM_OBJCOPY=/scratch/llvm-agx-build/bin/llvm-objcopy \
AGX_RUN=/scratch/agxsim-target/release/agx_run \
AGX_COST=/scratch/agxsim-target/release/agx_cost \
make verify-llvm-region-cache
```

The direct script and Make target both pass. The baseline override is
optional; without it the independent output checks still run. The LLVM
regression suites pass 133 available tests (three AMDGPU-only tests are
unsupported), plus eight new cache-lifetime unit tests. Blender's measured
partial compilation wall time falls from 137.784 to 105.882 seconds, with
byte-identical intermediate IR; full shader compilation and rendering are
separate validation steps.

## Address-candidate compaction

The next LLVM checkpoint, `c1a33023280a`, replaces repeated individual
SetVector removals in AGXSinkAddresses with stable bulk compaction per
fixed-point round. Membership is erased as each rejected candidate is
visited, preserving pruning order and the final candidate set.

The same nested-loop gate passes with the pre-compaction compiler as its
baseline: all 512 lane outputs across both compilers match the reference,
and live code, execution and cost records agree. The Make target was also
rerun after this compiler change. See `address-sink-functional-cost.json`.
Three existing address fixtures additionally produce identical text for
G13G/G14C (`address-sink-fixtures.json`). All 79 AGX code-generation tests pass.

On the Blender surface checkpoint, address sinking now finishes in 1.367
seconds; the earlier resumed compilation timed out at 240 seconds while
still in that pass. A 50,000-candidate authored compilation fixture reduces
the pass from 0.2248 to 0.0045 seconds with byte-identical intermediate IR.
These are host compiler costs, outside the simulator cost model. Full Blender
compilation now exposes an unsupported constant generic-to-private pointer
cast; a small reproducer fails on both compilers, so rendering is not yet
validated. No shader timing parameters were adjusted.
