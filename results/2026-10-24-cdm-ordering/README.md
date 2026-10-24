# Runtime CDM ordering and simulator scope

ZLUDA `4db02c1` makes every dispatch in a combined command emit its CDM
barrier, including when the shader code is already cached. The previous
cache-generation condition could omit ordering between dependent dispatches
and corrupt Qwen inference. The repaired opt-in path matches the CPU on
counting, arithmetic, weekdays and an ocean-sentence prompt.

The authored PTX dependency probe and verifier are retained in playground
commit `4f5d9ad`:

* [Probe source](https://github.com/Jianwen-Ding/Apple-M2-Playground/blob/4f5d9ad/tests/cdm_batch.ptx)
* [Hardware verifier](https://github.com/Jianwen-Ding/Apple-M2-Playground/blob/4f5d9ad/scripts/verify_cdm_batch.py)
* [Measurements and limits](https://github.com/Jianwen-Ding/Apple-M2-Playground/tree/4f5d9ad/results/2026-10-24-cdm-batch-ordering)

The probe runs 64 dependent increments over 65,536 elements and an
`inc -> twice` chain, requiring zero incorrect outputs and exactly 64
inter-dispatch barriers. Both the direct runner and the wrapper verifier
pass three repetitions. The runner's barrier-count assertion catches the
omission even when a simple kernel happens to execute in order.

Run this integration gate from the simulator checkout with the sibling
playground and ZLUDA checkouts available:

```bash
ZLUDA_AGX_ZOC=/scratch/geekbench-extract/cargo-target/release/zoc \
AGX_CDM_BATCH=/scratch/geekbench-extract/cargo-target/release/agx_cdm_batch \
make verify-cdm-ordering
```

This is a **hardware integration gate**, not an extension of the simulator's
supported input format. `agx_run` consumes shader dispatches; the device
model schedules abstract commands with explicit queue dependencies. Neither
decodes these CDM packet streams, and the shader cost model does not predict
host submission overhead. No ISA decoder or cost parameter was changed to
fit the command timings. Missing intra-command barriers remain outside the
functional model's direct coverage; the hardware gate covers that boundary.

The performance tradeoff is mixed. Warm Qwen tests improve prompt throughput
about 9% and reduce generation throughput about 7%. Geekbench graph-launch
host time falls from 12.866 to 6.486 seconds, but synchronization time rises
from 29.331 to 35.010 seconds and total runtime remains about 134 seconds.
The runs have additional cache-miss differences and fail network upload,
so they do not establish an overall speedup or published score. Batching
remains off by default pending better workload selection.
