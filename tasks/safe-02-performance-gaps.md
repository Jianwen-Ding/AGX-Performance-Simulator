# Task safe-02: close the largest cost-model gaps

Read [safe-probes.md](safe-probes.md) first (safety rules: AC power, clock state, device check).

**Goal:** measure the behaviours behind the worst predictions in `results/2026-10-16-asahi-retime` and fold them into
the cost model (`rust/src/cost.rs`) with an offline check.

**Timing:**
- Use the two-timestamps-before Vulkan runner (`tests/vk_run.c`) or direct-DRM firmware timestamps.
- Report the fast-clock cluster (runs within 5 % of the minimum) and the power source.
- Use bounded kernels only: straight-line code or emulator-gated `agx_loop` loops.

## 1. Same-address atomic contention (model off by 100x)

- Today the model predicts no slowdown at all. The `atomcontend` probe measured 3,633 us for 1 hot counter against
  154-170 us for at least 32,768 counters.
- **Direct-DRM microbenchmark:** device atomic add (validated form: `agx_loop.atomic_add`) from L lanes to K
  distinct addresses, with L = 32 .. 65,536 and K = 1 .. 65,536.
  - Also run 1 lane per SIMD group, all lanes of a SIMD group, and one address per core.
  - Also run threadgroup-memory atomics.
- Derive the serialisation rate per address (ns per atomic), how it scales with the number of cores, and whether
  lanes of one SIMD group combine.
- **Model change:** serialise same-address device atomics at the measured rate. Verify against `atomcontend`,
  `cellsort`, `hough` and `csrbuild`.

## 2. Strided and uncoalesced access (model over-penalises about 5x)

- `coalesce` measured 5.96x for stride 16, while the model predicts 31x.
- Use `agx_loop.stream_body` with element strides 1-64 words and 1/2/4/8 loads per round, footprints in L1 / L2 /
  SLC / DRAM.
- Fit how many 64-byte segments per cycle the L1 really serves for scattered lanes.

## 3. Barrier cost after ALU work (model 2.6x low)

- `barriercost` measured 1.57 us per round against the model's 0.60.
- Reuse the playground's `barrier-stress` / `sync-probe` kernels (0x68 and 0x78 forms, already safe), varying the
  ALU work before the barrier, the SIMD groups per workgroup (1-32) and the workgroups per core.

## 4. Store write-back cost (checklist Q2b)

- The stream `fill` dispatch (pure stores, 225 us) costs about as much as `read` (222 us). Neither the default nor
  `AGX_CHARGE_WRITEBACK=1` matches.
- Measure store-only, load-only and load+store streams at footprints from 1 MiB to 1 GiB.
- Fit the DRAM cost per dirty line, then fix `charge_writeback`. Verify with adamw and stream.

## 5. Throughput of the PTX-compliant sin / cos

- The fma-split sequence (`scripts/record_sincos_chain.py` `sin2` / `cos2`) adds one fmul and turns an fmul into an
  fmadd.
- Measure its throughput and latency against Mesa's sequence in a bounded loop, so ZLUDA knows what PTX compliance
  costs.

## Acceptance

- `make verify-cost-model` is updated deliberately: re-baseline only with a recorded justification.
- `verify-dispatch` and `verify-scaling` do not regress.
- `spirv_task.py verify all --timing` improves on the affected workloads.
- Results go in a new `results/<date>-...` directory and are pushed.
