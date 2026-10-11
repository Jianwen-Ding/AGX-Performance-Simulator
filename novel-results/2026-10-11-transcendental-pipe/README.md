# Transcendentals: which pairs mix freely depends on latency and a slow stage, not on a shared per-op cost

Status: one SIMD group: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM; 4 groups per core:
partly understood. Real cycles unless stated (1 reference unit = 1.049 cycles).

Found because the cost model's 4-group error correlated with `log2` (+0.46), `sin_pt_2` (+0.41) and
`convert` (+0.36) across 60 random kernels. The model charged each transcendental a fixed share of one
unit per core. Hardware does not.

## Throughput-bound pairs (8 independent chains, the two ops alternating)

| pair | 1 group: cycles / op | 4 groups/core: reference units per trip |
| --- | --- | --- |
| `log2`+`log2`, `floor`+`floor`, `log2`+`exp2` | 1.97–1.99 | 122 |
| `log2`+`rcp` | **1.97** (pure `rcp`: 2.94) | **122** (as pure `log2`) |
| `rcp`+`rcp` | 2.94 | 232 (4 chains instead of 8: 197) |
| `log2`+`floor` | **2.99** (worse than either alone) | 207, later 232 |
| `floor`+`rcp` | 2.91 | 194 |
| `rsqrt`+`log2` | 2.47 | 153 |
| `rsqrt`+`rcp` | 3.43 | 258 |

## Claims

1. **One SIMD group: a pipe with three rules** fits all nine pairs within 1.2%:
   * transcendentals issue at most one per **2 cycles** and occupy the group for 2 cycles (not their full issue
     interval: an `rcp` chain with 3 independent `iadd`s costs 72.6 units, chain-bound);
   * `rcp` and `rsqrt` also hold a **slow stage** for 3 and 4 cycles;
   * results are **written back in order, at least 2 cycles apart**. `floor` (latency 3) after `log2` (latency 5)
     waits for its writeback slot, so mixing latencies costs more than either op alone.
2. **4 groups per core: separate slow and fast units** with in-order writeback 1 cycle apart.
   `log2`+`rcp` costing the same as pure `log2` means `rcp` does not use the unit `log2` saturates.
   Unit costs come from the benchmark's 4-group rows (`rcp` 1.6, `rsqrt` 1.02, others 0.92 cycles), not from
   these probes; only the writeback gap was chosen on them (0 / 0.5 / 1.0 cycles). On the 14 probe cases the
   median error goes from ~17% to 7.3% (max 26%, `rsqrt`+`rcp`). Held-out random kernels at 4 groups:
   24 instructions 10.4% → 7.9%, 48 instructions 9.9% → 9.6%. A grid fit that used the probes for the unit
   costs (`sfu-coregrid.json`) needed `rsqrt` at 2.3 cycles and broke the `rsqrt` benchmark row (+126%).
3. **Not understood at 4 groups:** pure `rcp` costs 232 units with 8 chains but 197 with 4 (repeatable,
   `sfurepeat.json`); `rsqrt`+`rcp` costs 258 although `rsqrt` alone sustains ~1 per cycle per core.
   A chain-count sweep (`rcpchains.json`, 30–32 ops per trip) isolates it: `rcp` with 2 / 4 / 6 chains costs
   1.62–1.67 cycles per op at 4 groups (model 1.6), with 8 chains 1.88; `log2` with 8 chains is normal (1.00).
   So it is specific to 8 concurrent `rcp` chains, not to `rcp` or to 8 chains. Also from the sweep: latency-bound
   `log2` (2 chains) costs 152.5 at 4 groups vs 132.9 predicted, so a dependent transcendental's latency seems to
   grow by ~15% when other groups share the pipe (their writebacks interleave); not modelled. `log2`+`floor` measured 207 in one run and 232 twice later. The cost of mixing latencies
   across groups seems to depend on how the groups interleave.

## Integer multiply, bit-field and convert share the same writeback pipe (`sfuimad.json`)

| pair (8 chains, alternating) | 1 group | 4 groups/core |
| --- | --- | --- |
| `imadd`+`imadd`, `bitrev`+`bitrev`, `convert`+`convert` | 102 / 96 / 96 | 137 / 122 / 121 |
| `floor`+`imadd` (both latency 3) | **97** (mix freely) | 149 |
| `log2`+`imadd` / `bitrev` / `convert` (latency 5 vs 3) | **126 / 126 / 126** | 182 / 208 / 233 |
| `rcp`+`imadd` | 127 | 217 |
| `log2`+`fadd32` | 67.8 | 74.1 |

4. **The model's classification was wrong.** It treated `imadd`/`imul`, the bit-field ops (`bfi`, `bfeil`, `extr`,
   `shlhi`, `shrhi`, `asr`, `asrh`, `bitrev`, `popcount`, `ffs`, `intl`) and `convert` as a separate shared unit.
   They behave as latency-3 members of the transcendental pipe: mixing them with latency-5 transcendentals costs
   3 cycles per op at one group (exactly like `log2`+`floor`), and up to 2.3× the prediction at 4 groups.
   This is what the 4-group kernel error correlated with (`convert` +0.40, `bitrev` +0.27).
5. **ALU work issues in a transcendental's 2-cycle slot** (`log2`+`fadd32` 67.8 = pure `log2` chain), so
   per-probe the group should be blocked 1 cycle, not 2. On held-out random kernels 1-cycle blocking was clearly
   worse (24 instructions 5.4% → 8.5%, 48 instructions 3.6% → 7.8%), so the model keeps 2 cycles and reports
   `log2`+`fadd32` (82.6 predicted) as a residual. Something else in random code costs what the 2-cycle block
   happens to charge; it is not identified.
6. **Simulator artifact found:** round-robin-by-tick arbitration kept the 4 groups in lockstep, so every group
   reached the loop branch together and the transcendental pipe idled for the branch each trip (pure `log2`
   predicted 152 vs 122 measured). Hardware hides the loop overhead completely at 4 groups. A hashed per-tick
   rotation removes the lockstep; greedy-then-oldest made it worse (154).

## Effect on the cost model

Held-out random kernels (median whole-trip error, bias):

| set | before this finding | after (pipe incl. imad class, 2-cycle block, hashed arbitration) |
| --- | --- | --- |
| 48 instructions, 1 group | 3.6% | 3.8% |
| 24 instructions, 1 group | 5.3% | 5.7% |
| 48 instructions, 4 groups | 9.1% (bias −9.0%) | **4.3%** (bias −3.6%) |
| 24 instructions, 4 groups | 10.4% | 8.3% |
| selection set, 4 groups | 10.4% | 6.3% |

## Reproducer and evidence

`python3 scripts/cost_probes.py sfupairs`; `results/2026-10-11-cost-model/sfupairs.json`, `sfumix.json`,
`sfuswitch.json`, `sfurepeat.json`, `sfu-coregrid.json`, `sfuimad.json` (`cost_probes.py sfuimad`), `rcpchains.json`. Checked by `scripts/verify_cost_model.py`.

## Compiler implications

* Schedule transcendentals of equal latency next to each other: `log2`/`exp2`/`sin_pt_2`/`rcp` (5 cycles)
  interleave freely; putting `floor`/`rint`/`sin_pt_1` (3 cycles) between them costs up to 50% more.
* `rcp` is cheap when interleaved with `log2`/`exp2` and expensive back to back.
* Integer multiplies, bit-field ops and conversions (latency 3) conflict with 5-cycle transcendentals the same
  way: group them with `floor`/`rint`/`sin_pt_1`, away from `log2`/`exp2`/`rcp`.
