# Core structure from 1–32 SIMD groups per core: 4 schedulers, two halves with one complex unit and 2 register reads each

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, compute stage. Only
instruction forms that already ran on hardware (the 60-form bench) were used; 0 faults.
Cycle figures are reference units (dependent 8-byte `iadd` step = 2), as in the instruction bench;
real cycles ≈ value × 1.049.

## Claim and scope

Earlier throughput figures stopped at 4 SIMD groups per core, because with more groups the in-dispatch
`iadd` reference also needs the core's issue rate ([failed approaches](../../notes/failed-approaches.md)).
Here only the first SIMD group of each workgroup (SR53 = 0) runs the reference while the others wait at a
`threadgroup_barrier` (0x68). The timed phases are bracketed by barriers, so group 0's timestamps span
every group's work. One workgroup of G groups (32·G lanes) lands on each core (checked per workgroup
with SR20; the ~4 cores that receive two workgroups are discarded). The quantity measured is **core cycles
for one step of every group** (each group issues one instruction per step), over 1, 2, 4… 32 groups and
1 (latency-bound) or 8 (throughput-bound) dependency chains, for 23 instruction forms.

| G groups | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 | 24 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fadd32 x, x, 0.125`, 8 chains | 1.02 | 1.05 | 1.05 | 1.05 | 2.24 | 2.02 | 2.59 | 3.00 | 4.17 | 6.22 | 7.93 |
| `fadd32 x, x, c` (2 reg reads) | 1.02 | 1.06 | 2.03 | 2.03 | 3.03 | 3.03 | 3.96 | 5.97 | 7.91 | 11.86 | 15.61 |
| `fmadd32` (3 reg reads) | 2.00 | 2.03 | 3.11 | 3.11 | 4.64 | 4.63 | 6.00 | 9.12 | 12.21 | 18.14 | 24.38 |
| `imadd` | 2.00 | 2.02 | 3.94 | 3.97 | 5.86 | 5.86 | 7.88 | 11.80 | 15.71 | 23.43 | 31.36 |
| `log2` | 1.99 | 2.01 | 4.31 | 4.31 | 5.93 | 5.86 | 7.82 | 11.74 | 15.65 | 23.48 | 31.30 |
| `rcp` | 2.94 | 2.98 | 5.68 | 5.68 | 8.64 | 8.66 | 11.76 | 17.59 | 23.44 | 35.21 | 47.01 |
| `rsqrt` | 3.93 | 3.96 | 7.70 | 7.69 | 11.61 | 11.59 | 15.58 | 23.51 | 31.31 | 46.85 | 62.69 |
| `iadd` 64-bit | 3.94 | 3.97 | 7.85 | 7.62 | 11.81 | 11.83 | 15.73 | 23.56 | 31.26 | 46.93 | 62.75 |
| `simd_shuffle` | 1.99 | 2.02 | 1.99 | 1.98 | 4.29 | 4.34 | 4.35 | 6.60 | 8.53 | 12.44 | 16.36 |
| `simd_iadd` (reduction) | 23.11 | 23.48 | 23.47 | 23.43 | 29.54 | 30.02 | 28.00 | 45.70 | 77.11 | 93.13 | 130.74 |

The data fit this structure:

* **4 schedulers per core, 1 issue per cycle each.** Up to 4 groups issue in parallel, at 5, 6 and 8
  groups a step takes 2 cycles, and at 24–32 groups the core sustains **4.0 instructions per cycle**
  (128 lanes, Apple's figure). Groups of one workgroup get hardware slots 0…G−1 (SR17); with whole
  workgroups, scheduler = slot mod 4 fits within ~10% (largest deviations at 9–10 and 13–14 groups).
* **An instruction's issue interval blocks its scheduler**, not only its group: `simd_shuffle`
  (interval 2) takes 4.3 cycles per step with 5 groups (2 on one scheduler) where 1-cycle forms take 2.
  Reductions are the exception: they block only their own group.
* **Two halves**: schedulers {0, 2} and {1, 3}. Each half has:
  * **one unit for transcendentals, integer multiply-add, bit-field ops, conversions, floor/rint and
    64-bit add**, shared by its two schedulers. Two groups (one per half) run at the single-group rate;
    a third group doubles the time; 3 groups cost as much as 4 and 5 as much as 6. Occupancy per
    instruction: **1.96** (imad, bit-field, convert, floor, log2), **2.94** (rcp), **3.92** (rsqrt,
    64-bit add). Core rates: 1.0, 0.68 and 0.51 instructions per cycle.
  * **2 32-bit register-operand reads per cycle**: `fadd32` with a register constant takes 2.0 cycles at
    3 groups and 16 at 32 (2 reads × 16 instructions ÷ 2), `fmadd32` 3.1 and 24.4.
* A source written by the same group a few cycles earlier needs no register-file read: one dependent
  chain is cheaper than 8 independent ones for `fadd32` (11.7 vs 15.6 at 32 groups) and `icmpsel`.
* SIMD reductions sustain 0.245 (`simd_iadd`), 0.28 (`simd_prefix_iadd`) and 0.65 (`quad_iadd`)
  instructions per cycle per core; shuffles and ballots ~2; `quad_shuffle` 3.5; `fadd16` 3.5; `fmadd16` 2.3.

**Correction to the instruction bench.** Its 4-group throughput column used 16/64-step loop bodies;
those understate contention for the shared units by up to 2× (`rsqrt` 0.97 cycles per group-step with
16/64 steps, 2.0 with 32/128, in both the bench's span method and this probe; workgroup size 32, 64 or
128 lanes makes no difference). Use this finding's rates for more than one group per core.

## Not established (open)

* **The group→scheduler assignment is not a static function of the slot.** With 5 groups working
  (slots 0–4) a step takes 2.0 cycles. Giving slot 1, 3 or 4 no work brings it to ~1.05, but idling
  slot 2 leaves 2.0. No static slot→scheduler map with 1 issue per cycle fits all four cases.
* **Latency-bound chains between 10 and 20 groups**: one dependent `iadd` chain per group takes 5.6
  cycles per step at 16 groups (8 chains: 4.3; 12 groups: 2.6), and similar 20–60% excesses appear for
  shuffles, ballots and fp32/fp16 forms. The cause is unknown.
* `icmpsel` and `fmadd16` read fewer ports than their operand counts suggest; groups beyond 32 per core
  (up to 96, [occupancy](../2026-10-11-occupancy/README.md)); groups from several workgroups.

## Reproducer and evidence

* Hardware: `python3 -B scripts/scale_probes.py ops --groups 1,2,3,4,5,6,8,10,12,16,20,24,32 --chains 1,8
  --trips 500 --ref-trips 500 --name grid.json` (~10 min); idle slots: `--groups 5,5/1,5/2,...`.
  The probe also lives in [AGX-Performance-Simulator](https://github.com/Jianwen-Ding/AGX-Performance-Simulator)
  (`scripts/scale_probes.py`), whose cost model now implements this structure.
* Offline: `make verify-core-scaling` (in Apple-M2-Playground) (18 checks over [the evidence](https://github.com/Jianwen-Ding/Apple-M2-Playground/tree/main/results/2026-10-11-core-scaling)).

## Relationship to existing knowledge

Extends the [instruction bench](../2026-10-11-instruction-bench/README.md) (≤ 4 groups, register-read
limit ~4 per core: here split into 2 per half), the [transcendental pipe](../2026-10-11-transcendental-pipe/README.md)
(one shared pipe per core: here one unit per half, same instruction classes plus 64-bit add) and the
[barrier finding](../2026-10-11-barrier-0x68-vs-0x78/README.md) (0x78 covers aligned blocks of 4 groups).
Apple describes 128 FP32 ALUs per core; the 4 × 32-lane schedulers are consistent with that. The
half structure and the per-unit costs are measured here.

## Compiler implications

* Keep at least 8 SIMD groups per core for latency-bound code; only immediate-operand simple ALU code
  reaches 4 instructions per cycle.
* Transcendental, integer-multiply, bit-field, conversion and 64-bit add instructions share
  **two units per core**: a kernel with one such instruction in every 4 runs at ≤ 1 instruction per cycle.
  `rsqrt` and 64-bit adds cost 2× a `log2`/`imad`; prefer 32-bit address arithmetic and `rcp`+multiply
  forms where accuracy allows.
* Register operands limit throughput to 2 reads per cycle per pair of schedulers: fold constants into
  immediates, and reuse just-computed values (forwarded reads are free).
* Warp reductions (`__reduce_*_sync`, warp scans) peak at ~1 per 4 cycles per core; use shuffle trees
  when many values are reduced.
