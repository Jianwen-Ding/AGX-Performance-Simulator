# Dispatch launch cost, workgroup placement over clusters, and a validated whole-dispatch model

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, compute stage, known-safe forms
only; 0 faults.

## Claim and scope

**Launch cost.** Empty dispatches (firmware start/end, minimum of 8 after a warm-up dispatch):

| Threads | Workgroup | Time | Busiest core |
| --- | --- | --- | --- |
| 4 M | 256 | 67 µs | 456 workgroups, 3,648 SIMD groups |
| 16 M | 256 | 195 µs | 1,821 workgroups |
| 16 M | 1024 | 181 µs | 456 workgroups, 14,592 groups |
| 16 M | 64 | 688 µs | 7,282 workgroups |
| 8 M | 32 | 682 µs | 7,282 workgroups |

Per core, launching costs about **max(100 ns per workgroup, 13.6 ns per SIMD group)**, plus ~5.6 µs fixed per
dispatch (a 32-thread empty dispatch). Workgroups of 32 or 64 threads are launch-bound at ~100 ns each
regardless of their work; 256 threads and above cost ~14 ns per SIMD group.

**Placement.** Each lane records SR20 (core) and SR0 (workgroup). Workgroups are split evenly over the 4
clusters (10, 9, 9 and 10 cores) and then over each cluster's cores, so the busiest core holds
**ceil(ceil(n / 4) / 9)** workgroups: 38 workgroups use 34 cores (4 take two), 152 put 5 on 8 cores. This held
for 38–228 workgroups of 64 and 256 lanes, short and long kernels. Above that, workgroups are taken as slots free
and the load approaches n / 38. The dispatch time follows the busiest core: one ALU-bound kernel took 212, 321,
406, 503, 603, 698 and 787 µs for 2–8 workgroups on the busiest core (~95 µs each).

**Clock.** Dispatches shorter than ~1 ms often stay at a low DVFS level for the whole run: the same kernel took
484 µs in one process and a steady 1,055 µs in the next. Time dispatches of several ms, after a warm-up dispatch,
and use the minimum.

**Whole-dispatch model.** AGX-Performance-Simulator's `agx_dispatch` combines occupancy, this placement and
launch cost, the [core structure](../2026-10-11-core-scaling/README.md) and the
[memory hierarchy](../2026-10-11-memory-hierarchy/README.md), simulating the busiest core's workgroups with
per-group traces. Against 29 dispatches (empty, ALU-, latency-, shared-unit-, register-read-, DRAM-bound and divergent;
1,216 to 16 M threads; 32–1,024-lane workgroups; 64–192 register halves) at 1,398 MHz: **median error 5.4%,
p90 13.4%, max 21%**.

**Divergence.** A counted loop with `if (lane < k) { 64 fadd32 } else { 32 fadd32 }` in Mesa's structured form
(`if_icmp`, `jmp_exec_none`, `else_icmp`, `jmp_exec_none`, `pop_exec`), 155,648 threads:

| k (lanes taking the then-side) | 0 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- |
| dispatch time | 686 µs | 1,671 µs | 1,670 µs | 1,139 µs |

A SIMD group with lanes on both sides runs both sides in full (1,671 ≈ 686 + 1,139 − shared overhead), and where
the split falls inside the group does not matter; `jmp_exec_none` skips an empty side. With many groups per core
the exec-mask instructions (`if`, `else`, `push`, `pop`, `jmp_exec_none`) hold their scheduler for about 3 cycles:
the simulator otherwise under-predicts these dispatches by 12–19%; with that rule they are within −5% to +2%.

## Not established

Placement on other core counts; whether the cluster split is per cluster or per die half; dynamic balancing
details for many waves; launch cost with threadgroup memory or preambles; 2-D/3-D grids.

## Reproducer and evidence

* Hardware: `scripts/wg_placement.py --wg 256,64`, `scripts/dispatch_probes.py --repeats 8` (~5 min).
* Offline: `make verify-dispatch` here (launch-cost and placement checks over [the evidence](https://github.com/Jianwen-Ding/Apple-M2-Playground/tree/main/results/2026-10-11-dispatch/README.md));
  `make verify-dispatch` in AGX-Performance-Simulator re-predicts all 25 stored dispatches from their binaries.

## Relationship to existing knowledge

The [direct-DRM finding](../2026-10-10-command-timing-direct-drm/README.md) measured ~5.7 µs per empty command
and the clock ladder; the [SR finding](../2026-10-10-sr-geometry-core-id/README.md) gave SR20 = (cluster << 4) | core
over 10 + 9 + 9 + 10 cores. The per-workgroup / per-group launch cost and the placement rule are measured here.

## Compiler and runtime implications

* Use workgroups of ≥ 128 threads: 32- and 64-thread blocks pay ~100 ns of launch each per core.
* For grids that fit at once, the busiest core holds one workgroup more than an even split: a multiple of 36
  workgroups (4 × 9) avoids a straggler core; 38 workgroups take the time of 76.
* Benchmark with ms-scale dispatches after a warm-up; sub-ms timings mostly measure the DVFS state.
