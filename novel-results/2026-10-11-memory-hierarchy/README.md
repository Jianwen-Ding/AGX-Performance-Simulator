# Device-memory throughput with up to 32 SIMD groups per core: 64-byte segments, L1 8 KiB, L2 ~256 KiB per core, ~410 GB/s DRAM

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, compute stage. Only load and
store forms that already ran on hardware (pointer-chase `device_load`, the control's `device_store`, the
int64 finding's register-pair base) were used; 0 faults. Cycle figures are reference units (× 1.049 = real).

## Claim and scope

Method: `scripts/mem_probes.py`, on the barrier-bracketed one-reference-group timing of the
[core-scaling finding](../2026-10-11-core-scaling/README.md). One workgroup of G SIMD groups per core runs a
counted loop of `device_load` (or `device_store`) instructions, `wait 0` after every 8. The result is core cycles
per memory instruction of every group. Lane i of a group addresses word `base + i·S` (S = stride in words),
so one instruction touches 1–32 distinct 64-byte segments.

**L1 (all addresses resident, 32 groups):**

| Stride S (words) | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64-byte segments per instruction | 1 | 2 | 4 | 8 | 16 | 32 | 32 |
| load, core cycles per instruction | 1.22 | 1.22 | 1.96 | 3.96 | 7.90 | 16.75 | 16.36 |
| store, core cycles per instruction | 1.96 | 1.96 | | 7.83 | | | 31.37 |

* **Loads: 2 segments per cycle per core** (128 B/cycle). A coalesced 128-byte load costs 1.0 cycle per core
  when groups touch different lines and 1.22 when all groups hit the same lines.
* **Stores: 1 segment per cycle per core**, at least 2 cycles per instruction.
* **One group** waits for its own segments: 8 loads + `wait` take 63 cycles with 1 segment each and 173 with 32.

**Capacity and bandwidth (each workgroup streams its own window; 32 groups per core):**

| Per-core window | 4–8 KiB | 16 KiB | 32–256 KiB | 512 KiB–4 MiB |
| --- | --- | --- | --- | --- |
| core cycles per coalesced load | 1.24 | 1.70 | 2.7–3.1 | 14–16.5 |
| served by | L1 | L1/L2 | **L2: ~0.71 segments (45 B) per cycle** | DRAM |

* L1 holds 8 KiB per core (the pointer-chase finding's first level), L2 ~256 KiB per core (256 KiB still at L2 rate,
  512 KiB DRAM-bound). 38 × 512 KiB = 19 MiB of private data already runs at DRAM speed, so the system-level
  cache adds little for private per-core data (a shared 16 MiB window read by every core was fast; see below).
* **DRAM: 7.9 bytes per reference cycle per core at 32 groups = ~410 GB/s over 38 cores** (256 MiB window; 4 MiB per
  core: 527 cycles per 32 loads). 8 groups per core already reach ~390 GB/s. Whole dispatches streaming 240 MiB
  reach ~338 GB/s including launch and tail ([dispatch finding](../2026-10-11-dispatch-launch/README.md)).
* **ALU work after a `wait` delays the next whole batch of loads**: 4 ALU instructions add ~27 cycles to an 8-load
  round (the load-after-ALU quiet time), each further independent instruction ~1. The quiet time holds back
  every load of the batch, not only the first (loads complete in order).
* **A load issued after ALU work waits ~50 cycles behind the previous load, not for its completion**: one `iadd`
  after every load (private windows, one group) costs 50.5 cycles per load at L1, 57 at L2 and 75 at DRAM. A full
  wait for completion would cost 135–500. So the earlier "load after ALU waits for the pending load" rule (measured
  on L1 hits) is a wait for the pending load to pass the L1, plus any completion deferral by ALU work.
* **CPU load does not disturb these measurements**: with 10 CPU cores busy, L1 and DRAM-bandwidth rows moved < 2%.
  L2 rows vary ±10% between runs either way, and latency-bound DRAM rows with 1 group vary with the GPU clock
  (latency is fixed in ns). Use medians of repeats.

## Not established

L2 between 256 and 512 KiB; associativity; reuse of lines that other cores fetched (shared windows of
1–64 MiB read by all cores ran 2–5× faster than a per-core model predicts); store bandwidth to DRAM;
vector (multi-component) and 8/16-bit accesses; atomics; texture paths.

## Reproducer and evidence

* Hardware: `scripts/mem_probes.py run` (coalescing `--stride 0,1,…`, `--kind store`, `--private --mask-bytes …`,
  `--global` streaming, `--pad N --pad-kind indep|dep|idx`). Also in AGX-Performance-Simulator, whose memory
  model implements these rules (`make verify-memory-model`: stored sets at 1.8–7.2% median error).
* Offline: `make verify-memory-hierarchy` (8 checks over [the evidence](https://github.com/Jianwen-Ding/Apple-M2-Playground/tree/main/results/2026-10-11-memory-hierarchy/README.md)).

## Relationship to existing knowledge

Extends the [load-latency finding](../2026-10-10-agx-load-latency/README.md) (one group, one address per
load: L1 ≤ 8 KiB, second level to ~64 KiB, ~350 GB/s streaming) and the [load-after-ALU finding](../2026-10-11-load-after-alu/README.md).
Apple advertises 400 GB/s for the M2 Max; the 64-byte segment granularity and per-core rates are measured here.

## Compiler implications

* Coalesce: consecutive lanes on consecutive 4-byte words cost 2 segments per instruction; a 16-byte lane stride
  costs 4× and a ≥ 64-byte stride 16×. Stores cost twice as much per segment as loads.
* Keep a workgroup's hot data ≤ 8 KiB per core for L1 and ≤ ~256 KiB for L2 (shared-memory tiling targets).
* Keep ≥ 8 SIMD groups per core for bandwidth-bound kernels.
* Issue loads before ALU work, not after: ALU instructions between a `wait` and the next loads delay the whole batch.
