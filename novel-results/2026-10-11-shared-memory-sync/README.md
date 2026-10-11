# Threadgroup memory, barriers and device atomics: throughput with 1-32 SIMD groups per core

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, compute stage. Only encodings
that already ran on hardware (Mesa-form `threadgroup_load/store` with a byte base register, `threadgroup_barrier`
0x68 and the 0x78 barrier, the preshader-validated device `atomic` add without return); 0 faults.
Cycles are reference units (× 1.049 = real), measured with the barrier-bracketed one-reference-group method of
the [core-scaling finding](../2026-10-11-core-scaling/README.md).

## Claim and scope

**Threadgroup memory (32 banks of 4 bytes).** Lane i of every group accesses byte `4·S·i` (`mem_probes.py --kind tgload|tgstore`):

| Stride S (words) | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| load, core cycles per instruction (32 groups) | 0.98 | 0.98 | 1.97 | 3.92 | 7.86 | 15.68 | 31.39 |
| store | 0.98 | 0.98 | 1.97 | | 7.83 | | 31.34 |

* One access per cycle per core; a k-way bank conflict costs k cycles; lanes reading the same word share one access.
* One SIMD group alone gets one threadgroup access per **~31 cycles (loads, 8 per wait)** or **~24 (stores)**,
  plus about one cycle per extra conflict way: a group's threadgroup accesses are serial. It takes ~32 groups
  per core to reach the core's rate with conflict-free loads.

**Barriers.**

| | 1 | 2 | 4 | 8 | 16 | 32 groups |
| --- | --- | --- | --- | --- | --- | --- |
| 0x68 back to back | 13.0 | 13.2 | 13.4 | 13.3 | 13.6 | 14.2 |
| 0x68 after 4 independent iadds | 42.1 | 43.8 | 44.8 | 54.6 | 72.1 | 105.5 |
| 0x78 back to back | 7.1 | 7.5 | 7.6 | | | |

* A barrier releases ~13 cycles (0x68) or ~7 (0x78) after the last group arrives, for any workgroup size up to 32 groups.
* A group arrives only after its in-flight ALU results drain (~24 cycles after the last ALU issue, like `wait`):
  4 iadds before a barrier turn 13 into 42 cycles.

**Device atomic add (no return).**

* Distinct words per lane (global index): **~1.7 core cycles per 32-byte sector, at least 6.9 per instruction**: 32
  consecutive words 6.9, lanes 32 or 128 bytes apart 54. Without a return value one group pays the same (fire and forget).
* Every lane of every core on one word: 619 core cycles per instruction, i.e. **~2 atomic operations per cycle GPU-wide** on one address.
* 32 words shared by all cores (no global index): 72–76 cycles per instruction; contention across cores on the
  same lines dominates.

## Not established

Threadgroup-memory capacity per core and its effect on occupancy; vector and 8/16-bit threadgroup accesses;
atomics with return values (latency) and threadgroup atomics; float atomics; barriers with divergent groups
or at more than 32 groups; why ALU work before a barrier costs more at 8–32 groups than issue time explains
(the simulator is 13–31% low there).

## Reproducer and evidence

`scripts/mem_probes.py run --kind tgload|tgstore|atomic [--global]`, `python3 -c "import scale_probes as S; S.gate(); S.barriers([1,2,4,8,16,32])"`
(hardware, minutes). Offline: `make verify-shared-memory-sync` (9 checks over [the evidence](https://github.com/Jianwen-Ding/Apple-M2-Playground/tree/main/results/2026-10-11-shared-memory-sync/README.md)).
AGX-Performance-Simulator models these rules (`make verify-memory-model`: threadgroup 2.2%, atomics 1.4%, barriers 5.2% median error).

## Relationship to existing knowledge

Extends the [barrier finding](../2026-10-11-barrier-0x68-vs-0x78/README.md) (scope of 0x68/0x78, 9.3 / 2.9 ns per
use for one group), the [threadgroup addressing finding](../2026-10-11-tg-memory-addressing/README.md) (byte base,
slot-scaled index) and the [atomics finding](../2026-10-11-atomics/README.md) (semantics). 32 banks of 4 bytes is the
conventional GPU layout; it is measured here for AGX.

## Compiler implications

* Shared memory: pad arrays so that lanes of a SIMD group fall in different banks (stride ≠ multiple of 2 words);
  a 32-word stride (e.g. a column of a 32×32 float tile) is 32× slower.
* One SIMD group cannot stream shared memory faster than ~1 access per 31 cycles: keep many groups per core,
  and prefer one wider access to several narrow ones.
* `__syncthreads()` costs ~13 cycles plus the ALU drain; batch work between barriers.
* Global atomics: a per-thread `atomicAdd` on distinct consecutive words is cheap (6.9 cycles per SIMD instruction);
  a single global counter is ~2 operations per cycle for the whole GPU: reduce within the SIMD group first
  (`simd_iadd`, ~0.25/cycle/core) and issue one atomic per group.
