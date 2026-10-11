# Device-load latency, overlap and read bandwidth on G14C

> **Scope note (2026-10-11).** These latencies were measured with no ALU instruction between a load's `wait`
> and the next load. A load issued after ALU work costs ~18 units (~19 cycles) more, and ALU work in flight
> defers completion; see [load after ALU](../2026-10-11-load-after-alu/README.md).

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM.

## Claim and scope

Dependent `device_load` (i32) + `wait 0` latency for one SIMD group:

* ≤ 8 KiB: ~36–38 ns warm, ~50 cycles in both the lowest and the top clock state.
* 12–64 KiB: ~91–104 ns, ~122–134 cycles, GPU-clocked.
* Rising through 128–512 KiB to a flat ~349 ns plateau for 1–32 MiB, then ~388 ns
  at 256 MiB.
* The first-level boundary lies between 8 and 12 KiB. The page count adds at most
  ~32 ns up to 16,384 pages; capacity dominates.

* A single SIMD group keeps ≥ 16 independent loads in flight; 16 cost one latency.
* Grid-stride streaming reads 328–356 GB/s from 8 SIMD groups per core upward
  (~89% of the advertised 400 GB/s), with per-lane exact oracles.

Not established: threadgroup memory, stores, multi-group behaviour,
exact cache sizes beyond the first level, and what serves the 1–32 MiB plateau.

## Reproducer and evidence

`make agx-loop-test` (includes hardware chases), `make verify-memory` (11 offline
checks over 21.7 M exact dependent loads and 19.3 GB streamed), `make agx-bandwidth`, `make agx-memory` (re-measurement).
[Report](../../experiments/memory/load-latency.md) ·
[evidence](../../results/2026-10-10-agx-memory-latency/README.md).

## Relationship to existing knowledge

The load encoding and element-scaled indexing follow applegpu's `device_load`
model. Its uniform-base exec path has a call-signature bug, worked around in
`agx_loop.emulate`. No prior Asahi measurement of these latencies was consulted.
The 16 KiB page size is the kernel's.

## Compiler implications and limitations

A first-level hit is ~25× a dependent ALU op. Schedule loads early, batch
independent loads before one `wait`, and keep hot per-core data ≤ 8 KiB. Misses
(~500 cycles at top clock) need high occupancy to hide. All lanes here use one
address, so divergent or coalescing behaviour is untested.
