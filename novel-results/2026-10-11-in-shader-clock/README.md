# An in-shader clock: a CPU-written timer read through coherent memory

Status: confirmed on one device (driver-reported M2 Max G14C B1), direct DRM, compute stage.
Answers the open request "find any behavior related to microbenchmarking and time". No
special register runs as a clock (the earlier SR scans), but the shader can read time anyway.
Only decoded, Mesa-emitted instructions are used.

## Claim and scope

A CPU thread keeps writing the low 32 bits of `CNTVCT_EL0` (ARM generic timer, 24 MHz =
41.7 ns per tick) plus a spin counter into a GPU buffer while the dispatch runs. The
shader reads the timer at two points:

| Read method (second read of the clock) | Fresh value? |
| --- | --- |
| Same cache line, normal `device_load` (Mesa u4 = 4) | **no**: returns the first read's value, even 3 ms later |
| Same line, coherent `device_load` (u4 = 7, bits 44-46; Mesa's coherent/volatile form) | **no** |
| Same line, normal load after any Mesa barrier | **no** |
| Same line, coherent load after `memory_barrier` (`f596`) or `device_barrier_2` (`f59e`) | **no** |
| **Same line, coherent load after `device_barrier_1` (`f59d`) or `unknown_barrier_1` (`f533`)** | **yes** |
| **A cache line this dispatch has not read yet** (normal or coherent load) | **yes** |

**CPU→GPU coherence works without CPU cache maintenance.** The first read of a line always
returns a CPU write from that dispatch's own time window, with or without `dc cvac`. The GPU
then keeps the line cached, and only `device_barrier_1` or `unknown_barrier_1` followed by a
coherent load re-fetches it. This also gives meaning to two of Mesa's unnamed barriers: both
**invalidate the GPU-side cache for coherent loads**. Mesa emits them in its device-scope
"big hammer" sequence (`agx_compile.c:1520-1534`).

**Accuracy.** Over a counted loop of 0 to 100,000 trips (0.4 µs to 5.2 ms), the shader-measured
interval equals the firmware dispatch duration minus a constant **3.5–7.2 µs** (dispatch
start/end overhead outside the shader) in all 30 runs, across clock states. Back-to-back reads
are **0.4–0.75 µs** apart, which is the cost of one read (barrier + coherent load round trip).
Resolution is one timer tick (41.7 ns). All 32 lanes read the same value.

Usage, in the order emitted:

```
f59d                         device_barrier_1    (invalidate)
device_load (u4=7) ..., r5   coherent load of the CPU-written timer word
wait 0
```

Not established: cost under load (many lanes or workgroups reading at once), whether the GPU
L2/SLC adds staleness across cores, and the exact semantics of the two invalidating
barriers on other memory types.

## Reproducer and evidence

* `tests/drm_compute.c`: `AGX_CPU_CLOCK=<table word>` (CPU timer thread),
  `AGX_CPU_CLOCK_FLUSH=1` (`dc cvac` after each write), `AGX_CPU_CLOCK_LINES=N` (copies 128 B apart).
* `scripts/run_drm_cpu_clock.py` (hardware), `make test-in-shader-clock`
  ([test](../../tests/test_in_shader_clock.py)), `make verify-in-shader-clock` (12 offline checks).
* [Evidence](../../results/2026-10-11-in-shader-clock/README.md).

## Relationship to existing knowledge

Earlier SR scans found no progressing clock among the special registers
([failed approaches](../../notes/failed-approaches.md)). Mesa marks `device_barrier_1` and
`unknown_barrier_1` as "not clear what these are" (`agx_opcodes.py:424-429`). The firmware
timestamps (`drm_asahi_cmd_compute.ts`) give only whole-dispatch start/end.

## Compiler implications and limitations

* Microbenchmarks can now time a region *inside* one dispatch, per lane or per workgroup, with
  about 1 µs of overhead per read. That enables per-workgroup launch timelines, occupancy
  studies and latency measurements without whole-dispatch subtraction.
* For PTX `%globaltimer`, a runtime could provide the CPU timer thread, and the compiler would
  emit `device_barrier_1` + coherent load. It is only coarse (µs-scale cost, 41.7 ns ticks) and
  costs a CPU core while the kernel runs. `%clock`/`%clock64` (cycle counters) remain unmapped.
* `volatile`/coherent PTX loads that must see host writes during a kernel need the invalidating
  barrier, not just the coherent bit.
