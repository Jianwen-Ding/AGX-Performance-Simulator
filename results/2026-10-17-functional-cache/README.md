# Functional tracing: shared lazy decoding and bounded execution

Offline validation on x86_64 Linux, Rust 1.99.1. No new GPU measurements.

## Changes

The functional decoder used a full `Option<Info>` at every byte offset, even
though most bytes are not instruction starts. Each traced workgroup allocated
and populated its own table. The replacement stores pointers in once-cells and
allocates an instruction record only when execution visits its PC. Independent
tracing workers share these immutable records. `PreparedDispatch` also lets
`agx_dispatch` reuse decoding across its individual workgroup callbacks.

Registers, execution masks, counters, shared memory and copy-on-write device
memory snapshots remain separate for each workgroup. Decoding stays lazy;
unreachable malformed bytes are not validated or executed.

The per-group instruction limit now accumulates across barrier resumes.
Previously every barrier reset the counter, so a loop with a barrier could run
indefinitely despite `--max-steps`. Truncated instructions and taken jumps
outside the program or to odd byte addresses now return errors. Finally,
`agx_dispatch` strips zero padding in halfwords: its byte-wise stripping removed
the high byte of the final stop, a truncation that the old decoder hid.

## Benchmarks

`rust/examples/functional_trace.rs` checks every group's complete PC sequence,
instruction count, memory-access list, skipped-op set and warnings. It traces
128 workgroups of two SIMD groups, each running get_sr, 128 iadds and stop.
Five runs are measured after warm-up; all checksums are 17,040,384 over 33,280
dynamic instructions. The baseline uses the previous full-record cache; the
execution-bound fixes are present in both tracing benchmark builds.

| Configuration | Before, median ms | After, median ms | Speedup |
| --- | ---: | ---: | ---: |
| 1 worker, compact program | 14.231 | 4.654 | 3.06x |
| 1 worker, 64 KiB unreachable suffix | 48.751 | 5.312 | 9.18x |
| 4 workers, 64 KiB unreachable suffix | 13.579 | 1.624 | 8.36x |

For the four-worker padded case, child-process peak RSS (`getrusage`) fell from
94,780 KiB to 8,448 KiB. Results depend on the workload and host; the padding
case isolates the cost of allocating a cache for unvisited program bytes.

Reproduce the new tracing runs with:

```sh
cargo build --release --example functional_trace
AGX_THREADS=1 target/release/examples/functional_trace 128 128 0 5
AGX_THREADS=1 target/release/examples/functional_trace 128 128 64 5
AGX_THREADS=4 target/release/examples/functional_trace 128 128 64 5
```

The existing checked special-register CLI benchmark remained within about 3%
between builds in longer, reversed-order runs; this change primarily speeds up
tracing and reduces cache memory. A simple full functional-dispatch prediction
test measured 10.959 ms before and 10.481 ms after, including parameter loading;
all twelve prediction JSON reports were exactly equal (predicted 5.9 us).
These focused measurements are not a whole-corpus average speedup.

## Accuracy checks

- 15 Rust library tests, two kernel equivalence integration tests and the new
  `agx_dispatch` padding regression pass (18 total).
- A prepared-tracing regression runs four workers over geometry- and data-
  dependent loops, compares every trace against fresh decodes, changes the input
  memory between calls, and verifies that atomic writes never escape snapshots.
- All 74 playground atomic-variant recordings replay with zero failures.
- 30 retained hardware recordings (three each from ALU, masked ALU, atomics,
  control flow, geometry, kernels, loads, random programs, threadgroup atomics
  and shared memory) pass exactly, with no ULP tolerance needed. This is a
  representative subset, not the complete testsuite.
- Cost-model parameters and retained evidence were not changed.
