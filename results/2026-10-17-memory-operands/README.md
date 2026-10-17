# Cached memory operands and stricter offline replay

Offline x86_64 Linux, Rust 1.99.1; no M2 device was available. Timing calibration
and hardware evidence were not changed.

## Execution change

Device and threadgroup loads/stores now extract immutable address, format,
component-mask, register and scoreboard fields once per visited PC. The lazy
shared decode owns these operands. Register values, lane addresses, masks and
memory are still evaluated during execution. Operand payloads are boxed only
for memory instructions, keeping the common instruction record compact.

Lane/component ordering, normalized conversions, scoreboard deferral and
out-of-range threadgroup behavior retain their previous implementations.

## Measured scope

`functional_memory` runs 2048 SIMD groups, each with 256 device loads and one
store. Every timed dispatch checks 65,536 output words, 532,480 instructions and
2048 groups. Input values vary by global index. Timing includes dispatch setup,
decoding and execution; mapping buffers and verifying outputs are outside it.
Each invocation discards a warm-up, then measures nine runs. Five baseline/current
pairs alternate their order. Both builds use the same example, Rust toolchain
and release options (`-C linker-features=-lld` for this host's linker setup).
Baseline source is main at `de526ce6a96c8306e9bc32e894cf68b362b7fc1b`.

| Workload | Baseline median ms | New median ms | Ratio, baseline/new |
| --- | ---: | ---: | ---: |
| Checked memory execution | 323.578 | 297.460 | 1.088 |
| ALU trace, 1 worker, compact | 4.661 | 4.884 | 0.954 |
| ALU trace, 4 workers, 64 KiB padding | 1.782 | 1.591 | 1.120 |

These are medians of the five invocation medians. The memory workload is 8.8%
faster. The control workloads vary in both directions; the compact ALU control
is 4.8% slower in this run. This is a focused improvement, not a general speedup
claim. Raw paired results are in `benchmark.json` and `trace-control.json`.

Reproduce the new benchmark from `rust/`:

```sh
cargo build --release --example functional_memory --example functional_trace
target/release/examples/functional_memory 2048 256 9
AGX_THREADS=1 target/release/examples/functional_trace 128 128 0 15
AGX_THREADS=4 target/release/examples/functional_trace 128 128 64 15
```

Copy the new benchmark source into a separate checkout of the baseline to build
the comparison binary. Alternate invocation order to reduce host drift.

## Accuracy and replay gate

All 126 selected retained recordings compare exactly: 50 device loads, 42 device
stores, 10 threadgroup cases and the previous 24 cases across eight other
categories. `replay-manifest.tsv` identifies every compressed file by Git blob
hash, relative to the simulator repository; evidence remains at the baseline
commit. This is a selected subset, not the full testsuite.

All 74 playground atomic-variant replays pass. The Rust suite has 19 passing tests
(15 library, the padding regression, two kernel equivalence tests and the new
replay CLI regression). Documentation tests complete successfully with no cases.
The retained cost-model verifier reports zero failures.

`suite_check` previously printed failures but exited successfully. It now returns
failure for mismatches, unsupported cases and empty selections. Deterministic
expected buffers must also have the complete simulated length; zip-based
comparison previously allowed a shortened expected buffer to pass. A synthetic
stop-only CLI fixture verifies success, mismatches, missing selections,
unsupported code, and truncated output/table expectations. It is not hardware
evidence. Existing approximate-op ULP allowances remain unchanged.

The playground's `make replay-memory-suite` runs the three memory categories
through this gate. For a subset checkout, override `MEMORY_SUITE=/path/to/suite`.
