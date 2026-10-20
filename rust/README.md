# agx-sim: Rust port of the AGX cost model

The repository's cost model: a Rust port of `agx_cost.py` (from Apple-M2-Playground; the Python
version has been removed), together with the parts of
[applegpu](https://github.com/dougallj/applegpu) (BSD-3-Clause, revision `4c5bae6`) that the model
uses: instruction decoding, disassembly, the single-line assembler and the emulator that derives
the dynamic path. Python and applegpu are not needed at run time.

```sh
cargo build --release
./target/release/agx_cost code.bin [--groups G] [--annotate] [--units real|ref] [--allow-unknown] [--load-latency N]
./target/release/verify_cost_model        # offline verifier (port of scripts/verify_cost_model.py)
./target/release/agx_bench [reps]         # stage timings on the stored random kernels
./target/release/agx_predict A.bin B.bin UNITS GROUPS   # used by scripts/cost_hw.py predict()
cargo test --release
```

`agx_cost` takes the same arguments as the former Python script and prints the same JSON line and
`--annotate` table. Parameters come from `../results/` (override the repository root with
`AGX_SIM_REPO`).

By default the costed path is the cost model's emulation of one SIMD group, which reads uniforms and memory as
zero. A branch or loop on a kernel argument (`tid < n`) then takes the zero path, and a loop on loaded data follows
zeros; `warnings` names such conditions. Launch inputs make the path real:

```sh
# the group's geometry and arguments: special registers and uniforms as in a launch, memory still zero
./target/release/agx_cost code.bin --uniform 5=1024 [--wg 128] [--wg-id 3] [--simd-group 1] [--regs H] [--tg-bytes N]
# plus memory: the path comes from the functional simulator (exact control flow for data-dependent loops)
./target/release/agx_cost code.bin --uniform 0=0x100000 --map 0x100000=row_ptr.bin --alloc 0x300000=4096 [--grid X]
```

Uniforms are 32-bit registers (`u4 = 4`), as for `agx_dispatch`; `--map`, `--alloc` and `--preamble` are
`agx_run`'s. With launch inputs the JSON line gains `"paths": "geometry"` or `"functional"`; without them, the
output is unchanged.

## Equivalence with the Python model

> **Since 2026-10-11 the model deliberately differs from the Python version when more than one SIMD
> group runs per core** (per-scheduler issue, per-half register-read ports and shared units, forwarding
> of fresh sources, reduction rates from 32 groups; see `../novel-results/2026-10-11-core-scaling`).
> Every 1-group check still prints exactly what the Python model printed; the statements below
> describe the port as it was at commit `9754f83`.

The port has to give the same numbers, not just similar ones:

* `verify_cost_model` reproduces `results/2026-10-11-cost-model/verify-output.txt` byte for byte:
  every check over ~400 kernels (bench, rotation, memory, wait, control, transcendental probes and
  all random and held-out kernel sets), to the printed 3 decimals.
* Differential test of the simulator core: all 1,238 distinct simulations the verifier performs
  (12.7 M dynamic instructions) were exported (`verify_cost_model --export FILE`) and run through
  the original `agx_cost.simulate`: 0 mismatches, cycle totals bit-identical, and the branch
  penalty re-fitted in Python matches (32.88197995991983).
* The random kernel generators reproduce every stored kernel's instruction list (CPython's
  Mersenne Twister is ported), and each line passes the assembler round trip
  (`tests/kernels_match.rs`).
* The optimized emulator is checked against the straightforward reference interpreter on bench
  loops, random ALU kernels and random memory kernels (`compiled_trace_matches_reference`).

What makes it exact: Python's round-half-to-even for every tick conversion, the same float
operations in the same order (token buckets, port refill), `defaultdict` insertion semantics in
the control-crossing rule, and applegpu's quirks (which instructions raise in the emulator and
therefore count as skipped and taint their destination, operand decode edge cases).

Not verified: applegpu itself was not executed in this port's environment, so the decoder and
emulator were checked through the results above, not instruction by instruction against applegpu.
Kernels using instructions outside the verifier's workloads (textures, tile, iterators, atomics)
decode through ported but less exercised paths.

## Speed

The context-aware cost tracer caches get_sr operands, supports half-width reads
with the other half preserved, and computes SR56..59 ranks/counts from the current
execution mask (SR16 remains physical quad position). Uniform values resolve
once per executed read, never across calls or launches; unknown state retains
the existing skip/taint fallback. Odd 32-bit destination encodings do not write
an adjacent high register. Context-free tracing/reference behavior is unchanged.
`cargo build --release --example cost_special_registers` builds a checked CPU
benchmark and JSONL replay probe used by playground `replay-simulator-cost-sr`.

Functional dispatch preparation caches workgroup dimensions and group counts;
linear workgroup indices and race-log writer bases are computed once per
workgroup, not per SIMD group/barrier resume. Requested trace coordinates must
lie inside the dispatch. Empty grids remain no-ops, including huge dimensions
on the other axes. The functional executor accepts at most 2^48 workgroups
(48 index bits + 16 group bits in its existing race-log writer tag); larger
grids return an error rather than aliasing writers. This is a simulator
representation limit, not a hardware geometry restriction.

The large interpreter loop stays separate from workgroup setup. Uniform SR and
active-SIMD-count broadcasts bypass per-lane selector-class dispatch while
retaining checked lane-ordered writes. Focused CPU measurements, control
regressions and intermediate candidates are retained in
`../results/2026-10-17-functional-geometry/README.md`.

`agx_bench` (132 stored random kernels in a 500-trip counted loop, ~15.5 k dynamic instructions
each; one core of the build container):

| stage | ms per kernel |
| --- | --- |
| decode | 0.3 |
| trace (emulator) | 5.6 |
| simulate, 1 SIMD group | 2.5 |
| simulate, 4 SIMD groups | 19 |

The full offline verifier takes ~9 s. For comparison, the Python `simulate` alone took 547 s on
the verifier's 1,238 simulations, before decode and applegpu's per-lane emulation.

Exact optimizations used: the emulator compiles each static instruction once (field values,
operand locations, validated register indices) and memoizes the soft-float FMA across lanes with
identical inputs; the simulator caches per-group source readiness until the group issues again and
skips refilling token buckets that are already at their cap.

## Layout

| file | contents |
| --- | --- |
| `src/isa/` | applegpu port: descriptor table, operands, assembler, emulator (`exec.rs` reference, `compiled.rs` fast), `fma.rs` soft float; `mesa.rs` the Mesa ISA mode check (`--mesa-isa`) over the generated `mesa_table.rs` |
| `src/cost.rs` | Params, decode classification, mark_loops, trace / trace_group (per-group SRs and memory segments), fit_branch, simulate / simulate_groups, MemModel |
| `src/dispatch.rs` | whole-dispatch model (occupancy, placement, launch, waves) |
| `src/kernels.rs` | counted / bench loops, store and threadgroup encodings, random kernels |
| `src/pyrandom.rs` | CPython `random.Random` |
| `src/bin/` | `agx_cost`, `agx_predict`, `agx_dispatch`, `verify_cost_model`, `scale_check`, `mem_check`, `dispatch_check`, `agx_bench` |

