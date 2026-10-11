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

## Equivalence with the Python model

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
| `src/isa/` | applegpu port: descriptor table, operands, assembler, emulator (`exec.rs` reference, `compiled.rs` fast), `fma.rs` soft float |
| `src/cost.rs` | Params, decode classification, mark_loops, trace, fit_branch, simulate |
| `src/kernels.rs` | counted / bench loops, store and threadgroup encodings, random kernels |
| `src/pyrandom.rs` | CPython `random.Random` |
| `src/bin/` | `agx_cost`, `agx_predict`, `verify_cost_model`, `agx_bench` |
