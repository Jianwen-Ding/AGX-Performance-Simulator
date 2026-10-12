# Hardware-recorded AGX test suite (721 cases)

Exact behaviour of the M2 Max GPU (driver-reported G14C B1, Asahi Linux, direct DRM) captured case by case, so the
functional simulator can be developed and checked without the device. Recorded by `scripts/record_suite.py`
(hardware); checked offline by `rust/target/release/suite_check` (`make suite-check`).

| Category | Cases | What each case is |
| --- | --- | --- |
| `alu` | 253 | one instruction form (`scripts/suite_templates.py`) over 1,024 lanes; per-lane operands r6..r13 from the table (float specials, fp16 pairs, edge integers, random); r14/r15 preset to 0xdeadbeef, written back to the table |
| `alu_masked` | 89 | the cross-lane forms (shuffles, quad/SIMD reductions and prefix ops, ballots) with lanes where `gid & 3 == 3` inactive |
| `mem_load` | 50 | device loads: formats i32/i16/i8/u8norm, masks x..xyzw, unsigned/signed index, `lsl` 1/2, uniform or register-pair base |
| `mem_store` | 42 | device stores with the same field combinations (encodings derived from the control store) |
| `threadgroup` | 10 | threadgroup store, barrier, neighbour load; i32/i16/i8, masks, immediate or register index |
| `atomic` | 44 | device atomics: 11 ops × distinct / 4 shared words × return / no return |
| `tg_atomic` | 44 | threadgroup atomics, same matrix |
| `control` | 23 | lane-dependent if/else (integer and float conditions), nesting, per-lane loop trip counts, nested loops, `while_fcmp` |
| `geometry` | 8 | 1-D/2-D/3-D launches (8 shapes incl. non-power-of-two and partial groups): 23 geometry special registers per thread |
| `random` | 150 | random programs of 24-40 template instructions chained through r16..r23 (100 straight-line, 50 with lane-dependent if/else and a per-lane loop); r16..r23 stored per lane |
| `kernel` | 8 | tree reductions (3 sizes), saxpy (fmadd32), 64×64 integer matmul, per-workgroup scan, histogram, 16×16 transpose; each matched a host oracle on hardware before it was recorded |

## Case format (`<category>/<name>.json.gz`)

```text
code       hex: the main shader exactly as submitted (harness control prefix: r1 = global id + u4, exec mask for
           r1 < u5; the case body; out[8 + r1] = r2; stop)
launch     grid [x,1,1], wg [x,1,1], regs (16-bit halves), shared_bytes (threadgroup memory)
layout     table_va, out_va, out_words; uniforms u4 = 0, u5 = grid x, u6_u7 = out_va, u8_u9 = table_va (32-bit uniform
           registers, low word first); out buffer before the dispatch: words 0 .. out_words-17 = 0xcdcdcdcd, the last 16 =
           0xa5a50000 + i
input      table: base64 little-endian u32 words, the table before the dispatch
expect     out, table: base64 u32 words after ONE dispatch
deterministic, deterministic_out, deterministic_table: two hardware runs agreed (compare only the deterministic parts;
           contended atomics return order-dependent values, and xchg/cmpxchg on shared words leave an order-dependent
           final word)
fw_ns      firmware durations of the two runs (ns)
```

A simulator reproduces a case by placing the two buffers at those VAs, setting the uniforms, running every SIMD
group of every workgroup (barriers within a workgroup, threadgroup memory per workgroup) and comparing the buffers
word for word.

Safety of re-recording: only applegpu-assemblable named forms are used; the funop functions and convert encodings
that hung the GPU are excluded (`suite_templates.py`), every body passes a decode gate, memory cases are bounds
checked, and control cases are emulated first.

## Status and use

`make suite-check` (≈1 s) prints per-category pass counts and the instructions the functional simulator cannot yet
run (`-v` lists every failure with its first differing word; `--only CATEGORY`, `--case NAME`). Baseline when the
suite was recorded: 330 / 721 (the applegpu-derived emulator lacks convert, the funop functions, quad and SIMD
reductions / prefix ops, quad shuffles and ballots, intl, some float compare codes and u8norm loads).

Timing is tested separately against stored hardware measurements: `make verify-cost-model`, `verify-scaling`,
`verify-memory-model`, `verify-dispatch` (each prints median errors and fails past a limit).
