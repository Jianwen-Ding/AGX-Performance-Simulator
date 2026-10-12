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
run (`-v` lists every failure with its first differing word; `--only CATEGORY`, `--case NAME`).

Current: **691 / 721**. Every category passes completely except `alu` (245 / 253) and `random` (128 / 150), and
every remaining failure is an fp32 transcendental (`rcp`, `rsqrt`, `rsqrt_special`, `log2`, `exp2`, `sin_pt_2`
on 32-bit operands; the 22 failing random programs all contain one). The hardware computes those with a ~26-bit
approximation: 10-15% of lanes differ by 1-3 ulp from the correctly rounded value the simulator uses, and the
suite's samples cannot pin the approximation down. `make record-transcendentals` (hardware, `scripts/
record_transcendentals.py`) sweeps every mantissa: for rcp / rsqrt / rsqrt_special it writes exact tables to
`transcendental/`, which the simulator applies automatically (validated here with tables built from the suite's own
lanes: those alu cases then pass); for log2 / exp2 / sin_pt_2 it records dense samples for fitting their algorithm.

Semantics fitted from this suite (beyond the applegpu emulator), all exact on every recorded lane:

| Instructions | Behaviour |
| --- | --- |
| floor, ceil, trunc, rint | fp32 (denormal in = 0, tiny out flushed) or fp16; canonical NaN; sign of zero kept |
| intl | interleave the low 16 bits of A (even result bits) and B (odd bits) |
| convert | applegpu's `round` field is the direction (0 float→int, 1 int→float); mode = size<<2 \| rte<<1 \| signed. Float→int saturates, NaN → 0, zero-extended from the type width; int→float rounds to the destination precision (fp16 for 16-bit destinations) |
| simd_/quad_ reductions | pairwise tree; float add/mul flush denormal inputs; min/max return raw operand bits (denormals compare as zero, left operand on ties, NaN loses, of two NaNs the left; inactive lanes take part as canonical NaN) |
| prefix ops (exclusive) | quad: lane-order chain; SIMD: per-quad chain for lanes 0-2 and tree for the quad total, then Kogge-Stone over quads |
| quad_shuffle family | per-lane index; out-of-quad keeps the own value; rotate_up wraps |
| u8norm loads | v / 255 rounded to the destination precision |
| sin_pt_1 | folds onto [-1, 1]: x, 2-\|x\|, \|x\|-4 with the sign of x; zero → +0; \|x\| > 4 → NaN |
| sin_pt_2 | sin(πy/2)/y; 0 → RN(π/2); \|y\| > 1 → NaN |
| fp16 transcendentals | correctly rounded; fp16 denormal inputs are real values |
| special values | rcp(±0) = ±inf; rsqrt(±0) = ±inf, rsqrt(negative or -inf) = NaN; rsqrt_special(±0, +inf) = 1; log2(0) = -inf; exp2(0) = 1, exp2(-inf) = 0 |

Timing is tested separately against stored hardware measurements: `make verify-cost-model`, `verify-scaling`,
`verify-memory-model`, `verify-dispatch` (each prints median errors and fails past a limit).
