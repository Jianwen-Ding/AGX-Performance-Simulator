# AGX coverage suite

A test catalog for everything known about the G14 (M2) GPU. It has no expected values yet: an agent on Asahi
Linux records them on the GPU (`tasks/agx-coverage.md`), and the functional simulator and cost model are then
checked against the recordings offline.

| Path | Contents |
|---|---|
| `scripts/coverage_catalog.py` | The catalog. Each case is generated from tables, so adding a form is one line. |
| `scripts/coverage.py` | Runner with `list`, `validate`, `record`, `check` and `status` subcommands. |
| `coverage/expected/<area>/<name>.json.gz` | Recorded functional cases, in the same format as `testsuite/`. Written by `record`. |
| `coverage/timing/<name>.json` | Recorded instruction-bench results (latency, SIMD issue, per-core throughput). Written by `record`. |
| `coverage/log.jsonl`, `coverage/faults.json` | Log of every recording outcome, and the cases that faulted. |

## Catalog (4,247 cases)

| Area | Safe | May fault | What it covers |
|---|---|---|---|
| `alu` | 863 | 292 | Integer, float and bit ALU forms. Varies operands (16/32/64-bit, `.sx`, `.abs`/`.neg`, `.cache`/`.discard`, uniforms, immediates), all 16 bitop truth tables, shifts, saturation, every funop function code (0–63), every convert mode×round field, every compare/ballot condition code including the unnamed ones, shuffles with every index kind, reductions and prefix ops (16/64-bit), and simd_op sub-ops. |
| `alu_masked` | 213 | – | Cross-lane forms running with a quarter of the lanes inactive. |
| `mem_load` / `mem_store` | 202 / 194 | 116 / 112 | All 12 device formats × component masks × 16/32-bit registers, `lsl 0..3` shifts, signed/unsigned index, uniform/register base, and immediate index. |
| `geometry` | 760 | – | Every special register 0–255. The identified ones run under 13 launch geometries (1-D, 2-D, 3-D, partial groups, workgroup sizes 1–1024), the rest under one. All 256 are safe to read (no fault in 126+ full scans); the former 214 `may_fault` cases were recorded on 2026-10-16 (tasks/safe-06). |
| `control` | 59 | – | if/else/while for every int and float condition, nesting depth 1–8, and `push_exec` counts 1–3. |
| `threadgroup` | 44 | 60 | Barrier 0x68 vs 0x78 at workgroup sizes 32–1024. Threadgroup store/load in 9 formats × masks × immediate/register index. |
| `sync` | 5 | 251 | Every `memory_barrier a, b, c`. |
| `atomic` | 11 | 11 | All 11 device atomics, with uniform base (safe) and register base (may fault). |
| `undecoded` | 403 | – | The 49 non-faulting undecoded opcodes (playground opcode-descriptions, sync-and-end-opcodes) in the 8 operand encodings that ran there: 31 six-byte no-ops, 0x18 (bit 7 clear), and the 16 ends opcodes plus 0x98 / 0xf8 (also inside a 3-of-4-lane branch). Recorded 2026-10-16 (tasks/safe-08): 403/403 deterministic, simulator exact. |
| `bench` | 651 | – | Timing cases: every safe ALU form whose destination and first source have the same width. Each becomes a dependent-chain instruction-bench form. |

**Risk levels:**

- `safe` forms have already run on the hardware, or are plain variations of forms that have (other registers, operands or sizes).
- `may_fault` forms are encodings that have never run, or that sit next to encodings that hung the GPU:
  - funop functions 2, 3, 5, 7, 11 and 15;
  - convert with round-field bit 27 set;
  - the "never" condition codes;
  - unknown sub-ops and special registers;
  - unproven memory formats.

  `record` runs these only when given `--risky`.

Before any functional case is submitted, `record` runs the exact program in `agx_run` with the same uniforms and memory
layout (`coverage.simulate`): it must finish and, when the case has a host oracle, match it. A refusal is logged as
`sim-refused` and nothing is submitted. `--no-sim` skips this step.

**Not generated** (`coverage_catalog.MANUAL`): behaviours that need hand-built encodings or a Mesa capture:

- the fmadd lane-offset form;
- the stack instructions;
- the preshader;
- imageblock special registers;
- cross-core memory ordering;
- occupancy limits;
- the in-shader clock.

The task file lists them.

## Encoding

Every case assembles with the Rust assembler, `rust/target/release/agx_asm`:

- It is bit-identical to applegpu. Rebuilding the 669 stored test-suite programs through it reproduces their machine code byte for byte.
- So recording does not need applegpu.
- For forms the assembler has no syntax for, the case text is written `@field=value,...<TAB>TEXT`. The assembler assembles `TEXT`, then sets the named decode fields, or the raw bit ranges `bSTART_SIZE`. Examples:
  - `@cc=3<TAB>icmpsel ult, ...` sets a condition code that has no name;
  - `@b28_6=5<TAB>floor r14, r6` gives funop function 5.

`coverage.py validate` builds every case offline and checks that it assembles and decodes. It applies the same gate the recorder uses: no undecodable or control-transferring instruction. Run it after editing the catalog.
