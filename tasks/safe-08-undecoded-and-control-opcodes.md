# Task safe-08: the 49 non-faulting undecoded opcodes, 0x18, and the ending opcodes

Read [safe-probes.md](safe-probes.md) first (safety rules).

**Goal:**
- Add every non-faulting undecoded opcode to the coverage suite and the simulator.
- Narrow down what 0x18 synchronises.
- Look for any difference among the 18 ending opcodes.

These matter for the PTX front end because the backend must never emit them by accident, and because a cheap "end this SIMD
group" or a sync primitive may be useful.

**Background** (playground `novel-results/2026-10-11-opcode-descriptions` and `2026-10-11-sync-and-end-opcodes`,
direct DRM, no fault in any run):
- **31 opcodes with low bits `0b11`** (0x03, 0x07, ... 0x7f): 6-byte true no-ops that issue at full rate.
- **0x18:** a 2-byte fixed-cost op (4.3 ns) that waits for nothing tested.
- **0x78:** the 4-SIMD-group barrier, already modelled.
- **16 "ends" opcodes** (0x01, 0x0d, 0x1d, 0x2c, 0x2d, 0x3c, 0x3d, 0x4d, 0x50, 0x5d, 0x60, 0x6c, 0x6d, 0x70, 0x7c,
  0x7d), plus 0x98 and 0xf8: end the whole SIMD group whatever the exec mask.
- The scan tooling is `scripts/run_drm_opcode_describe.py` (both repositories) and the playground's
  `run_drm_sync_probe.py`.

## Safety

- Only the 8 operand encodings the finding used: `base`, `zero`, `d6`, `dt0`, `dt1`, `dt3`, `b15` and `hi`.
- **Never run** 0x06, 0x24, 0x34, 0x44, 0x46, 0x54, 0x5c, 0x64, 0x74, or 0x6e (data leakage). `run_drm_opcode_describe.py`
  refuses them; keep that check.
- **Never run** the bit-7 forms the finding lists as hanging or unrun: 0xb8 (`wait` + bit 7), 0xc8/0xd8 (the bit-7
  forms of `wait_pix` / `signal_pix`), 0xa8 (`doorbell` + bit 7).
- An "ends" opcode placed early leaves the rest of the code unrun. That is fine: the program still ends.
  - Keep the `build_main()` suffix anyway.
  - Check in `agx_run` that the simulator agrees, once it models the opcode.

## Steps

1. **Coverage cases.**
   - Add one case per opcode and operand encoding to `scripts/coverage_catalog.py` (`MANUAL` item `undecoded_opcodes`).
     That makes 49 × 8, and 0x18 / 0x78 only with bit 7 clear.
   - The expected effect is the finding's: no register change, or "nothing after it runs".
   - Record them and check them with `suite_check`.
2. **Simulator.**
   - Teach the decoder the lengths: 6 bytes for the `0b11` group, 2 bytes for 0x18, 0x98 and 0xf8.
   - Treat the 16 ends opcodes, 0x98 and 0xf8 as "end the SIMD group" in `func.rs`.
   - Give 0x18 its measured 4.3 ns (about 6 cycles) in `cost.rs`.
3. **What 0x18 waits for**, using only forms that already ran:
   - The finding tested own loads, stores, threadgroup memory, atomics and cross-group sync.
   - Untested, still safe on direct DRM:
     - after a device **atomic with return** (`agx_loop.atomic_add` form);
     - after a **uniform_store** in the preamble (`AGX_PRESHADER=1`, see [safe-09](safe-09-preshader.md));
     - after an **f533 / f59d** barrier;
     - with 2-32 SIMD groups of one workgroup contending (repeat the 17.9 ns case with `wait` as a control, at
       least 8 runs, fast-clock cluster only).
   - Report whether the cost changes in any case.
4. **Ending opcodes:** look for a difference that the earlier tests could not see.
   - The SIMD-group slot reuse rate: dispatch 2^20 one-group workgroups ending with each opcode against `stop`.
   - **Do not** combine an ending opcode with a barrier in another SIMD group. A group that ends early may leave the
     others waiting at the barrier forever. List it as a follow-up instead.
   - Report "indistinguishable" again if nothing differs.

## Deliverables

- About 400 recorded coverage cases.
- Decoder and simulator support, with `make coverage-check` passing.
- An update to the playground findings.
- A `make verify-undecoded-opcodes` target. Push both repositories.
