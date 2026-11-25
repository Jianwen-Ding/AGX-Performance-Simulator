# Cold jump-target fetch by jump distance (2026-11-25)

Hardware finding: playground `novel-results/2026-11-25-jump-fetch-distance`. Probe: `scripts/probe_jump_distance.py`.
`hardware.jsonl` holds 80 code shapes, measured on one SIMD group with agx_run_hw back to back. The
`probe_jump_fetch.py` reruns are tagged `script`.

## Hardware

- **Cost depends on the jump, not on code size.** ns per one-line block is the same at 16 KiB, 512 KiB and 8 MiB of code.
- **A first-touch target costs more the farther the jump.** In lines: d = 1 → 324, 2 → 327, 3 → 342, 4 → 368, 5 → 376,
  7 → 399, ≥ 8 → 400-430 (this includes ~30 ns of execution). Direction makes no difference, and random gaps cost the same
  as constant ones.
- **A forward jump of 4 lines or fewer keeps the sequential stream.** The lines it passes over are fetched too: a later
  jump to one of them costs ~130-190. A jump of 8 lines or more fetches nothing in between.
- **Random order is cheap (235) only through reuse of 512 B blocks.** When other lines of the target's block were fetched
  before, the jump is cheap. Random order over every 4th line costs 405, and a block with only one line fetched gives no
  gain. This explains the 2026-11-07 puzzle "a jump to the next line (324) is slower than a random one (235)".
- **One run faulted.** Kind `win32_8_1` hit a GPU fault (fence -EFAULT). The GPU recovered, and the kind was not retried
  (`coverage/faults.json`).

## Model: `cost::JumpFetch` (default on, `AGX_JUMP_FETCH=0` restores the flat cost)

It is used for first-touch lines entered by a taken jump in two places:

- the i-cache cold path (`cost.rs`, in place of the flat `fetch_cold_jump_ns`);
- the first-wave fetch sum (`dispatch.rs`, in place of `fetch_jump_line_ns` × jumped lines), when the profile charges
  those lines separately (runtime profile).

The costs:

- distance table 294 / 296 / 312 / 334 / 346 / 344 / 369 ns for 1-7 lines, 380 ns beyond;
- lines a ≤ 5-line forward jump passed over: 140 ns;
- a target in a 512 B block with ≥ 2 lines already fetched: 150 ns.

## Accuracy

The `verify-jump-fetch` row is 79 probe shapes ([check-output.txt](check-output.txt),
[flat 300 ns](check-output-flat300.txt)). The other rows are reference sets (capcheck, runtime profile); "before" is the
2026-11-22 defaults (results/2026-11-22-merge-session).

| Set | Before | After |
|---|---|---|
| `make verify-jump-fetch` (79 probe shapes) | mean 18.2%, median 19.6%, max 53% | **6.0%, median 3.2%**, max 33% |
| llama tg (10 launches) | mean 9.4%, median 9% | **8.2%, median 8%** (Q8_0 mat-vec -19 → -16%, Q5_0 -7 → -3%, rope -13 → -11%) |
| llama fa (4) | 7.8% | **7.5%** |
| llama pp512 (20) | 10.2%, w10 15 | 10.2%, w10 15 |
| held-out (159) | 7.5%, median 5, w10 118 | 7.5%, median 5, w10 118 |
| tinygrad (5) | 13.4% (same binary, rule off) | 14.6% (E_8_8_8_16_4 +9 → +13%; others ±1) |

The tg "before" run used the same binary with the rule off: `/scratch/agx-r8/tg-base.txt`.

`verify-cost-model` is identical. `verify-dispatch` is unchanged (36 dispatches, median 4.8%), and `cargo test` passes
(new `jump_fetch_tests`). The baseline pass of the full evaluation was stopped by host memory pressure, so the "before"
column for pp, fa, held-out and tinygrad comes from the stored 2026-11-22 scores (pp, fa, held-out) and a separate
tinygrad rerun.

## Not fitted

| Probe shape | Hardware | Rule | Error |
|---|---|---|---|
| interleaved chains of stride 16-64 lines | 343-348 | 238 | -31% |
| backward jump to the line before, after the lines above were fetched | 323 | 216 | -33% |
| 4 KiB / 8 KiB strides | 364-367 | 410 | +12% |

The variant "block reuse needs a fetched line below the target" fixes the backward case but worsens the chains: probe mean
5.3%, fwd16 -43%. It was not adopted.
