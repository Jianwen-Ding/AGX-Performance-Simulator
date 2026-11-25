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
- a target in a 512 B block with ≥ 2 lines already fetched: 150 ns, except after a backward jump of 1-7 lines
  (hardware: as cold as a forward jump of that length, 322-323 ns for 1-2 lines).

## Accuracy

The `verify-jump-fetch` row is 79 probe shapes ([check-output.txt](check-output.txt),
[flat 300 ns](check-output-flat300.txt)). The other rows are reference sets (capcheck, runtime profile); "before" is the
2026-11-22 defaults (results/2026-11-22-merge-session).

| Set | Before | After |
|---|---|---|
| `make verify-jump-fetch` (79 probe shapes) | mean 18.2%, median 19.6%, max 53% | **4.9%, median 2.5%**, max 32% (6.0% before the backward-jump exception) |
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
| 4 KiB / 8 KiB strides | 364-367 | 410 | +12% |

The variant "block reuse needs a fetched line below the target" fixed the backward case but worsened the chains (probe mean
5.3%, fwd16 -43%). The adopted exception (no block reuse after a backward jump of 1-7 lines) fixes back / bskip1 /
bskip2 (-33 / -31 / -20% → 0 / +3 / +3%) and leaves every other shape unchanged. The tg, tinygrad, fa and held-out (160) sets are identical, launch for launch.

The chain and stride misfits all involve 1 KiB or 4 KiB strides (fwd16 / fwd64 interleaved chains, skip64 / skip128 /
bskip64, two64_*: 4 KiB strides cost ~350-375 against ~405-428 for other long jumps). This looks like address
aliasing in the memory below the fetch unit rather than a fetch rule. Real kernels rarely jump this way, so it is not
modelled.

## Follow-up: streamed cold line 64 ns (runtime profile, `Launch::fetch_cold_stream_ns`)

One group's straight-line code beyond the i-cache measures 70.4 ns per line on hardware, unchanged from the
2026-11-07 runs. The simulation charged it 76.3, because it adds ~6 ns of its own per line on top of the 70 ns fetch
cost. With 64 it gives 70.3.

This applies to the i-cache refetch path only. The first-wave fetch charge for code that fits the cache keeps its
calibrated 70.

| Set | Jump rule only | + cold stream 64 |
|---|---|---|
| tinygrad (5) | 14.6% | **13.0%** (E_8_8_8_16_4 +13 → +7%) |
| llama fa (4) | 7.5% | **6.8%** |
| llama tg (10) | 8.2% | 8.2% |
| llama pp512 (20) | 10.2%, w10 15 | 10.3%, w10 15 (Q8_0 MMQ -6 → -8%) |
| held-out (160) | 7.5%, w10 118 (159) | 7.5%, w10 119 (heartwall ran this time); myocyte 0 → -7%, likelihood +8 → +5% |

The 2026-11-22 fetch-overlap probe (code beyond the cache, 1-8 groups) moves from +9..16% to +2..11%. The remaining
growth with group count is ALU contention: the model's groups stay in lockstep after their simultaneous loads.
`AGX_PICK_GREEDY=1` (experimental, off) targets it. With it the probe's 8-group cases come to +4% / +2% and dispatch
improves (4.8 → 4.0%), but core scaling worsens (grid 3.0 → 3.5%, chains 3.3 → 4.1%).

## Experiment: greedy scheduler pick (`AGX_PICK_GREEDY=1`, not adopted)

Each scheduler first offers the tick to the group that issued on it last. Scored on the reference sets
([eval-greedy](eval-greedy/)) against the defaults above:

| Set | Default | Greedy |
|---|---|---|
| held-out (160) | 7.5%, median 5.5 | **7.1%, median 4** |
| tinygrad (5) | 13.0% | **10.8%** |
| llama fa (4) | 6.8% | **6.2%** |
| llama tg (10) | 8.2% | 8.3% |
| llama pp512 (20) | 10.3% | 10.8% (rms_norm -27 → -31%, quantize_mmq -12 → -15%) |
| `verify-dispatch` (36) | 4.8% | **4.0%** |
| core scaling grid / chains | 3.1 / 3.3% | 3.5 / 4.1% (within the 5% limit) |
| `verify-memory-model` stream-dram / stream | 5.9 / 8.8% | **17.7 / 13.3%: FAIL** |

Across launches 59 improve and 41 get worse. Greedy is not the default because it breaks the measured DRAM streaming
sets, which are direct hardware evidence. The real-program gains say that group desynchronisation after shared waits
matters. A rule that de-synchronises groups without making streaming loads greedy is the open problem.
`AGX_PICK_FIXED=1` (static priority) was rejected earlier: grid median 3.9%, p90 39%.

## Experiment: greedy only for ALU work (`AGX_PICK_GREEDY=2`, not adopted)

The scheduler keeps offering the tick to its last group only while that group's next instruction is ALU work, so
streaming loads stay fair. Scored the same way ([eval-greedy2](eval-greedy2/)):

| Set | Default | Greedy=2 |
|---|---|---|
| fetch-overlap probe G=8, F=48 / 120 (hw 51.8 / 98.3 us) | 57.5 / 108.3 (+11 / +10%) | **54.1 / 100.0 (+5 / +2%)** |
| `verify-memory-model` | pass | pass (stream sets unchanged) |
| held-out (160) | 7.5%, median 5.5, w10 119 | **7.1%, median 4.5, w10 120** |
| tinygrad (5) | 13.0% | **11.0%** |
| llama fa (4) | 6.8% | **6.5%** |
| llama tg (10) | 8.2% | 8.3% |
| llama pp512 (20) | 10.3%, w10 15 | 10.8-11.1%, w10 14 |
| `verify-dispatch` (36) | 4.8% | 5.1% |
| core scaling grid / chains / idle / dests (median) | 3.1 / 3.3 / 7.1 / 5.8% | 3.5 / 3.8 / 8.6 / 6.0% |

Mode 2 fixes what broke mode 1, but all four core-scaling microbenchmark sets and dispatch get worse. Those are the
calibration hardware measurements, so it stays opt-in. The held-out gain (0.4 points) is the size of the remaining
group-desynchronisation effect. A rule that wins on both would have to come from a measurement of how several groups
share one scheduler after a common wait. That measurement does not exist yet (single-group probes only).
