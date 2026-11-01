# Threadgroup xy loads cost what x loads do; the settle-on-read rule fixes the two largest misses and exposes a post-barrier cost

## xy threadgroup loads (new probe kind `tgload_xy`)

`scripts/mem_probes.py run --kind tgload_xy` (two consecutive words per lane into a register pair, 8 independent
loads then `wait`), next to `--kind tgload`, M2 Max G14C, core cycles per load (tgxy-*.json):

| stride (words between lanes) | groups | x | xy |
|---|---|---|---|
| 0 (broadcast) | 1 / 4 | 30.9 / 31.3 | 32.0 / 32.3 |
| 1 | 1 / 4 | 30.9 / 31.4 | 32.1 / 32.3 |
| 2 (xy: each lane its own pair) | 1 / 4 | 31.9 / 36.8 | 33.0 / 33.3 |

The model's 31 + (bank-conflict degree - 1) gives 32 for the xy case: right. Threadgroup load cost is not why
Geekbench's gemm_block_size_bias (8 xy + 2 broadcast loads feeding 16 FMAs per iteration) is +42%; varying
`AGX_TG_GROUP_LOAD` 8-31 leaves it at 76.5 us (hw 54.1).

## The +42%: FMAs on threadgroup loads deferred by every ALU instruction

Its issue profile charges each fmadd32 ~39 cycles (no `wait`: the FMAs read the loaded registers directly). That is the
behaviour `AGX_TG_SETTLE_ON_READ=1` removes (a threadgroup load read by an instruction has completed; ALU work no
longer defers it). With the rule, per capture (default -> settle; evaluation after first_touch_ns and inflight):

| capture | default | settle |
|---|---|---|
| gemm_block_size_bias | +42 | -4 |
| mul_mat_vec_f | +22 | -8 |
| gemm_non_block_size_bias | +3 | -19 |
| mul_mat_vec_q type7 1 column | +3 | -23 |
| mul_mat_vec_q type7 8 columns | -2 | -27 |
| soft_max_f32 | -2 | -15 |
| particle | -20 | -21 |

Means: 40 captures 10.3 -> 10.8, all 56 9.0 -> 9.5. The four that become fast end in a threadgroup reduction after a
barrier (the post-barrier tail the earlier bisection found in mul_mat_vec_q): the deferral was standing in for a
cost there that the model lacks. Still opt-in; next: a hardware probe of a write - barrier - read-by-other-groups
reduction tail, to model that cost directly and then enable settle-on-read.

## Follow-up: the generic reduction tail is modelled; the gap is in shuffle/add chains

- Hardware probe of a block-reduction tail (playground results/2026-11-01-tg-reduction-tail/tail.ptx: 4 groups store,
  barrier, every lane reads all 4 groups' words, adds, barrier; 38 and 304 workgroups): 375 / 400 ns an iteration;
  model 395 / 416, settle 369 / 409. Variants without barriers or reading only the own word: within -19..+27%, same
  either way. The post-barrier cross-group read is not what the four kernels lack.
- Their issue profiles (mul_mat_vec_q 8 columns, default vs settle): the whole difference is in the final
  simd_shuffle_xor / fadd32 / device_store chain, ~82 cycles an instruction by default (the deferral), ~0 with
  settle. Hardware needs that time, so the deferral masks a real cost of shuffle-reduction chains after threadgroup
  reads.
- The existing shuffle experiments do not supply it: settle + AGX_SHFL_PIPE 0.6 / 1: all 56 9.5 / 9.4; + AGX_SHFL_QUEUE
  8 / 16: 11.4 / 15.9 (default 9.0). Next: a hardware probe of mul_mat_vec_q's own tail (threadgroup partials ->
  shuffle-xor tree -> stores) to measure that chain directly.

## Shuffle-tree probes: settle-on-read on by default

Playground results/2026-11-01-tg-reduction-tail/shfltree.ptx (ITERS x C values reduced by a 5-step shfl.sync.bfly +
add tree; `_tg`: the values come through shared memory and bar.sync first; workgroups of 64), per-iteration slope ns:

| probe | WGs | hw | default | settle | settle + SHFL_PIPE 0.6 |
|---|---|---|---|---|---|
| tree1 | 38 / 608 | 66 / 135 | 51 / 133 | 51 / 133 | 51 / 168 |
| tree8 | 38 / 608 | 161 / 1243 | 145 / 984 | 145 / 984 | 172 / 1300 |
| tree1_tg | 38 / 608 | 260 / 328 | 397 / 397 | 272 / 275 | 272 / 275 |
| tree8_tg | 38 / 608 | 658 / 1317 | 1719 / 1722 | 662 / 986 | 693 / 1467 |

The default deferral is +161% on chains read from threadgroup memory; settle-on-read is +1%. At 32 groups a core,
8 parallel chains are 21-25% fast without a shared shuffle/add pipe (`AGX_SHFL_PIPE` 0.6 fixes these: +5 / +11%), but
the pipe does not move the four one-shot tails (mul_mat_vec_q, soft_max, gemm_non_block), so it stays opt-in.

Evaluation (tree8 = these slopes, now in evalset_captures.sh): default 56: 9.0, tree8: 40.1, all 64: 12.9; settle
9.5 / 12.8 / **9.9**; settle + pipe 9.5 / 12.0 / 9.8. Settle-on-read is now the default (`AGX_TG_SETTLE_ON_READ=0`:
the old deferral). Open: the one-shot reduction tails at the end of mul_mat_vec_q / soft_max / gemm_non_block
(-15..-27%), which the deferral had been standing in for; the looping probes do not reproduce their cost.
