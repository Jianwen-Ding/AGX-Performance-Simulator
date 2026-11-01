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
