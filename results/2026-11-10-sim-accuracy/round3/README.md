# 2026-11-10 round 3: k_bin_bcast store form, ALU issue / register-read rate, MMQ core sampling and busy-core slowdown

Hardware findings (apple-documentation-playground novel-results/2026-11-10-store-address-form,
2026-11-10-alu-issue-rate, 2026-11-10-l2-region-streaming). Raw hardware logs and job scripts in `hw/` (queue
numbers 200-270, zz8, zz9). Evals: `evalsets.sh OUTDIR [env]` (no GPU; per-set summary `summ.py DIR...`).
Probes added: `scripts/probe_sustained.py`, `scripts/probe_l2_share.py`; `probe_hot_lines.py` ROWW / NOSTORE /
OUTMB / PRE / POST / REGS; `probe_tg_alu.py` patterns a / j / g / k.

## Model changes

Adopted (default on, switch to turn off):
1. `AGX_PLACED_ALL_N` (default 38; 0 = off): placed grids of at most 38 workgroups simulate every core's list.
   The budget-limited sampling (2e7 cycles: 5 of 34 cores for MMQ Q6_K) ranked cores by group-0 path length; for
   MMQ's stream-k workgroups those are 204.9k-206.4k instructions for all, and the three slowest (2.68-2.82 ms,
   the rest 2.04-2.25) ranked 23rd-32nd. Q6_K 4156 -> 5225 us (hw 5526).

Experiments (default off):
- `AGX_PLACED_BUDGET`: the placed grids' cycle budget (else AGX_CORE_CYCLE_BUDGET, else 2e7).
- `AGX_RANK_GROUPS=1`: rank placed cores by their longest group's path (no better for MMQ).
- `AGX_STREAM_OWNER=1|2|3`: fetch streams per group (1), another group's line gives the stream cost but no early
  start (2), or the early start but the full cost (3). `AGX_STREAMS=k`: k stream heads per core (no effect).
- `AGX_ISSUE_PER_CORE`, `AGX_HALF_READ_PORTS`: core issue width (4) and register reads per half per cycle (2).
- `AGX_STACK_STORE_FLOOR=1|2`: stack stores against the GPU-wide store floor (all lines / distinct per group).

## Experiments on the eval sets (before = e5, round 2 final)

| run | tg mean | tiny mean | fa mean | pp med / mean / w10 | held-out med / mean / p90 / w10 |
|---|---|---|---|---|---|
| e5 (round 2 final) | 9.3 | 13.8 | 9.0 | 8.5 / 12.6 / 12 | 5.5 / 7.6 / 18 / 119 |
| AGX_STREAM_OWNER=1 | 9.3 | 22.8 | 6.0 | 9.5 / 11.9 / 12 | 5 / 7.6 / 18 / 119 |
| AGX_ISSUE_PER_CORE=2 AGX_HALF_READ_PORTS=1.3 | 8.5 | 13.2 | 6.8 | 7 / 12.2 / 13 | 7 / 9.5 / 19 / 111 |
| AGX_HALF_READ_PORTS=1.3 | 8.9 | 13.2 | 8.5 | 7.5 / 12.2 / 13 | 5 / 7.7 / 18 / 115 |
| AGX_HALF_READ_PORTS=1.6 | 8.9 | 10.8 | 8.8 | 8.5 / 12.6 / 13 | 5 / 7.6 / 18 / 115 |
| AGX_PLACED_BUDGET=2e8 | 9.3 | 13.8 | 9.0 | 8 / 11.6 / 13 | 6 / 7.7 / 18 / 118 (heartwall failed: host load) |

AGX_STREAM_OWNER=1: bucketcount -21 -> +1, MMQ Q6_K -25 -> +2, Q5_K -22 -> +10, fa prefill -14 -> +2, but Q5_1
-7 -> +18, Q8_0 +3 -> +14, tinygrad r_64_16_32 +21 -> +66. Modes 2 / 3 lie between (`modes23.txt` in scratch).
With every MMQ core simulated the owner rule overshoots the per-workgroup grids (Q6_K 17 workgroups: 3488 vs hw
2693), so the MMQ gain was the sampling, not the fetch rule.

## Final (branch head, defaults) vs round 2 final (e5)

| set | before: median / mean / p90 / within 10% | after |
|---|---|---|
| llama tg (10) | 9 / 9.3 / 19 / 6 | 9 / 9.3 / 19 / 6 |
| tinygrad (5) | 13 / 13.8 / 21 / 2 | 13 / 13.8 / 21 / 2 |
| llama fa (4) | 9.5 / 9.0 / 14 / 2 | 9.5 / 9.0 / 14 / 2 |
| llama pp512 (20) | 8.5 / 12.6 / 27 / 12 | 8 / 11.6 / 27 / 13 |
| held-out Rodinia + Halide (160) | 5.5 / 7.6 / 18 / 119 | 5.5 / 7.6 / 18 / 119 |

The only launch that moved by 2 points or more: MMQ Q6_K (pp) 4155.6 -> 5224.8 us (hw 5526.2): -25 -> -5%.
Per-launch: `eval/fin1`. Still -22% Q5_K (all its cores simulated already: busy-core slowdown below), -48% k_bin_bcast.

## Still open
- k_bin_bcast -48%: the cost is the register-address store form (hw: uniform form 22.3 us, register form 32.8, no
  store 17.4), but a synthetic probe with the same form and addresses does not reproduce it.
- MMQ Q5_K / Q6_K: per-workgroup time on hw grows with busy cores beyond 16 (Q5_K 1645 -> 2146 us from 16 to 34;
  clusters 2 and 3 joining cost +195 / +254), the model is flat. Not load bandwidth (probe_l2_share: 1.8 TB/s with
  +7%), not stack-store bandwidth at the 365 GB/s floor (735 us floor < 1485 us core time).
- ALU issue / register-read limits: measured (2 issues, ~2.4-3 reads per cycle per core at 8+ groups), not adopted
  (real kernels overshoot with them).
- bucketcount -21% (fetch stream sharing), Cycles intersect_closest -36%, rms_norm pp -27%, soft_max +17%.
