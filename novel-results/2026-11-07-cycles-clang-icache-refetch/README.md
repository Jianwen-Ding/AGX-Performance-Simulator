# Cycles (clang build): the simulator was 3x low on shade_surface; instruction-cache refetch is most of the gap

Status: two captured launches of Blender Cycles built with clang (/scratch/perf-goal/sim/ccap), replayed on the M2 Max with
the clock held up (warm_captures.py's keeper), compared with agx_dispatch --launch-profile runtime.

## Findings

1. **The reported -25% was against the wrong hardware time.** capcheck.py found no other launch with the captured
   arguments (the work count differs), so it fell back to the median of every one-workgroup shade_surface launch of the run
   (1 to 64 paths, 385-1543 us): 916.9 us. The captured 24-path launch itself took 2526 us cold. Replayed warm (8 runs,
   memory restored each run): **2072 us** (min 2008; without the keeper 4209). Simulator 684 us: **-67%**, not -25%.
   intersect_closest (695 workgroups) replayed 3549 us against 2104 (-41%). capcheck's geometry fallback should not be
   trusted for kernels whose time depends on a work-count argument.
2. **Not divergence, not occupancy.** Replaying shade_surface with its work-count uniform at 1-24 paths
   (`hw-sweep-shade-surface.jsonl`): hardware/simulator is 2.95x with **one active lane** (876 vs 297 us, 34,000
   instructions, 36 cycles each) and 3.0-4.3x at every count. One lane, one SIMD group: per-instruction serial cost is missing.
3. **Stack memory is modelled right** (probe_stack_latency.py, one group, 16 KB frame): L1 103 vs 104 ns, L2 161 vs 168 ns
   per load-wait-store iteration; only beyond L2 (512 KB of rows) the model is low (417 vs 648). The kernel's stack loads are
   78% L1 / 22% L2 in the model.
4. **Instruction fetch is the gap.** The kernel is 1.8 MB; one lane executes 1,910 distinct lines (122 KB), 24 lanes 3,187
   (204 KB), against a 12 KiB per-core cache: an LRU of that size misses 2.8k / 7-8k times, against the 1,910 / 3,187 cold
   lines the runtime profile charged (104 ns each, overlapped with execution). Hardware (playground
   novel-results/2026-11-07-uncached-jump-fetch): 70 ns per line falling through, **235 ns** for a taken jump to an
   uncached line; the model had ~110 for both.

## Change

`Launch::icache_lines` (runtime profile 168, `AGX_ICACHE_LINES`, 0 = off): when a wave executes more distinct lines than
that, the simulation fetches through a per-core LRU instruction cache (fall-through miss 64 ns, taken-jump miss 200 ns, so
that probe_jump_fetch.py's straight / rand / rand4 come out 70 / 230 / 441 ns against 70 / 235 / 451 measured), the group
waiting, plus the 32-entry taken-jump structure (+127 cycles for a taken jump to a cached line without an entry); the
launch's per-line fetch term is then not charged. Kernels whose executed code fits are unchanged.

| | hardware | before | after |
|---|---|---|---|
| shade_surface, 24 paths | 2072 us | 684 (-67%) | 1332 (-36%) |
| shade_surface, 1 path | 876 | 297 (-66%) | 516 (-41%) |
| intersect_closest | 3549 | 2104 (-41%) | 2299 (-35%) |
| stream/FFT 8, latency 8, tree 8 (evalset_captures.sh) | | errors unchanged | unchanged (code fits) |

cargo test, verify_cost_model (identical) and dispatch_check (36 dispatches, median 4.8%) pass.

## Open

- A third of shade_surface is still missing. Not measured: call / ret (self-built call kernels fault the GPU), the
  replay's data being colder than in the application (memory restored from the CPU: 4.9 GB, TLB), the 1.19x capacity loss
  of 8-byte crossings beyond the 168 lines taken, the non-LRU replacement, the seq-jump case (324 vs 230 ns).
- The replay (2072 us) may be slower than the same work inside Blender (neighbouring launches 900-1500 us).
- Not re-fitted: the runtime launch profile's 104 ns per line on 56 captures (small code, unaffected).
- GPU faults of another process coincided with the hardware runs here (playground note): replay call-kernel captures only on
  an idle GPU.
