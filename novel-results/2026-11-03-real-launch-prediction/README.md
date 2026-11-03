# Predicting real launches: Blender Cycles against warm hardware time

Status: measured on one device (driver-reported M2 Max G14C B1). Scene: Blender Open Data junkshop, six captured launches, compiled by ZLUDA
(`claude/cycles-from-source`). Data: `warm.jsonl` (GPU), `pred.jsonl` (model).

## Findings

1. **The model could not run real kernels.** It had no cost class for `call`, `ret` or `jmp_incomplete`, its decoder failed on padding between
   functions, and `agx_dispatch` had no `--sampler-heap`.
2. **Launches take their slowest workgroup's time, and workgroups differ.** `intersect_closest`, 26 workgroups: 424K to 1.25M cycles. The model sampled
   a few: 409 us where tracing all gives 900 us. `--all-workgroups` traces them all.
3. **Cold instruction fetch is a large, simple cost.** Straight-line code runs at 8.85 ns per 8-byte instruction on the M2 Max from 64 KB to 4 MB of
   code, warm and back to back, with 1 or 12 SIMD groups alike: the model had 0.8 ns (`scripts/probe_fetch_scaling.py`). Taken as about 62 ns per 64-byte
   line on top of issue, per core, the first group to reach a line paying. The line size is an assumption; the rate is what is measured.
4. **Stack memory goes through the caches.** A stack load: 68 ns with a 256-byte frame, 172 ns with 4 KB; stores 55-62 ns at both
   (`scripts/probe_stack_cost.py`). Traces now place each word in a 128-byte slot (32 lanes of 4 bytes, ZLUDA's scratch layout), per SIMD group.
5. **A lone launch runs at a low clock.** The same launch is 2.4-3.2x slower alone than with the GPU kept busy, and varies 2x. Reference times must hold the
   clock up: `scripts/warm_captures.py` keeps a small kernel running in another process (the same launch then repeats within about 1%). The model
   predicts the warm time, and the low clock is not modelled.

## Result

Predicted / warm: intersect_closest 0.81, queued_paths_array 1.10, sorted_paths_array 0.90, intersect_shadow 0.73, shade_light_nee 0.57. Before: an error
on every one. The curated verifiers are unchanged.

Not established: `call` / `ret` cost (taken as a taken forward jump), line size, fetch contention between cores, texture sampling (these captures use the
software sampler), other scenes.
