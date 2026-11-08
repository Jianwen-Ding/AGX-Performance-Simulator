# Predicting real launches: Blender Cycles against warm hardware time

Status: measured on one device (driver-reported M2 Max G14C B1). Scene: Blender Open Data junkshop, six captured launches, compiled by the PTX front end
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
   (`scripts/probe_stack_cost.py`). Traces now place each word in a 128-byte slot (32 lanes of 4 bytes, the PTX front end's scratch layout), per SIMD group.
5. **A lone launch runs at a low clock.** The same launch is 2.4-3.2x slower alone than with the GPU kept busy, and varies 2x. Reference times must hold the
   clock up: `scripts/warm_captures.py` keeps a small kernel running in another process (the same launch then repeats within about 1%). The model
   predicts the warm time, and the low clock is not modelled.

## Result

Predicted / warm: intersect_closest 0.81, queued_paths_array 1.10, sorted_paths_array 0.90, intersect_shadow 0.73, shade_light_nee 0.57, shade_surface 0.57
(1447 / 2536 us). Before: an error on every one. Still optimistic where calls and shading are, and all but two are low, so a real cost is missing. The
curated verifiers are unchanged.

Not established: `call` / `ret` cost (taken as a taken forward jump), line size, fetch contention between cores, texture sampling (these captures use the
software sampler), other scenes. These are six launches of one scene, with the fetch and stack constants from probes and not fitted to them, but the same
launches guided which costs to look at, so they are not an independent test.

**Measuring calls directly failed, and 6 runs faulted the GPU** (`Bad address`, in `gpu_safety`): kernels with real calls built by `llc` or by `zoc` for
`agx_run_hw` fault, inline ones and captured kernels with calls do not, and why is not known. Stopped there. Find the difference from a capture before
trying again, offline first. Ruled out, a sixth fault: code length (the same kernel padded with `stop` to 1 MB). Same in both: `--regs 256`, `--stack 12336`, 384
threads, a stack_adjust. Not yet compared: the captures' mapped memory and uniforms against this kernel's single `--alloc`.
