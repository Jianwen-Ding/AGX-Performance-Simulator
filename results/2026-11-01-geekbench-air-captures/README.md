# Geekbench 7 GPU kernels on the AIR route: simulator vs hardware (2026-11-01)

The 29 kernels with the most GPU time in a Geekbench 7 GPU (CUDA) run on zluda_agx's AIR route (ZLUDA
claude/perf-air-agx), except the ray tracer's (trace_bvh*, miss, hit_surface, generate_rays, resolve_radiance,
accumulate_*: replays of BVH kernels with wrong data have hung the GPU), each captured after its 20th launch. Every
capture ran on the simulator first (none hit the step limit), then `scripts/capcheck.py --hw --hw-repeat 3` (the
median of 3 runs of 1000 back to back) against `agx_dispatch --launch-profile zluda`. No GPU faults.

| Model | Median abs. error | Within 10% | Within 25% |
| --- | --- | --- | --- |
| zluda profile, store floor launch + stores (`capcheck-hw3.txt`) | 18% | 9 / 29 | 23 / 29 |
| store floor overlapping the launch beyond the stores' time | 9% | 17 / 29 | 25 / 29 |
| and later waves warm when the data fits the system-level cache (`capcheck-final.txt`) | **8%** | 17 / 29 | 25 / 29 |

- The large image kernels write a lot over thousands of workgroups per core: their launch and their stores overlap
  (gradient: 967 us of workgroup launch, ~500 us of stores, 1036 us measured), so the store floor adds the launch only
  up to the stores' time; the store microbenchmarks (1.5-24 MB) stay within +4% to +14%.
- Later waves warm without the system-level-cache condition gave 6% (21 / 29 within 10%, 27 / 29 within 25%:
  image_rotate +35% -> -5%, convolve_2d +18% -> +1%) but made the direct-DRM calibration's DRAM-bound kernels too
  fast (verify-dispatch p90 9% -> 16%, max 74%): with the condition, the calibration is unchanged. The image kernels
  that would gain are classed as not resident by the footprint estimate (single-wave segments scaled to all cores,
  which counts the rows neighbours share several times). Capping the estimate at the captured data does not change
  that: their mapped allocations are 48-189 MB, above the 19 MiB the model takes for the system-level cache. Their
  data does not stay resident; what warms their later waves is reuse across waves (a stencil's next rows were read by
  the wave before). Next: warm only the lines a wave shares with the previous one (simulate wave 0 first, then the
  others against its lines).
- Open: gemm_block_size_bias +43% (several workgroups on one core modelled as overlapping less than they do, the
  known gap); blur_gray, image_rotate, hough_scores +27-38% (hundreds of waves, each simulated with its own latency,
  which the hardware overlaps); particle -22%; depthwise_convolution_2d +24%.
