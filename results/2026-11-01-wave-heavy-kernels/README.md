# Wave-heavy image kernels (+15..+34%): not chain depth, not DRAM latency

After the fetch-overlap change the largest misses are Geekbench's blur_gray +34%, image_rotate +22%,
canny_nonmax_suppression +21%, convolve_2d +15% (9,672 / 3,307 workgroups of 64 a core, 175-359 waves; hardware
~1.0 ms, just above the model's workgroup-launch floor of 967 us; the model's execution, waves x ~5,580 cycles,
is above it).

- Chain depth (`AGX_CHAIN_DEPTH`, new, default 2 = unchanged): 4 / 8 waves a slot move blur_gray 1437 -> 1409 / 1364,
  image_rotate 1249 -> 1170 / 1140, canny 1216 -> 1217 / 1229 (hw 1077 / 1024 / 1004). The head latency of each chain
  is not the gap.
- blur_gray's issue profile: ~half of a group's cycles in three waits on device loads (~560 cycles each) of a 23.6 MB
  image (beyond the SLC: DRAM latency). DRAM-resident first-touch latency measured with the column-sum probe
  (128 groups, 2 columns each): 32 / 64 MiB buffers 403 / 428 ns a wait on hardware, 430 modelled: correct.
- Same probe at 1216 groups, 159 MiB: hardware 360 GB/s, model 317 GB/s (+13% time): a bandwidth gap at full
  streaming load, separate from these kernels (they move ~44 GB/s).
- Remaining suspect: contention inside a wave. One workgroup alone takes ~3,400 cycles, a full wave of 27 (54 groups)
  ~5,580; hardware fits ~4,200 a wave. Next: a probe of a short latency-bound kernel at 1-54 groups a core.
