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

## Occupancy and wave probes: both modelled; the gap is cross-core reuse (hypothesis)

Column-sum kernel with 3 waits a group, workgroups of 64 (playground colsum.ptx, `--rows 96`):

- One wave, k workgroups a core (k = 2..29): hw 4.2 / 4.4 / 4.9 / 6.2 / 8.5 us at 3 / 5 / 9 / 17 / 29, model -1..-7%.
  Contention inside a wave is right.
- 27 a core x 3 / 11 / 42 waves: hw 22.5 / 78.9 / 307 us, model +5 / +8 / +8%. Steady-state waves are right.
- `AGX_PARTIAL_WARM=0` does not move the four image kernels (their data exceed the SLC; the warm rules do not apply).

What the image kernels have and the probes do not: stencil reads of lines that a neighbouring workgroup, usually on
another core, read shortly before (3x3 neighbourhoods; rotated reads). On hardware those are GPU-wide L2 hits; the
model simulates one core's workgroups and charges that core's first touch of a line as a fetch whatever the other
cores did (`inflight` covers only this core's groups; `sharing` only scales capacity). Next: a hardware probe of
reads of lines another core loaded a little earlier, then a cross-core reuse rule.

## Cross-core reuse confirmed; DRAM traffic now shared between neighbouring workgroups (`dram_share`)

Probe (playground colsum.ptx `colsum_overlap`: group t sums columns t and t + 1, each column read by two neighbouring
groups, usually on different cores; 1024 rows, 27 workgroups of 64 a core x 4 / 16 waves, 64 / 256 MiB buffers):

| | hw | model before | model now |
|---|---|---|---|
| private columns, 4 / 16 waves | 197 / 739 us | -1 / +4% | unchanged |
| overlapping columns, 4 / 16 waves | 109 / 383 us | +35 / +50% | -8 / +3% |

Hardware reads ~350-363 GB/s of distinct data either way: a line two cores read costs DRAM bandwidth once. The
model now estimates, for a mid-grid workgroup and its x +-1 / y +-1 neighbours, the mean of 1 / (1 + neighbours that
also read each line) and divides the core's DRAM traffic by it (`AGX_DRAM_SHARE=0`: off). image_rotate +22 -> -5,
canny_nonmax_suppression +21 -> -3; 40 captures 9.2 -> 8.6, all 56 8.2 -> 7.7, all 64 8.8 -> 8.3.

Rejected: also starting a (1 - share) part of the lines in L2 (a neighbour fetched them first): all 56 7.5, blur_gray
+34 -> +29, but the overlapping-column probe goes to -30% at 4 waves, so the probe does not support it. blur_gray
(+34%, latency-bound at ~44 GB/s) stays open.
