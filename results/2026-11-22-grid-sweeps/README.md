# Grid sweeps of under-predicted held-out launches (2026-11-22, hardware vs model)

`scripts/grid_sweep.py CAPDIR F1,F2,...` replays a capture on the M2 Max (`agx_run_hw --back-to-back 300`, twice,
warm clock) and in `agx_dispatch` (runtime profile) with the grid's largest dimension shrunk to each fraction. The data
is the capture's, so a smaller grid touches a subset of it.

## Observations

**cfd `cuda_compute_flux`** (no stack; 979 instructions, 56 device loads per thread, neighbour gathers; 12.8 MB,
SLC-resident). Data in `sweep-cfd-gpyramid4.txt`.

| workgroups / core (waves) | 1 (1) | 2 (1) | 3 (1) | 5 (2) | 7 (3) | 10 (4) | 14 (5) |
|---|---|---|---|---|---|---|---|
| hw us | 15.5 | 17.9 | 24.1 | 27.1 | 33.5 | 42.4 | 50.8 |
| model us | 12.4 | 12.8 | 13.3 | 18.5 | 23.6 | 29.3 | 36.4 |

Inside one wave, the hardware pays about 4.3 µs for each extra resident workgroup (6 SIMD groups). The model overlaps
them almost completely (1 -> 3 workgroups: +2083 core cycles = 1.5 µs). Past the first wave both grow about linearly
with waves, but the model starts from the low first-wave value. So the error (-20% to -45%) comes from resident
workgroups that do not overlap on the hardware. That suggests a per-core throughput limit the model lacks for this
kernel's loads. In the model 85% of them are SLC hits at 0.5 segments / cycle / core. The first-touch rule charges
the 2 MiB footprint at the DRAM rate (`AGX_DEBUG_FOOTPRINT`: first_touch_frac 1.0), and `AGX_FIRST_TOUCH_FRAC=0` /
`AGX_SLC_SMOOTH=0` change nothing.

**Halide local_laplacian `gPyramid4`** (2x2-thread workgroups, one SIMD group of 4 lanes each): within one wave,
6 -> 33 workgroups per core cost the hardware +6.1 µs (226 ns per workgroup per core), and the model +1.3 µs (48 ns).

**hotspot3D `hotspotOpt1`** (160 B stack, 77 stack loads / 44 stack stores of 302 instructions). Data in
`sweep-hotspot3d.txt`. Inside the first wave each workgroup costs about 15 µs on both, and the model is 9% low at
4 per core. Later waves cost the hardware 27-29 µs each against the model's 19-23 µs. The model charges the stack-slot
set-up (5 µs + 1.15 µs x 32 slots) once in the first wave, and its per-wave compute is about 30% fast.

## Next

- Count the segments per load for cfd's gathers in the model (memtrace), and measure the per-core gather throughput on
  the hardware with a probe that gathers from an SLC-resident buffer at 1-3 workgroups per core.
- Check whether the throughput cap that fits cfd also fixes gPyramid4. Its 4-lane groups each issue loads that
  touch few segments, so a per-instruction or per-group cost is the other candidate there.
- Model changes must be re-scored on the full held-out and llama sets (`/scratch/agx-r5/r9/evalsets.sh`).
