# Task SPIR-V 8: applications (ray tracer, neural networks, image processing, cellular automaton, graph search)

**Workloads:** `raytrace`, `mlp`, `sobel`, `life`, `bfs`, `conv_layer`, `hist_eq`. These are recorded and verified by `make asahi-verify` with the others;
they are **PENDING** until then. The detailed procedure is in `tasks/spirv-workloads.md`.

## Why these

The other workloads test instructions and algorithms. These three are small but complete applications: the kind of
thing someone would actually run on the GPU. Each produces a result a person can check by eye, as well as by the
reference.

| Workload | What it computes | Output |
|---|---|---|
| `raytrace` | Three spheres on a checkered ground plane, a directional light with hard shadows, Lambert shading, one mirror bounce (the blue sphere and the floor reflect) | RGBA8 image: `image0.ppm` from `run`, `sim_image0.ppm` from `sim` |
| `mlp` | A 128 → 256 (ReLU) → 16 (softmax) perceptron over a batch, one workgroup per input | Class probabilities; the check also requires the same predicted class as float64 |
| `sobel` | Sobel gradient magnitude, min(\|gx\| + \|gy\|, 255), on an 8-bit 4K image | 8-bit image, `image1.pgm`; integer-exact |
| `life` | Game of Life, 64 generations on a 2048² torus (one dispatch each) | Final grid as `image0.pgm`; exact |
| `bfs` | Breadth-first search from vertex 0 over a 1 Mi-vertex random graph, one dispatch per level | Distances and the number of vertices found per level; exact |
| `conv_layer` | A 3×3 convolution layer, 16 → 32 channels, ReLU, on 224×224 | Feature maps; within a summation error bound |
| `hist_eq` | Histogram equalisation of a low-contrast 4K image | Equalised image `image2.pgm` (contrast stretched to 0..255); exact |

## Offline sanity checks (not hardware verification)

- **lavapipe:** all three pass at both sizes.
- **Shim-compiled Mesa 25.2.8 and 26.0.3 code on the functional simulator:** 14 of 14.
  - `bfs` found a simulator bug. Mesa aggregates `atomicAdd(found[l], 1)` under divergence: one lane, chosen with SR58
    (`active_thread_index_in_simdgroup`), adds SR59, the number of active lanes. The simulator modelled both from the
    launch shape (lane index, 32), which is only right when every lane is active, so the counts came out as multiples
    of 32. They are now the rank among, and the count of, the active lanes. The recorded suite only has full-exec SR
    reads, so it could not catch this.
  - **On the M2, confirm the active-lane meaning of SR58 / SR59.** `bfs` PASSing word for word confirms it.
- **Cost model:** predictions for the large sizes.
- **`raytrace`:** checked against an independent float64 numpy version of the same scene. Pixels on silhouettes and
  shadow edges may flip between float implementations. The check allows 1% of pixels to differ by more than 2 levels:
  lavapipe is at 0.09% at 1920×1080, and the simulator at 0.4% at 96×64. The far ground is a uniform grey on purpose:
  a distant checkerboard aliases, and any rounding difference flips its squares.

## On the M2

1. After `make asahi-verify`, look at the images: the GPU's render and the simulator's render of the recorded code
   should be indistinguishable. Convert them to PNG with any image tool.
2. `sobel` must be word-for-word identical (PASS).
3. `raytrace` and `mlp` use sqrt / exp / rcp, so expect CLOSE. Record the number of differing pixels and their
   largest difference.
4. Compare the large-size times with `predict`, and report frames per second for the 1080p ray tracer, and inferences
   per second for `mlp`.
5. `life`, `bfs` and `hist_eq` are exact and multi-dispatch. They must PASS, and they show whether results carry
   correctly across dispatches. For `life` and `bfs`, compare the time per dispatch (64 and 32 small dispatches) with
   the launch overhead.
