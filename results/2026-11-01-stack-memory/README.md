# Stack (scratch) memory: simulator vs hardware (2026-11-01)

`stack.ptx` (`stackrw`): a 64-word `.local` array per thread indexed by a per-lane LCG (scattered offsets), 2000
load + add + store iterations; `stackuni.ptx`: the same with one index for all lanes per iteration ((7i + 3) & 63).
zoc (AIR route) puts the array on the stack (304 bytes per thread). Run through zluda_agx (`agx_bench_kernel`,
captured), then `agx_run_hw --back-to-back 200 --stack 304` and `agx_dispatch --launch-profile zluda`.

| Launch | hw | sim before (stack = L1 hit) | sim with stack_segments |
| --- | --- | --- | --- |
| scattered, 38 x 32 | 450 us | 220 (-51%) | 370 (-18%) |
| scattered, 38 x 256 | 1997 us | 220 (-89%) | 1464 (-27%) |
| scattered, 304 x 256 | 10079 us | 656 (-93%) | 5838 (-42%) |
| uniform, 38 x 32 | 345 us | 211 (-39%) | 340 (-2%) |
| uniform, 38 x 256 | 384 us | 211 (-45%) | 341 (-11%) |
| uniform, 304 x 256 | 1362 us | 629 (-54%) | 1022 (-25%) |

- The uniform index is 5.2x faster than the scattered one at 16 groups per core: the stack is interleaved across the
  lanes of a SIMD group (lanes at one offset share a row), not one contiguous stack per lane.
- `func::stack_segments` gives each SIMD group a region with that layout, so stack accesses go through the memory
  model like device accesses. make verify-dispatch / verify-cost-model / suite-check unchanged.
- Open: scattered stack traffic at high occupancy is still 27-42% too fast (store-to-load ordering on the same row,
  or the store rate); llama.cpp's mul_mat_vec_f kernels (95 workgroups of 32 threads per core, 64-80 bytes of stack)
  stay at -72% / -39%, so their gap is elsewhere (not their stack traffic).
