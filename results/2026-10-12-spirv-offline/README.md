# SPIR-V workloads on the real AGX compiler, offline (2026-10-12)

The eight SPIR-V workloads (`spirv/shaders`, `tasks/spirv-workloads.md`) compiled by Mesa's own Asahi Vulkan driver
(honeykrisp) without an Apple GPU, then run on the functional simulator and timed by the cost model. This uses no
hardware measurements: the predictions here are what the M2 runs (`tasks/spirv-0*.md`) will test.

## How the code was produced

`scripts/build_mesa_shim.sh VERSION` builds Mesa from the Ubuntu archive's release tarball with the Asahi drivers
and the no-op drm-shim, patched to report the M2 Max (G14C B1, 38 cores) with the kernel's VM range. The upstream
shim reports a G13G with a VM range too small for the driver's heaps (Mesa 25.2) and lacks two ioctls. `vk_run`
then runs normally: the driver compiles for G14C and exports its binaries. Submissions do nothing, so `compile`
ignores outputs and times.

`make spirv-compile` (default 26.0.3) wrote `spirv/compiled/mesa-26.0.3/`, and the same with 25.2.8 wrote
`spirv/compiled/mesa-25.2.8/`. The M2 runs 25.3.6, and neither source archive is reachable from here.

## What the binaries say (offsetof() on both built drivers)

- **Blob layout.** The MesaVkShaderBin is laid out as:
  1. `vk_shader_bin_header`: 72 bytes up to 25.3, 80 bytes from 26.0;
  2. a stage byte;
  3. `hk_shader_info`: 976 bytes;
  4. `agx_shader_info`: 1176 bytes, unaligned;
  5. code size and data size;
  6. the code.

  `agx_shader_info` gives the main and preamble offsets, the workgroup size, the threadgroup bytes, `nr_gprs` (at
  byte 1094 in 25.2 and 1096 in 26.0) and the constants (`rodata`). The driver loads `rodata` into uniforms from
  `base_uniform` before the preamble runs. Example: sgemm's 254 halves of constants go to u20..u146. `extract` now
  reads all of this, so it no longer has to guess the programs.
- **Root table** (`hk_root_descriptor_table`; 32-bit word indices).

  | Field | 25.2 | 26.0 |
  |---|---|---|
  | group-count pointer | word 2 | word 18 |
  | base_group | words 4..6 | words 20..22 |
  | push constants | from word 92 | from word 122 |
  | set addresses | word 156 | words 2..17 |

  - In both versions the driver preloads u0_u1 with the table's own address.
  - 26.0 also preloads u2_u3 with the set address.
  - The M2's 25.3.6 control shader reads base_group from words 20..22 and a push word from word 124. That matches
    26.0, so it should use `spirv/root_layout.mesa-26.0.json`. `extract` chooses this layout whenever the preamble
    reads `u2_u3`.
- **Statistics** (26.0.3).

  | Workload | Instructions (25.2.8 / 26.0.3) | Register halves | Max threads per core | Threadgroup bytes | Workgroup |
  |---|---|---|---|---|---|
  | bitonic | 104 / 104 | 29 | 1024 | 8192 | 512 |
  | conv2d | 132 / 132 | 28 | 1024 | 1600 | 16×16 |
  | histogram | 42 / 43 | 16 | 1024 | 1024 | 256 |
  | nbody | 60 / 60 | 40 | 1024 | 4096 | 256 |
  | reduce | 87 / 85 | 24 | 1024 | 512 | 256 |
  | scan | 131 / 131 | 22 | 1024 | 256 | 256 |
  | sgemm | 761 / 741 | 68 | 1024 | 8320 | 16×16 |
  | sha256 | 1149 / 1117 | 160 | 640 | 0 | 64 |

  Every instruction in the 16 binaries decodes.

## Functional simulation: 16 / 16 pass (`make spirv-sim`, `sim.json`)

All eight workloads match their numpy references (small size) with both compilers' code. Integer results are exact,
and float results are within the reference tolerance.

Getting there needed these simulator fixes, each found by these binaries:

| Fix | Symptom before | Workload |
|---|---|---|
| Threadgroup base operand: a 16-bit uniform (`At` = 1 or 3) or zero (`At` = 2). Mesa keeps constant shared offsets such as `part[1]` in uniforms, which the simulator had read as 0. The same fix applies to the cost model's bank-conflict words. | wrong totals | scan, reduce |
| `get_sr` into a 16-bit register (Mesa writes `sr58` to `r0h`). | stopped | reduce |
| Register-file size passed in 16-bit registers, as `Compiled::new` expects. The simulator had passed 32-bit registers, which cut the file in half. | stopped at r64 and above | sha256 |
| Uniform file of 512 halves (u0..u255). applegpu's file has 256 halves, but sgemm's driver code uses up to u146. | — | sgemm |

The hardware test suite is unchanged at 691 / 721, and every timing verifier gives identical output.

## Timing from real paths (`make spirv-predict`, `predictions.json`)

`agx_dispatch` normally traces SIMD groups with the cost model's emulation, which reads memory as zero. Driver code
cannot run that way: the preamble fills the uniforms from the root table, and loops and addresses depend on data.

`func::trace_workgroups` now records each workgroup's paths in the cost model's trace format, using real memory. The
cost model's own traces are recorded before execution. `agx_dispatch --preamble/--map` uses these functional paths
instead, and `spirv_task.py predict --code-dir` drives it.

The check is `make verify-traces`. On the hardware suite's first workgroups the two tracers give identical paths and
accesses in 652 / 721 cases. The other 69 cases all depend on data:

- 18 control cases with data-driven conditions;
- the 50 random programs with if/else and loops;
- histogram64, whose bin addresses come from data.

Predicted times (large size, 26.0.3 code, 1398 MHz). Not yet validated: the M2 runs will check them.

| Workload | Predicted | Dispatches | Implied rate |
|---|---|---|---|
| reduce | 467.5 us | 2 (412.8 + 54.7) | 287 GB/s |
| scan | 43.9 us | 3 | 383 GB/s |
| sgemm | 2069.2 us | 1 | 1.04 TFLOP/s |
| conv2d | 290.0 us | 1 | 723 GFLOP/s |
| histogram | 58.6 us | 1 | 286 GB/s |
| nbody | 6596.7 us | 1 | 814 GFLOP/s |
| bitonic | 1719.7 us | 66 | – |
| sha256 | 4761.8 us | 1 | – |

`predict` starts each dispatch from the initial buffers, so later dispatches do not see earlier ones' writes. This
matters only where control flow or addresses depend on those writes. In these workloads the candidate is bitonic's
compare-exchange, which has not been checked.
