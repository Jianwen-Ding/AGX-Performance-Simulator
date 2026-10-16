# Task safe-11: occupancy limits, imageblock SRs and the in-shader clock

Read [safe-probes.md](safe-probes.md) first (safety rules).

**Goal:** close the open parts of three direct-DRM findings that a CUDA runtime depends on, and add them to the
coverage suite (`MANUAL` items `occupancy`, `imageblock_srs`, `in_shader_clock`). This covers
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, the hardware limits behind `cudaDeviceProp`, and `clock()` /
`%globaltimer`.

## 1. Occupancy (`novel-results/2026-10-11-occupancy`, runner `scripts/run_drm_occupancy.py`)

**Known:** resident SIMD groups per core = min(96, 2 · floor(3328 / (2 · halves))), and 64 KiB of threadgroup memory
per core.

**Open:**
- **Why small workgroups cap lower:** 32-lane workgroups reach about 55 groups, 64-lane ones 84.
  - Sweep workgroup sizes 32, 64, 96, 128, 160 ... 1024 at 16 halves.
  - Fit "workgroup slots per core" against "launch rate". Count with the existing per-core atomic counter.
- **2-D and 3-D workgroups** (`AGX_LOCAL_Y/Z`) with the same thread count: is occupancy the same?
- **Register and threadgroup-memory limits together:** a small grid over both, to confirm that the most restrictive
  limit wins.
- **256 declared halves:** the launch field wraps to 0.
  - **Do not run it.** List it as unsafe in the report: the result of a wrapped register count is unknown.
  - Record instead the largest count that does not wrap (248).

**Coverage:** one case per row of the finding's table, checking SR19 / SR18 and the peak counter.

## 2. Imageblock SRs (`novel-results/2026-10-11-sr12-sr13-sr15-imageblock`)

- Record SR12 / SR13 / SR15 / SR18 under the 13 `AGX_SHARED_WORD` values the finding used, as coverage cases with
  launch geometry.
- **New, using only those same Shared-word values:** does an imageblock layout change threadgroup-memory capacity or
  occupancy, as measured with the §1 counter? Tile memory and threadgroup memory may share the 64 KiB.

## 3. In-shader clock (`novel-results/2026-10-11-in-shader-clock`, `scripts/run_drm_cpu_clock.py`)

**Known:** a CPU thread writes `CNTVCT_EL0` (24 MHz). The shader reads it with `f59d` + a coherent load, at
0.4-0.75 µs per read.

**Open:**
- **Cost under load:** 1, 32, 1024 and 65,536 lanes reading the clock in every SIMD group at once. Measure the read
  cost and whether lanes see stale or skewed values.
- **Staleness across cores:** the clock-word spread between cores at the same instant. Run a straight-line program
  where every core reads twice, and record SR20.
- **Lines:** with `AGX_CPU_CLOCK_LINES=N`, does reading a per-core copy reduce cost?
- **Cost model:** add the clock read as an op so ZLUDA's `clock()` / `%globaltimer` lowering can be costed.

**Coverage:** the read must be monotonic per lane, and two reads around a fixed loop must differ by at least the
expected ticks. This is a property check, not exact words.

## Safety

- Only the runners above and the encodings they already use.
- All loops are counted (`agx_loop`, emulator-gated). The clock is sampled a fixed number of times; never loop until
  a time is reached.

## Deliverables

- Coverage cases for the three items.
- Updates to the three playground findings.
- Occupancy formula changes in `cost.rs` / `dispatch.rs` if needed.
- Extended `make verify-occupancy` / `verify-in-shader-clock`. Push both repositories.
