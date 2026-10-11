# Counted loops and ALU latency/issue rate on G14C

Status: confirmed on one device (driver-reported M2 Max G14C B1, kernel
6.19.14-400.asahi), direct DRM submission with no Mesa.

## Claim and scope

1. A counted loop built from `push_exec`, `while_icmp`, `jmp_exec_any` and
   `pop_exec` (Mesa's nested-loop pattern) runs with exact trip counts for
   K = 1..100,000, with partial lane masks, and with 1/2/4 independent chains.
2. Per SIMD group, a dependent 32-bit `iadd` or `fadd32` chain advances every
   2 issue slots, and 2 independent chains reach 1 instruction per slot. That is
   2.01 and 1.01 cycles if the top clock state is the 1,398 MHz GET_PARAMS maximum.
3. A taken loop iteration costs ~38 cycles (86.4 ns in the lowest state). Bodies
   of a few instructions hide inside it.
4. The clock has at least four levels. The lowest/top ratio of 3.16 matches
   1398/444 MHz. (This corrects the earlier "two states" reading, which saw only
   part of the ladder.)
5. Cores are independent; one core saturates at ~3.5–3.6 single-group
   equivalents. GPU-wide fp32 add throughput is 5.67 × 10¹²/s, 83% of non-FMA peak.

## Reproducer and evidence

`make agx-loop-test` (emulator gate + hardware), `make verify-loops` (17 offline
checks), `make agx-loops` (full re-measurement).
[Report](../../experiments/control-flow/loops-and-alu-timing.md) ·
[evidence](../../results/2026-10-10-agx-loops/README.md).

## Relationship to existing knowledge

Instruction semantics come from applegpu pseudocode, and the loop shape from Mesa
`agx_compile.c` `emit_loop`. Both were verified here on hardware. 128 FP32 lanes
per core and a ~13.6 TFLOPS FMA peak are Apple's published figures; the measured
saturation is consistent with them. The cycle figures depend on the plateau being
1,398 MHz, while the 2:1 and 1:1 ratios do not.

## Compiler implications and limitations

Producer/consumer distance 2 suffices for full ALU issue. Unroll loops to roughly
8–16 instructions per iteration. ALU-bound kernels need ≥ 4 resident SIMD groups
per core. Not covered: other instruction classes, memory latency, divergent loops,
and why 4 groups per core gain only ~2×.
