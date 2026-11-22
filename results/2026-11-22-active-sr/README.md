# sr58 / sr59 under divergence (2026-11-22)

Direct check of follow-up 11 (`tasks/ptx-frontend-agx-followups.md`). Under the atomic optimizer it reported sr59
reading 11 with one active thread and 1 with 32. Eight masks were run on the M2 Max (`agx_run_hw`): `tid < 1/5/32/40/64`,
odd lanes, `tid & 7 == 3`, and `tid & 31 == 31`. On the hardware, sr58 is the lane's rank among the active lanes of its
SIMD group and sr59 is the number of active lanes in that SIMD group. With one active lane, sr58 = 0 and sr59 = 1;
with 40 active threads, the second SIMD group reads 8. `agx_run` matches all eight byte for byte, so the reported
values do not come from the simulator's SR emulation. `python3 scripts/active_sr_check.py` replays them offline.
