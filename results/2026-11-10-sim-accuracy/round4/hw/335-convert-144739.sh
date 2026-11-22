# convert u32_to_f / f_to_u32 throughput (soft_max's prologue: integer division by float reciprocal; the model
# prices 8 converts per trip at 134.6 cycles with 16 groups) vs iadd
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw
/scratch/agx-r5/waitidle.sh 5
cd /save/workspace/AGX-Performance-Simulator
NG="1 4 16 32" T1=2048 T2=6144 python3 scripts/probe_tg_alu.py both vvvvvvvv uuuuuuuu viviviviv iiiiiiii
