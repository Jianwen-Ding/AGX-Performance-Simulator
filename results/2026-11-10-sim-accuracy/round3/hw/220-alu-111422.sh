# ALU loop throughput per core vs groups (probe_sustained showed 8 fmadd chains x 8 groups at sim/hw 0.71)
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=200
/scratch/agx-r5/waitidle.sh 5
cd /save/workspace/AGX-Performance-Simulator
NG="1 2 4 8 16 32" T1=1000 T2=3000 python3 scripts/probe_tg_alu.py both ffffffff iiiiiiii ffff
