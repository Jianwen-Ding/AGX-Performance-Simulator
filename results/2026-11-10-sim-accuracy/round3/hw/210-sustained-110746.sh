# all-core / long-run slowdown? ALU loop, 1 workgroup per core
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw
/scratch/agx-r5/waitidle.sh 5
cd /save/workspace/AGX-Performance-Simulator
B2B=100 T="2000 20000 120000" N="1 8 32 38" python3 scripts/probe_sustained.py both
