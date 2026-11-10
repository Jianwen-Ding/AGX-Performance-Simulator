# do early-exiting SIMD groups free their slots? (rope: 7 of 8 groups exit)
cd /save/workspace/AGX-Performance-Simulator
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=1000
/scratch/agx-r5/waitidle.sh 5
python3 scripts/probe_partial_wg.py hw
ACTIVE="1 8" T="50 800" N=1140 python3 scripts/probe_partial_wg.py hw
ACTIVE="1" T="50 200 800" REGS=32 python3 scripts/probe_partial_wg.py hw
