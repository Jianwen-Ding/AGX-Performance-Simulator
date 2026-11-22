# store with a per-lane 64-bit register address (k_bin_bcast) vs uniform base + index
cd /save/workspace/AGX-Performance-Simulator
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=5000
/scratch/agx-r5/waitidle.sh 5
for rs in 0 1; do echo "REGSTORE $rs"; REGSTORE=$rs SERIAL=1 N="512 2048" BCAST=1 python3 scripts/probe_hot_lines.py both; done
