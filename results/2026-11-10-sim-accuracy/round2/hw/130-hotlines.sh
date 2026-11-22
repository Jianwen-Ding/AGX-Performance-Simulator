# broadcast (hot) lines vs distinct lines, k_bin_bcast shape
cd /save/workspace/AGX-Performance-Simulator
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=5000
/scratch/agx-r5/waitidle.sh 5
python3 scripts/probe_hot_lines.py hw
ITER=1 N="512 2048" python3 scripts/probe_hot_lines.py hw
