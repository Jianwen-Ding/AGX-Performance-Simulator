# load after a store to a nearby address (k_bin_bcast's next trip is 2 KB on): ordering wait?
cd /save/workspace/AGX-Performance-Simulator
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=5000
/scratch/agx-r5/waitidle.sh 5
for st in 512 4096 65536; do echo "STRIDE $st"; STRIDE=$st SERIAL=1 N="512 2048" BCAST=1 python3 scripts/probe_hot_lines.py both; done
echo "ITER 1"; SERIAL=1 N="512 2048" BCAST=1 ITER=1 python3 scripts/probe_hot_lines.py both
