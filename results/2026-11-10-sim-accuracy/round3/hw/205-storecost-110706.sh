# store increment: k_bin_bcast (ne512 one trip, full) with the store replaced by iadd r9,r9,0 vs the probe with/without store
export B2B=5000 REP=2 AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw
/scratch/agx-r5/waitidle.sh 5
for d in r7/ne512-nost r7/full-nost; do echo "== $d"; cd /scratch/agx-r5/$d && python3 /scratch/agx-r5/gsweep.py . 512,512,1 512,128,1 --hw; done
cd /save/workspace/AGX-Performance-Simulator
for ns in 0 1; do echo "NOSTORE $ns"; NOSTORE=$ns SERIAL=1 N="512 2048" BCAST=1 ITER=1 python3 scripts/probe_hot_lines.py both; NOSTORE=$ns SERIAL=1 N="2048" BCAST=1 ITER=2 python3 scripts/probe_hot_lines.py both; done
