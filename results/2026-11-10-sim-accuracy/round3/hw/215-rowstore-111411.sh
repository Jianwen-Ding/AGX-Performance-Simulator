# k_bin_bcast's store address pattern: 512 rows, 512 threads (4 workgroups) per row, rows ROWW words apart
export B2B=5000 AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw
/scratch/agx-r5/waitidle.sh 5
cd /save/workspace/AGX-Performance-Simulator
for rw in 512 896 1024 640; do for ns in 0 1; do echo "ROWW $rw NOSTORE $ns"; ROWW=$rw GRID2D=512 NOSTORE=$ns SERIAL=1 N=2048 BCAST=1 ITER=1 python3 scripts/probe_hot_lines.py both; done; done
