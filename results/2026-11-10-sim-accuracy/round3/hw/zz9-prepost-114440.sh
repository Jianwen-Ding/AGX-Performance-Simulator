# k_bin_bcast store: does ALU work before the loads (PRE) or after the store (POST) make the probe's store cost what kbb's does (+10 us / MB)?
export B2B=5000 AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw
/scratch/agx-r5/waitidle.sh 5
cd /save/workspace/AGX-Performance-Simulator
for v in "PRE=80" "POST=80" "PRE=200"; do for ns in 0 1; do echo "$v NOSTORE $ns $(env $v NOSTORE=$ns ROWW=896 GRID2D=512 SERIAL=1 N=2048 BCAST=1 ITER=1 python3 scripts/probe_hot_lines.py both)"; done; done
