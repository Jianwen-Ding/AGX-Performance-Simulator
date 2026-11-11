# one-resident placement beyond 34: static (i, i+34) vs dynamic (first free core); core id per workgroup
cd /save/workspace/AGX-Performance-Simulator
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw
for c in "38 0:0" "38 0:4" "38 30:34" "38 4:8" "34 0:0" "40 0:6" "72 0:4"; do set -- $c
  /scratch/agx-r5/waitidle.sh 5
  N=$1 LONG=$2 python3 scripts/probe_resident_order.py both
done
