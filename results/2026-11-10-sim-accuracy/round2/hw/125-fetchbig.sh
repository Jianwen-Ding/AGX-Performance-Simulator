# straight-line refetch cost for bodies beyond 64 KB (MMQ runs 175-225 KB of code)
cd /save/workspace/AGX-Performance-Simulator
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=1000
/scratch/agx-r5/waitidle.sh 5
for kb in 64 96 128 192 256 384; do
  T=$(python3 -c "print(max(2,int(16*48/$kb)))")
  echo "BODY_KB $kb"; G="1 8" K=0 T=$T BODY_KB=$kb python3 scripts/probe_fetch_groups.py hw
done
