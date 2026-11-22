# code fetch when every core is at a different place in a 48 / 192 KB body (MMQ stream-k workgroups) vs lockstep
cd /save/workspace/AGX-Performance-Simulator
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=1000
/scratch/agx-r5/waitidle.sh 5
for kb in 48:16 192:4; do for k in 0 4000 40000; do WGSTAGGER=1 G="1 8" K=$k T=${kb#*:} BODY_KB=${kb%:*} python3 scripts/probe_fetch_groups.py hw; done; done
