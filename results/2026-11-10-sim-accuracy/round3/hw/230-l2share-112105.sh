# per-core vs shared L2: each workgroup loops over its own REGKB region, N workgroups (one per core)
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=500
/scratch/agx-r5/waitidle.sh 5
cd /save/workspace/AGX-Performance-Simulator
T=2000 N="1 4 8 16 32" REGKB="16 64 256 512 2048" python3 scripts/probe_l2_share.py both
