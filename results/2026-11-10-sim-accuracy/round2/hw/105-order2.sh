# placement: core id per workgroup, uniform work, various N; 1 and 3 resident; pairing check LONG 24:28
cd /save/workspace/AGX-Performance-Simulator
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=2000
/scratch/agx-r5/waitidle.sh 5
N=38 LONG=24:28 python3 scripts/probe_resident_order.py both
for n in 1 9 17 25 33 40 48 56 64 72 80 96 136 200; do N=$n LONG=0:0 python3 scripts/probe_resident_order.py hw; done
for n in 38 114 152 200; do N=$n LONG=0:0 SHARED=20000 python3 scripts/probe_resident_order.py hw; done
for n in 38 76 200; do N=$n LONG=0:0 SHARED=0 python3 scripts/probe_resident_order.py hw; done
