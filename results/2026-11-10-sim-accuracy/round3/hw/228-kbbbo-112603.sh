# k_bin_bcast store cost vs buffer object size: dst in its 320 MB capture region vs a 1.75 MB region; probe OUT in 320 MB
export B2B=5000 REP=2 AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw
/scratch/agx-r5/waitidle.sh 5
for d in r7/small r7/smallall; do echo "== $d"; cd /scratch/agx-r5/$d && python3 /scratch/agx-r5/gsweep.py . 512,512,1 --hw; done
cd /save/workspace/AGX-Performance-Simulator
for mb in 0 320; do echo "OUTMB $mb"; OUTMB=$mb ROWW=896 GRID2D=512 SERIAL=1 N=2048 BCAST=1 ITER=1 python3 scripts/probe_hot_lines.py hw; done
