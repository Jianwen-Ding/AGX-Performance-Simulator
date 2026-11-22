# k_bin_bcast store mutations: k1 store moved up 5 instructions; k4 store to row 0 only (u4_u5 + i0: 2 KB for all rows)
export B2B=5000 REP=2
/scratch/agx-r5/waitidle.sh 5
for d in r7/k1-early r7/k4-row0 r7/k1-early-full; do echo "== $d"; cd /scratch/agx-r5/$d && python3 /scratch/agx-r5/gsweep.py . 512,512,1 --hw; done
