# full k_bin_bcast (ne0 896, 2 trips) with the uniform-form store (u4_u5 + r26) vs its register form (k1-early-full)
export B2B=5000 REP=2
/scratch/agx-r5/waitidle.sh 5
for d in r7/k5-full r7/k1-early-full; do echo "== $d"; cd /scratch/agx-r5/$d && python3 /scratch/agx-r5/gsweep.py . 512,512,1 512,128,1 --hw; done
