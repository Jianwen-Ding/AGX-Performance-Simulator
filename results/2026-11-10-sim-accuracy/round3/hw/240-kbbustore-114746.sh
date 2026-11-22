# k_bin_bcast ne512: store in the probe's form (u4_u5 + 32-bit index r26, unsigned) instead of r24_r25 64-bit; same output
export B2B=5000 REP=2
/scratch/agx-r5/waitidle.sh 5
for d in r7/k5-ustore r7/k1-early; do echo "== $d"; cd /scratch/agx-r5/$d && python3 /scratch/agx-r5/gsweep.py . 512,512,1 --hw; done
