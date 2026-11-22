# k_bin_bcast ne0=512 one-trip: real data vs zeroed src0 (m0) / src1 (m1) / both: data-dependent store cost?
export B2B=5000 REP=2
/scratch/agx-r5/waitidle.sh 5
for d in r6/kbb-ne512 r7/kbb-z0 r7/kbb-z1 r7/kbb-z01; do echo "== $d"; cd /scratch/agx-r5/$d && python3 /scratch/agx-r5/gsweep.py . 512,512,1 --hw; done
