# k_bin_bcast ne512 (k1-early): store address in r28_r29 instead of r24_r25 (r24 = the src0 load's destination), regs 64; control: r24_r25 with regs 64
export B2B=5000 REP=2
/scratch/agx-r5/waitidle.sh 5
for d in r7/k6-r28 r7/k7-regs64; do echo "== $d"; cd /scratch/agx-r5/$d && python3 /scratch/agx-r5/gsweep.py . 512,512,1 --hw; done
