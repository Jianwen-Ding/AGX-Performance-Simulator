# k_bin_bcast with ne0 = 512 (one trip per thread) and 1024 (two full trips) vs 896
export B2B=5000 REP=2
for d in kbb-ne512 kbb-ne1024; do echo "== $d"; cd /scratch/agx-r5/r6/$d && python3 /scratch/agx-r5/gsweep.py . 512,128,1 512,512,1 --hw; done
