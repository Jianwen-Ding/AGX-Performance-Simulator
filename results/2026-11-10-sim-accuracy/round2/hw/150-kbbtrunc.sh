# k_bin_bcast truncated after index arithmetic / first load / second load / before store
export B2B=5000 REP=2
for d in $(ls -d /scratch/agx-r5/r6/ktrunc/* | sort -t t -k3 -n); do
  echo "== $d"; cd $d && python3 /scratch/agx-r5/gsweep.py . 512,128,1 512,512,1 --hw
done
