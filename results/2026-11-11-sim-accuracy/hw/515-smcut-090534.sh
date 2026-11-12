# r9: soft_max pp truncated copies (hw only) at 14 and 1 heads, plus the no-store copy at 1 head
export B2B=5000 REP=2 NOSIM=1
for d in $(ls -d /scratch/agx-r5/r9/smtrunc/* | sort -t t -k4 -n) /scratch/agx-r5/r9/nostore/softmax; do
  echo "== $d"; cd $d && python3 /scratch/agx-r5/gsweep.py . 262144,14,1 262144,1,1 --hw
done
