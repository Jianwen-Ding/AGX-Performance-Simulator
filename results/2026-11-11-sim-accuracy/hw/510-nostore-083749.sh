# r9: pp kernels with their device stores replaced by iadd r9, r9, 0 (hw only), each next to its full kernel
export B2B=5000 REP=2 NOSIM=1
P=/scratch/sim-captures/r5/llama-pp-fa0/311445
for v in softmax:2418:262144,14,1 gated:2525:2490368,1,1 cpy:2436:458752,1,1 convert:1202:16384,1,1 rope:2377:7168,256,1 kbb:479:512,512,1; do
  IFS=: read n k g <<< "$v"
  echo "== $n nostore"; cd /scratch/agx-r5/r9/nostore/$n && python3 /scratch/agx-r5/gsweep.py . $g --hw
  echo "== $n full"; cd $(ls -d $P-cap$k-*) && python3 /scratch/agx-r5/gsweep.py . $g --hw
done
