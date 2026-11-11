# r9: rms_norm pp truncated copies (hw only) at 2 / 5 / 14 workgroups per core, plus the full kernel
export B2B=5000 REP=2 NOSIM=1
R=$(ls -d /scratch/sim-captures/r5/llama-pp-fa0/311445-cap1186-*)
for d in $(ls -d /scratch/agx-r5/r9/rmstrunc/* | sort -t t -k4 -n) $R; do
  echo "== $d"; cd $d && python3 /scratch/agx-r5/gsweep.py . 9728,1,1 38912,1,1 131072,1,1 --hw
done
