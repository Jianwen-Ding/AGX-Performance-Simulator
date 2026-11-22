# MMQ Q6_K stream-k at 38 / 34 / 17 / 8 workgroups: is the per-workgroup time short in the model, or the 38-grid pairing?
export B2B=200 REP=2
/scratch/agx-r5/waitidle.sh 5
cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap4582-* && python3 /scratch/agx-r5/gsweep.py . 1216,8,1 1088,8,1 544,8,1 256,8,1 --hw
