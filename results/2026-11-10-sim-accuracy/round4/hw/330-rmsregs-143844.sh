# rms_norm pp (-27%; 512 workgroups of 8 groups, 4 resident by the register cap 36 at 88 halves): hw vs sim with
# other register counts (occupancy) at the full grid
export B2B=5000 REP=3
/scratch/agx-r5/waitidle.sh 5
cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap1186-*
for r in 88 48 64 128; do echo "== regs $r"; REGS=$r python3 /scratch/agx-r5/gsweep.py . 131072,1,1 --hw; done
