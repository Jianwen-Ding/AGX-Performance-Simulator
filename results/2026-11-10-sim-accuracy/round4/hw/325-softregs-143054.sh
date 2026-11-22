# soft_max pp (+17%, 1 resident 16-group workgroup per core by the register cap 28 at 116 halves): does the hw time
# change with the register count (cap 26 / 28 / 34 / more)? hw and sim
export B2B=300 REP=2
/scratch/agx-r5/waitidle.sh 5
cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap2418-*
for r in 116 128 96 64; do echo "== regs $r"; REGS=$r python3 /scratch/agx-r5/gsweep.py . 262144,14,1 --hw; done
