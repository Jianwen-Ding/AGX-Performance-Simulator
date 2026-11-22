# rms_norm pp (-27%) and soft_max pp (+17%): hw vs sim per grid size (waves)
export B2B=2000 REP=3
/scratch/agx-r5/waitidle.sh 5
cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap1186-* && echo "== rms_norm" && python3 /scratch/agx-r5/gsweep.py . 9728,1,1 19456,1,1 38912,1,1 77824,1,1 131072,1,1 --hw
cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap2418-* && echo "== soft_max" && B2B=300 python3 /scratch/agx-r5/gsweep.py . 19456,1,1 38912,1,1 194560,1,1 262144,1,1 262144,14,1 --hw
