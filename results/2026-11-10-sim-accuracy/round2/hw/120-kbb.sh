# k_bin_bcast add in place (dst = src0) vs into another buffer
export B2B=5000
cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap479-* && python3 /scratch/agx-r5/gsweep.py . 512,128,1 512,512,1 --hw
cd /scratch/agx-r5/r6/kbb-sep && python3 /scratch/agx-r5/gsweep.py . 512,128,1 512,512,1 --hw
