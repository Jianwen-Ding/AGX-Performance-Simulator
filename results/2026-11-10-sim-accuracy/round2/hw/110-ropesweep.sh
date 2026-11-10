# rope_neox f32 / f16 (pp512) at 1, 3, 6, 30, 189 workgroups per core
export B2B=5000
for c in 2377 2386; do cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap$c-* && python3 /scratch/agx-r5/gsweep.py . 38,256,1 114,256,1 228,256,1 456,256,1 1140,256,1 7168,256,1 --hw; done
