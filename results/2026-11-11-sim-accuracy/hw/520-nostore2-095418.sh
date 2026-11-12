# r9: Halide stencil_chain stage_5 and ZLUDA agx_sgemm_tile64 with their device stores replaced (hw only), next to the full kernels
export B2B=5000 REP=2 NOSIM=1
cd /scratch/agx-r5/r9/nostore/stencil5 && echo "== stencil5 nostore" && python3 /scratch/agx-r5/gsweep.py . 896,1784,1 --hw
cd /scratch/sim-captures/r3/h_stencil/371618-cap37-_kernel_stage_5_s0_y_y_block_id_y && echo "== stencil5 full" && python3 /scratch/agx-r5/gsweep.py . 896,1784,1 --hw
cd /scratch/agx-r5/r9/nostore/sgemm && echo "== sgemm nostore" && python3 /scratch/agx-r5/gsweep.py . 8192,1,1 --hw
cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap182-agx_sgemm_tile64 && echo "== sgemm full" && python3 /scratch/agx-r5/gsweep.py . 8192,1,1 --hw
