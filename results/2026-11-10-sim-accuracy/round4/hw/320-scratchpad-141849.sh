# Stack slowdown is address aliasing? (device twin: 6 MB per-workgroup stride slows at 24 cores, 384 KB packed does
# not). Pad the scratch buffer between cores (AGX_SCRATCH_PAD, zcopy/target-pad agx_run_hw) and rerun the stack probe
# and MMQ Q5_K at 16-34 busy cores.
export B2B=200 REP=2 NOSIM=1 HWBIN=/scratch/agx-r5/zcopy/target-pad/release/agx_run_hw
P=/scratch/agx-r5/r8/stackprobe
/scratch/agx-r5/waitidle.sh 5
for pad in 0 16384 65536 1048576 2097152; do
  echo "== stack A pad $pad"; cd $P/c-A; AGX_SCRATCH_PAD=$pad python3 /scratch/agx-r5/gsweep.py . 4096,1,1 6144,1,1 8192,1,1 --hw
done
cd /scratch/sim-captures/r5/llama-pp-fa0/311445-cap5293-*
for pad in 0 16384 1048576; do
  echo "== Q5_K pad $pad"; AGX_SCRATCH_PAD=$pad python3 /scratch/agx-r5/gsweep.py . 512,8,1 1024,8,1 1088,8,1 1216,8,1 --hw
done
