# store in a 64-bit register address (r24_r25) vs uniform + index, k_bin_bcast's row layout (kbb: register form +9.4 us, uniform form +3.2)
export B2B=5000 AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw
/scratch/agx-r5/waitidle.sh 5
cd /save/workspace/AGX-Performance-Simulator
for rw in 896 512; do for rs in 0 1; do echo "ROWW $rw REGSTORE $rs $(REGSTORE=$rs ROWW=$rw GRID2D=512 SERIAL=1 N=2048 BCAST=1 ITER=1 python3 scripts/probe_hot_lines.py hw)"; done; done
echo "1-D REGSTORE 1 ITER 1 $(REGSTORE=1 SERIAL=1 N=2048 BCAST=1 ITER=1 python3 scripts/probe_hot_lines.py hw)"
