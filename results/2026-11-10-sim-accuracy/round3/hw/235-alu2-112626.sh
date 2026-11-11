# ALU throughput at many groups: operand count / registers (a fadd 2 regs, j iadd 2 regs, g fmadd r29 twice, k fmadd one reg)
export AGX_RUN_HW=/scratch/agx-r5/zcopy/target/release/agx_run_hw B2B=200
/scratch/agx-r5/waitidle.sh 5
cd /save/workspace/AGX-Performance-Simulator
NG="4 16 32" T1=1000 T2=3000 python3 scripts/probe_tg_alu.py both aaaaaaaa jjjjjjjj gggggggg kkkkkkkk iiii iiiiiiiiiiiiiiii
