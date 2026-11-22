#!/bin/bash
# evalsets.sh OUTDIR [env...]: every reference set (held-out w/o Cycles, llama tg, pp512, fa, tinygrad); no GPU
o=$1; shift; mkdir -p $o
cd /save/workspace/AGX-Performance-Simulator
C="--sim-arg=--launch-profile --sim-arg=runtime"
run() { env "$@" nice -n 10 python3 scripts/capcheck.py $D /dev/null --hw-from $H $C > $o/$N.txt 2>&1; }
D=/scratch/sim-captures/llama-tg H=/scratch/agx-r4/hw/llama-tg-b2b1000-r5.txt N=tg run "$@" &
D=/scratch/sim-captures/r5/tinygrad H=/scratch/agx-r5/hw/tinygrad-b2b5000.txt N=tiny run "$@" &
D=/scratch/sim-captures/r5/llama-fa H=/scratch/agx-r5/hw/fa-b2b5000.txt N=fa run "$@" &
wait
D=/scratch/sim-captures/r5/llama-pp-fa0 H=/scratch/agx-r5/hw/pp-fa0-b2b5000.txt N=pp run "$@"
EXCL=cycles J=4 bash /scratch/agx-r5/evalall.sh $o/heldout "$@" > $o/heldout.sum 2>&1
for f in tg tiny fa pp; do echo "$f $(tail -1 $o/$f.txt)"; done; cat $o/heldout.sum
