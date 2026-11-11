#!/bin/bash
# evalsets.sh OUTDIR BIN [env...]: every reference set (held-out w/o Cycles, llama tg, pp512, fa, tinygrad) with agx_dispatch BIN
o=$(realpath -m $1); bin=$2; shift 2; mkdir -p $o/heldout
cd /save/workspace/AGX-Performance-Simulator
export AGX_THREADS=${AGX_THREADS:-2}
C="--sim-arg=--launch-profile --sim-arg=runtime --dispatch $bin --timeout 5400"
{
echo "/scratch/sim-captures/llama-tg /dev/null /scratch/agx-r4/hw/llama-tg-b2b1000-r5.txt $o/tg.txt"
echo "/scratch/sim-captures/r5/tinygrad /dev/null /scratch/agx-r5/hw/tinygrad-b2b5000.txt $o/tiny.txt"
echo "/scratch/sim-captures/r5/llama-fa /dev/null /scratch/agx-r5/hw/fa-b2b5000.txt $o/fa.txt"
echo "/scratch/sim-captures/r5/llama-pp-fa0 /dev/null /scratch/agx-r5/hw/pp-fa0-b2b5000.txt $o/pp.txt"
R=/scratch/sim-captures/rodinia; N=/scratch/sim-captures/r3
for d in $(ls $R); do [ -d $R/$d ] || continue; ls $R/$d | grep -q cap || continue; echo "$R/$d $R/$d/profile.jsonl /scratch/agx-r5/hwref/rodinia-all.txt $o/heldout/r-$d.txt"; done
for d in $(ls $N); do [ "$d" = cycles ] && continue; [ -f /scratch/agx-r5/hwref/$d.txt ] || continue; echo "$N/$d $N/$d/profile.jsonl /scratch/agx-r5/hwref/$d.txt $o/heldout/n-$d.txt"; done
} | ENVS="$*" C="$C" xargs -P ${J:-3} -L 1 bash -c 'env $ENVS nice -n 10 python3 scripts/capcheck.py $0 $1 --hw-from $2 $C > $3 2>&1'
python3 /scratch/agx-r5/r8/summ.py $o
