#!/bin/bash
# runsub.sh OUTDIR DISPATCH SETS [env...]: pp (if SETS has pp) + named held-out sets (n-NAME or r-NAME), J=2 parallel
o=$(realpath -m $1); disp=$2; sets=$3; shift 3; mkdir -p $o/heldout
cd /save/workspace/AGX-Performance-Simulator
C="--sim-arg=--launch-profile --sim-arg=runtime --dispatch $disp"
export AGX_THREADS=${AGX_THREADS:-3}
for s in $sets; do
  case $s in
    pp) echo "env $* nice -n 10 python3 scripts/capcheck.py /scratch/sim-captures/r5/llama-pp-fa0 /dev/null --hw-from /scratch/agx-r5/hw/pp-fa0-b2b5000.txt $C > $o/pp.txt 2>&1";;
    tg) echo "env $* nice -n 10 python3 scripts/capcheck.py /scratch/sim-captures/llama-tg /dev/null --hw-from /scratch/agx-r4/hw/llama-tg-b2b1000-r5.txt $C > $o/tg.txt 2>&1";;
    fa) echo "env $* nice -n 10 python3 scripts/capcheck.py /scratch/sim-captures/r5/llama-fa /dev/null --hw-from /scratch/agx-r5/hw/fa-b2b5000.txt $C > $o/fa.txt 2>&1";;
    tiny) echo "env $* nice -n 10 python3 scripts/capcheck.py /scratch/sim-captures/r5/tinygrad /dev/null --hw-from /scratch/agx-r5/hw/tinygrad-b2b5000.txt $C > $o/tiny.txt 2>&1";;
    r-*) d=${s#r-}; echo "env $* nice -n 10 python3 scripts/capcheck.py /scratch/sim-captures/rodinia/$d /scratch/sim-captures/rodinia/$d/profile.jsonl --hw-from /scratch/agx-r5/hwref/rodinia-all.txt $C > $o/heldout/$s.txt 2>&1";;
    n-*) d=${s#n-}; echo "env $* nice -n 10 python3 scripts/capcheck.py /scratch/sim-captures/r3/$d /scratch/sim-captures/r3/$d/profile.jsonl --hw-from /scratch/agx-r5/hwref/$d.txt $C > $o/heldout/$s.txt 2>&1";;
  esac
done | xargs -P ${J:-2} -I{} bash -c '{}'
