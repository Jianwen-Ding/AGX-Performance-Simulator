#!/bin/bash
# evalall.sh OUTDIR [env assignments...]: like agx-r3/evalall.sh but at most $J (default 4) datasets at once
# (round 3 ran all ~30 in parallel and was OOM-killed)
out=$1; shift; mkdir -p $out
cd /save/workspace/AGX-Performance-Simulator
R=/scratch/sim-captures/rodinia; N=/scratch/sim-captures/r3
J=${J:-4}
jobs=()
for d in $(ls $R); do [ -d $R/$d ] || continue; ls $R/$d | grep -q cap || continue; jobs+=("$R/$d /scratch/agx-r5/hwref/rodinia-all.txt $out/r-$d.txt"); done
for d in $(ls $N); do [ -f /scratch/agx-r5/hwref/$d.txt ] || continue; jobs+=("$N/$d /scratch/agx-r5/hwref/$d.txt $out/n-$d.txt"); done
# EXCL=regex: skip datasets (EXCL=cycles: shade_surface alone simulates for ~10 min)
[ -n "$EXCL" ] && mapfile -t jobs < <(printf '%s\n' "${jobs[@]}" | grep -Ev "/($EXCL) ")
export ENVS="$*"
printf '%s\n' "${jobs[@]}" | xargs -P $J -L 1 bash -c 'env $ENVS nice -n 10 python3 scripts/capcheck.py $0 $0/profile.jsonl --hw-from $1 --sim-arg=--launch-profile --sim-arg=runtime > $2 2>&1'
cat $out/*.txt | grep " hw " | python3 -c "
import sys,re,statistics
e=[]
for l in sys.stdin:
  m=re.search(r'err\s+([+-]\d+)%',l)
  if m: e.append(abs(int(m.group(1))))
e.sort()
print('launches',len(e),'median',statistics.median(e),'p90',e[int(0.9*len(e))],'mean',round(sum(e)/len(e),1),'within10',sum(x<=10 for x in e),'within25',sum(x<=25 for x in e))
"
