#!/bin/bash
# evalall.sh OUTDIR [env assignments...]: model vs stored hardware times on every dataset (no GPU)
out=$1; shift; mkdir -p $out
cd /save/workspace/AGX-Performance-Simulator
R=/scratch/sim-captures/rodinia; N=/scratch/sim-captures/r3
run() { env "$@" nice -n 10 python3 scripts/capcheck.py $P $P/profile.jsonl --hw-from $HWF --sim-arg=--launch-profile --sim-arg=runtime; }
for d in $(ls $R); do [ -d $R/$d ] || continue; ls $R/$d | grep -q cap || continue; P=$R/$d; HWF=/scratch/agx-r3/rodinia-hw-all.txt; run "$@" > $out/r-$d.txt 2>&1 & done
for d in $(ls $N); do [ -f /scratch/agx-r3/hw/$d.txt ] || continue; P=$N/$d; HWF=/scratch/agx-r3/hw/$d.txt; run "$@" > $out/n-$d.txt 2>&1 & done
wait
cat $out/*.txt | grep " hw " | python3 -c "
import sys,re,statistics
e=[]
for l in sys.stdin:
  m=re.search(r'err\s+([+-]\d+)%',l)
  if m: e.append(abs(int(m.group(1))))
print('launches',len(e),'median',statistics.median(e),'mean',round(sum(e)/len(e),1),'within10',sum(x<=10 for x in e),'within25',sum(x<=25 for x in e))
"
