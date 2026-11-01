#!/bin/bash
# evalset.sh: model error over the llama (11), Geekbench (29) and new stream/FFT (8) captures with the current env
S=/save/workspace/AGX-Performance-Simulator
cd /scratch/perf-goal && python3 $S/scripts/capcheck.py gbcap /scratch/geekbench-extract/runs/perf-gbcap-profile.jsonl --hw-from gbcap-check.txt --sim-arg=--launch-profile --sim-arg=zluda 2>/dev/null > /tmp/claude-1000/ev_gb.txt
cd /scratch/perf-goal/sim && python3 $S/scripts/capcheck.py cap2 capprof2.jsonl --hw-from $S/results/2026-11-01-llama-air-captures/capcheck-current-build.txt --sim-arg=--launch-profile --sim-arg=zluda 2>/dev/null > /tmp/claude-1000/ev_ll.txt
new=""; i=0; hw=(24.5 88.7 46.0 185.3 69.8 114.0 98.2 162.6)
for d in /scratch/perf-goal/gridy/capr/* /scratch/perf-goal/gridy/capc/* /scratch/perf-goal/fft/cap8/*shared*; do
  s=$(./simone.sh $d --launch-profile zluda | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['time_us'])")
  new="$new $(python3 -c "print(round(100*($s-${hw[$i]})/${hw[$i]}))")"; i=$((i+1))
done
python3 - "$new" <<'PY'
import re,sys,statistics as st
errs=[]
for f in ('/tmp/claude-1000/ev_gb.txt','/tmp/claude-1000/ev_ll.txt'):
    for l in open(f):
        m=re.search(r'err\s+([+-]?\d+)%', l)
        if m: errs.append(abs(int(m.group(1))))
new=[abs(int(x)) for x in sys.argv[1].split()]
print(f"40 captures: n={len(errs)} median {st.median(errs)} mean {st.mean(errs):.1f} within25 {sum(e<=25 for e in errs)} | new8 {sys.argv[1].strip()} mean {st.mean(new):.1f}")
PY
