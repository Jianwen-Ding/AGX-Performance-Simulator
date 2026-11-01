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
# latency-bound chains (few groups, dependent loads to lines new in the command): the first-touch pointer chase
# (playground novel-results/2026-11-01-first-touch-pages, strides 0 / 64 / 16384) and ZLUDA's QR panel, Jacobi round,
# trsm block and split-gemv kernels (b2b medians, hwone.sh CAP --back-to-back 200, 2026-11-01)
lat=""; i=0; lhw=(8.792 23.666 24.208 514.7 7.50 188.0 23.2 62.5)
for d in /scratch/perf-goal/sim/chase/s0/* /scratch/perf-goal/sim/chase/s64/* /scratch/perf-goal/sim/chase/s16384/* \
         /scratch/perf-goal/sim/cap4/*qr_panel /scratch/perf-goal/sim/cap4/*jacobi_round_sg \
         /scratch/perf-goal/sim/cap4/*sgemv_split /scratch/perf-goal/sim/cap4/*sgemv_sum /scratch/perf-goal/sim/cap4/*strsm_sg; do
  s=$(/scratch/perf-goal/sim/simone.sh $d --launch-profile zluda | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['time_us'])")
  lat="$lat $(python3 -c "print(round(100*($s-${lhw[$i]})/${lhw[$i]}))")"; i=$((i+1))
done
python3 - "$new" "$lat" <<'PY'
import re,sys,statistics as st
errs=[]
for f in ('/tmp/claude-1000/ev_gb.txt','/tmp/claude-1000/ev_ll.txt'):
    for l in open(f):
        m=re.search(r'err\s+([+-]?\d+)%', l)
        if m: errs.append(abs(int(m.group(1))))
new=[abs(int(x)) for x in sys.argv[1].split()]
lat=[abs(int(x)) for x in sys.argv[2].split()]
print(f"40 captures: n={len(errs)} median {st.median(errs)} mean {st.mean(errs):.1f} within25 {sum(e<=25 for e in errs)} | new8 {sys.argv[1].strip()} mean {st.mean(new):.1f} | lat8 {sys.argv[2].strip()} mean {st.mean(lat):.1f} | all56 mean {st.mean(errs+new+lat):.1f}")
PY
