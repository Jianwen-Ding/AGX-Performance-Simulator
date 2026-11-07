#!/bin/bash
# evalllama.sh OUTFILE [env...]: model vs hardware on the llama.cpp token-generation captures (no GPU)
out=$1; shift
cd /save/workspace/AGX-Performance-Simulator
env "$@" nice -n 10 python3 scripts/capcheck.py /scratch/sim-captures/llama-tg /dev/null --hw-from ${HWF:-/scratch/agx-r4/hw/llama-tg-b2b1000-r5.txt} --sim-arg=--launch-profile --sim-arg=zluda > $out 2>&1
grep " hw " $out | sed 's/ELi1ELb.*EEvPKvS2_[^ ]*//' | sed -E 's/^(\S{0,40})\S*/\1/; s/\(n.*\)//' | column -t
tail -1 $out
