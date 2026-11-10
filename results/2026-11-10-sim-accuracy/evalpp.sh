#!/bin/bash
# evalpp.sh OUTFILE [env...]: model vs hardware (b2b 5000) on the llama.cpp pp512 captures
out=$1; shift
cd /save/workspace/AGX-Performance-Simulator
env "$@" nice -n 10 python3 scripts/capcheck.py /scratch/sim-captures/r5/llama-pp-fa0 /dev/null --hw-from /scratch/agx-r5/hw/pp-fa0-b2b5000.txt --sim-arg=--launch-profile --sim-arg=runtime > $out 2>&1
grep " hw " $out | sed -E 's/^(\S{0,40})\S*/\1/; s/\(n.*\)//' | column -t
tail -1 $out
