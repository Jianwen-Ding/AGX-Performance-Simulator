#!/bin/bash
# storetest.sh BIN ENV...: the pp store kernels (hw  model) under each env setting (columns)
cd /scratch/agx-r5/r9
for k in cap1186 cap479 cap2436 cap2525 cap2387 cap2418 cap2377 cap1202; do R=$(ls -d /scratch/sim-captures/r5/llama-pp-fa0/311445-$k-*); echo "$k $(grep -F "$(basename $R | cut -d- -f3- | cut -c1-30)" /scratch/agx-r5/hw/pp-fa0-b2b5000.txt | head -1 | grep -o 'hw *[0-9.]*') $(for e in "${@:2}"; do env $e AGX_FAR_SHARE=1 DISPATCH=$1 AGX_THREADS=2 python3 /scratch/agx-r5/simone.py $R 2>/dev/null | python3 -c 'import json,sys; print(round(json.load(sys.stdin)["time_us"],1))'; done | tr '\n' ' ')"; done
