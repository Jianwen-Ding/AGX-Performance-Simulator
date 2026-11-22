#!/bin/bash
# stackfit.sh "VARIANTS" BIN [env...]: sim of round-4 stack probes at 8/16/24/32 busy cores (hw below)
vs=$1; bin=$2; shift 2; cd /scratch/agx-r5/r8/stackprobe
for v in $vs; do w=$(grep -o -- "--wg [0-9]*" c-$v/replay.sh | awk '{print $2}'); (cd c-$v; echo "$v $(env "$@" AGX_THREADS=1 DISP=$bin python3 /scratch/agx-r5/gsweep.py . $((8*w)),1,1 $((16*w)),1,1 $((24*w)),1,1 $((32*w)),1,1 | awk '{printf "%s ", $6}')") & done; wait
# hw: A 737.6 743.0 1045.1 1183.6 | B 648.5 653.5 659.3 953.8 | E 737.2 742.8 749.4 866.5 | D 472.9 827.7 1211.9 1586.2 | G 540.5 824.7 1071.2 1545.4 | H 475.4 655.1 819.7 1085.1
