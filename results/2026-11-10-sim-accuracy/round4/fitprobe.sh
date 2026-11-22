#!/bin/bash
# fitprobe.sh TD TS: sim stack probes A (256 KB/core), E (128 KB), C (tiny), twin d6m / dpk at 8/16/24/32 busy cores
cd /scratch/agx-r5/r8/stackprobe
for v in A E C d6m dpk; do (cd c-$v; echo "== $v $(AGX_L2_GPU=$1 AGX_L2_GPU_STACK=$2 python3 /scratch/agx-r5/gsweep.py . 2048,1,1 4096,1,1 6144,1,1 8192,1,1 | awk '{printf "%s ", $6}')") & done; wait
