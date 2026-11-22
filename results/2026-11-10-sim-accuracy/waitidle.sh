#!/bin/bash
# waitidle.sh [secs]: wait until no other process has the GPU open for $1 (default 20) consecutive seconds
need=${1:-20}; [ $need -gt 6 ] && need=6; ok=0
while [ $ok -lt $need ]; do
  if [ -z "$(/scratch/agx-r5/gpuusers.sh $$ $PPID)" ]; then ok=$((ok+2)); else ok=0; fi
  sleep 2
done
