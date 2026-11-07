#!/bin/bash
# cap.sh NAME DIR SKIP CMD... : capture one launch per kernel of a real program through ZLUDA
name=$1; dir=$2; skip=$3; shift 3
out=/scratch/sim-captures/r3/$name; rm -rf $out; mkdir -p $out
cd $dir && env LD_LIBRARY_PATH=/scratch/cuda-clang-helpers/lib64:/scratch/zluda-build/release ZLUDA_AGX_ZOC=/scratch/zluda-build/release/zoc \
  ZLUDA_AGX_CAPTURE=$out ZLUDA_AGX_CAPTURE_SKIP=$skip ZLUDA_AGX_CAPTURE_MAX=1 ZLUDA_AGX_PROFILE=$out/profile.jsonl ZLUDA_AGX_LAUNCH_BATCH=0 \
  timeout 300 "$@" > $out/stdout.txt 2>&1
echo "$name exit $? launches $(wc -l < $out/profile.jsonl 2>/dev/null) caps $(ls -d $out/*-cap* 2>/dev/null | wc -l) $(du -sh $out | cut -f1)"
