#!/bin/bash
# cap.sh NAME DIR SKIP CMD... : capture one launch per kernel of a real program through the CUDA translation layer
name=$1; dir=$2; skip=$3; shift 3
out=/scratch/sim-captures/r3/$name; rm -rf $out; mkdir -p $out
cd $dir && env LD_LIBRARY_PATH=/scratch/cuda-clang-helpers/lib64:/scratch/ptx-frontend-build/release AGX_RUNTIME_ZOC=/scratch/ptx-frontend-build/release/zoc \
  AGX_RUNTIME_CAPTURE=$out AGX_RUNTIME_CAPTURE_SKIP=$skip AGX_RUNTIME_CAPTURE_MAX=1 AGX_RUNTIME_PROFILE=$out/profile.jsonl AGX_RUNTIME_LAUNCH_BATCH=0 \
  timeout 300 "$@" > $out/stdout.txt 2>&1
echo "$name exit $? launches $(wc -l < $out/profile.jsonl 2>/dev/null) caps $(ls -d $out/*-cap* 2>/dev/null | wc -l) $(du -sh $out | cut -f1)"
