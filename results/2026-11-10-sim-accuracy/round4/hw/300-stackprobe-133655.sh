# Stack traffic probe (r8/stackprobe/stackmix.ptx, MMQ's 1072 B stack): 3 lane-uniform stack loads (+1 store) per
# trip, 1 workgroup per core, 8/16/24/32/38 busy cores. Does per-workgroup time grow with busy cores like MMQ's?
export B2B=200 REP=2 AGX_CORE_CYCLE_BUDGET=1 AGX_PLACED_BUDGET=1
P=/scratch/agx-r5/r8/stackprobe; cd $P
/scratch/agx-r5/waitidle.sh 5
cap() { # name block mask st iters
  rm -rf $P/cap-$1; mkdir -p $P/cap-$1
  ZLUDA_AGX_CAPTURE=$P/cap-$1 /scratch/zluda-build/release/agx_bench_kernel $P/stackmix.ptx stackmix --grid 38 --block $2 --iters 2 --warmup 0 buf:$((38*$2*4)) u32:$5 u32:$3 u32:$4
  ln -sfn $(ls -d $P/cap-$1/*stackmix) $P/c-$1
}
cap A 256 255 1 3000
cap B 256 255 0 3000
cap C 256 15 1 3000
cap D 1024 255 1 1000
for v in A B C; do echo "== $v"; cd $P/c-$v && python3 /scratch/agx-r5/gsweep.py . 2048,1,1 4096,1,1 6144,1,1 8192,1,1 9728,1,1 --hw; done
echo "== D"; cd $P/c-D && python3 /scratch/agx-r5/gsweep.py . 8192,1,1 16384,1,1 24576,1,1 32768,1,1 38912,1,1 --hw
