# Stack footprint sweep (GPU-wide capacity below L1?): stackmix with 128 KB (mask 127) / 512 KB (512 threads) per core,
# with and without stores, at 8/16/24/32 busy cores (one workgroup each)
export B2B=200 REP=2 AGX_CORE_CYCLE_BUDGET=1 AGX_PLACED_BUDGET=1
P=/scratch/agx-r5/r8/stackprobe; cd $P
/scratch/agx-r5/waitidle.sh 5
cap() { n=$1 f=$2 k=$3 b=$4; shift 4
  rm -rf $P/cap-$n; mkdir -p $P/cap-$n
  ZLUDA_AGX_CAPTURE=$P/cap-$n /scratch/zluda-build/release/agx_bench_kernel $P/$f $k --grid 38 --block $b --iters 2 --warmup 0 "$@" > /dev/null
  ln -sfn $(ls -d $P/cap-$n/*$k) $P/c-$n; }
cap E stackmix.ptx stackmix 256 buf:38912 u32:3000 u32:127 u32:1
cap F stackmix.ptx stackmix 256 buf:38912 u32:3000 u32:127 u32:0
cap G stackmix.ptx stackmix 512 buf:77824 u32:2000 u32:255 u32:1
cap H stackmix.ptx stackmix 512 buf:77824 u32:2000 u32:255 u32:0
for v in E F; do echo "== $v"; cd $P/c-$v; python3 /scratch/agx-r5/gsweep.py . 2048,1,1 4096,1,1 6144,1,1 8192,1,1 --hw; done
for v in G H; do echo "== $v"; cd $P/c-$v; python3 /scratch/agx-r5/gsweep.py . 4096,1,1 8192,1,1 12288,1,1 16384,1,1 --hw; done
