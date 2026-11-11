# Stack probe follow-up: which workgroups are slow when clusters 2-3 are busy (active range [lo,hi) at grid 32), and
# does a device-memory twin with the scratch layout (6 MB per workgroup, 48 KB per group) slow the same way?
export B2B=200 REP=2 AGX_CORE_CYCLE_BUDGET=1 AGX_PLACED_BUDGET=1
P=/scratch/agx-r5/r8/stackprobe; cd $P
/scratch/agx-r5/waitidle.sh 5
cap() { # name ptx kernel block args...
  n=$1 f=$2 k=$3 b=$4; shift 4
  rm -rf $P/cap-$n; mkdir -p $P/cap-$n
  ZLUDA_AGX_CAPTURE=$P/cap-$n /scratch/zluda-build/release/agx_bench_kernel $P/$f $k --grid 38 --block $b --iters 2 --warmup 0 "$@" > /dev/null
  ln -sfn $(ls -d $P/cap-$n/*$k) $P/c-$n
}
sw() { cd $P/c-$1; shift; python3 /scratch/agx-r5/gsweep.py . "$@" --hw; }
for r in 0:8 8:16 16:24 24:32 0:16 16:32 0:32; do
  lo=${r%:*} hi=${r#*:}
  cap r$lo-$hi stackrng.ptx stackrng 256 buf:38912 u32:3000 u32:255 u32:1 u32:$lo u32:$hi
  echo "== stack range $r, grid 32"; sw r$lo-$hi 8192,1,1
done
echo "== stack range 16:24 at grid 24"; sw r16-24 6144,1,1
B=$((40*6291456))
cap d6m devmix.ptx devmix 256 buf:$B u32:3000 u32:255 u32:1 u32:6291456 u32:49152 u32:0 u32:64
echo "== dev twin 6MB/48KB"; sw d6m 2048,1,1 4096,1,1 6144,1,1 8192,1,1 9728,1,1
cap dpk devmix.ptx devmix 256 buf:$B u32:3000 u32:255 u32:1 u32:393216 u32:49152 u32:0 u32:64
echo "== dev twin packed 384KB/48KB"; sw dpk 2048,1,1 4096,1,1 6144,1,1 8192,1,1 9728,1,1
cap d6r devmix.ptx devmix 256 buf:$B u32:3000 u32:255 u32:1 u32:6291456 u32:49152 u32:16 u32:24
echo "== dev twin 6MB range 16:24 at grid 32"; sw d6r 8192,1,1
