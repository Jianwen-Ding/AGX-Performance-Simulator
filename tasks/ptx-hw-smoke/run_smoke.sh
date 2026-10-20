#!/bin/sh
# Runs the smoke kernels with RUNNER (agx_run_hw on the GPU, or the simulator's agx_run) and compares every output
# with expected/. Usage: RUNNER=/path/to/agx_run_hw sh run_smoke.sh [--record]
set -eu
cd "$(dirname "$0")"
: "${RUNNER:?set RUNNER to agx_run_hw or agx_run}"
mkdir -p inputs out expected
python3 -B - <<'PY'
import struct
open('inputs/a.bin','wb').write(struct.pack('<256f', *[float(i) for i in range(256)]))
open('inputs/b.bin','wb').write(struct.pack('<256f', *[1000.0 + 0.5 * i for i in range(256)]))
open('inputs/in.bin','wb').write(struct.pack('<256I', *[0x1000 + 3 * i for i in range(256)]))
PY
run() { name=$1; shift; echo "== $name"; "$RUNNER" "$@"; echo; }
run vadd vadd.bin --grid 256 --wg 64 --regs 16 \
  --map 0x100000=inputs/a.bin --map 0x200000=inputs/b.bin --alloc 0x300000=1024 \
  --uniform 0=0x100000 --uniform 1=0 --uniform 2=0x200000 --uniform 3=0 --uniform 4=0x300000 --uniform 5=0 \
  --uniform 6=4 --uniform 7=1 --uniform 8=1 --dump 0x300000=1024:out/vadd.bin
run ids2d ids2d.bin --grid 16,8 --wg 8,4 --regs 16 --alloc 0x100000=512 \
  --uniform 0=0x100000 --uniform 1=0 --uniform 2=16 --uniform 3=2 --uniform 4=2 --uniform 5=1 \
  --dump 0x100000=512:out/ids2d.bin
run tgrev tgrev.bin --grid 256 --wg 64 --regs 16 --shared 256 \
  --map 0x100000=inputs/in.bin --alloc 0x200000=1024 --alloc 0x300000=4 \
  --uniform 0=0x100000 --uniform 1=0 --uniform 2=0x200000 --uniform 3=0 --uniform 4=0x300000 --uniform 5=0 \
  --uniform 6=4 --uniform 7=1 --uniform 8=1 --dump 0x200000=1024:out/tgrev.bin --dump 0x300000=4:out/tgrev-count.bin
if [ "${1:-}" = --record ]; then cp out/*.bin expected/; echo recorded; exit 0; fi
fail=0
for f in expected/*.bin; do
  if cmp -s "$f" "out/$(basename "$f")"; then echo "PASS $(basename "$f")"; else echo "FAIL $(basename "$f")"; fail=1; fi
done
exit $fail
