#!/bin/sh
# One command on Asahi Linux (M2): record the GPU's results for every SPIR-V workload and verify the simulator
# against them. Writes spirv/expected/<workload>/ (hardware outputs, the driver's compiled code, timings) and
# results/<date>-asahi-verify/ (the verification table). Commit both afterwards.
#
#   scripts/asahi_verify.sh [--workloads a,b,...] [--no-large] [--coverage]
#
# --coverage also records the safe cases of the coverage catalog (scripts/coverage.py; ~30 min) and checks them.
set -eu
cd "$(dirname "$0")/.."
WORKLOADS=all; LARGE=""; COVERAGE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --workloads) WORKLOADS=$2; shift ;;
        --no-large) LARGE=--no-large ;;
        --coverage) COVERAGE=1 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
    shift
done

fail() { echo "preflight: $*" >&2; exit 1; }
echo "== preflight"
# In a container /proc/device-tree may be empty; the render node's own compatible string still names the AGX.
grep -qa apple /proc/device-tree/compatible 2>/dev/null || grep -qs 'apple,agx' /sys/class/drm/renderD128/device/uevent || fail "not an Apple machine running Asahi Linux"
[ -c /dev/dri/renderD128 ] || fail "no /dev/dri/renderD128 (AGX render node)"
case "${LD_PRELOAD:-}" in *drm*shim*) fail "LD_PRELOAD has a drm-shim: unset it to use the real GPU" ;; esac
for t in cc cargo python3 glslangValidator spirv-val; do
    command -v $t >/dev/null || fail "$t not found (dnf install gcc rust cargo python3-numpy glslang spirv-tools vulkan-headers vulkan-loader-devel)"
done
python3 -c 'import numpy' 2>/dev/null || fail "python3 numpy missing (dnf install python3-numpy)"
python3 scripts/gpu_safety.py status 2>/dev/null || true

echo "== build"
make -s rust
python3 -B scripts/spirv_task.py build >/dev/null

echo "== record (GPU)"
python3 -B scripts/spirv_task.py record "$WORKLOADS" $LARGE || echo "(some workloads did not match their references on the GPU: see the table)"

OUT=results/$(date -u +%Y-%m-%d)-asahi-verify
mkdir -p "$OUT"
echo "== verify (simulator vs the recordings)"
python3 -B scripts/spirv_task.py verify "$WORKLOADS" --timing --json "$OUT/verify.json" | tee "$OUT/verify.txt" || true
echo "== probes (what the recordings say about the cost model and the device)"
python3 -B scripts/spirv_task.py probes --json "$OUT/probes.json" | tee "$OUT/probes.txt" || true

echo "== bitfield shift question (70 safe ALU cases, ~1 min)"
python3 -B scripts/coverage.py record --area alu --match 'bfeil|bfi_|extr|shlhi|shrhi' || true
python3 -B scripts/bitfield_question.py | tee "$OUT/bitfield-question.txt" || true

if [ $COVERAGE = 1 ]; then
    echo "== coverage catalog (safe cases)"
    python3 -B scripts/coverage.py validate
    python3 -B scripts/coverage.py record
    python3 -B scripts/coverage.py check --timing | tee "$OUT/coverage-check.txt" || true
fi

{
    echo "# Asahi verification $(date -u +%Y-%m-%dT%H:%MZ)"
    echo
    echo "Device: $(python3 -c "import json;r=json.load(open('$OUT/verify.json'));print(next((x.get('device') for x in r if x.get('device')),'?'))")"
    echo "Kernel: $(uname -r)"
    echo
    echo '```'
    cat "$OUT/verify.txt"
    echo '```'
    echo
    echo '## Probes'
    echo
    echo '```'
    cat "$OUT/probes.txt"
    echo '```'
    echo
    echo '## bfeil shift question (applications notes 37-40)'
    echo
    echo '```'
    tail -1 "$OUT/bitfield-question.txt"
    echo '```'
    echo
    echo "PASS = simulator identical to the GPU word for word; CLOSE = differs, both within the reference tolerance;"
    echo "FAIL = a simulator (or GPU) bug to investigate; PENDING = not recorded."
} > "$OUT/README.md"
echo
echo "Done. Commit the recordings and the report:"
echo "  git add spirv/expected coverage/expected $OUT && git commit -m 'Asahi verification: SPIR-V workloads recorded on the M2'"
