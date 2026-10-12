#!/bin/sh
# Build Mesa's Asahi Vulkan driver (honeykrisp) with a no-op drm-shim, so the real AGX compiler runs on any Linux
# machine (no Apple GPU). vk_run then compiles and exports the driver's binaries exactly as on the M2; submissions
# are no-ops, so outputs and times are meaningless (spirv_task.py compile ignores them).
#
#   scripts/build_mesa_shim.sh [VERSION] [DIR]        default 26.0.3 into /home/user/mesa-src (or $MESA_SHIM_DIR)
#
# Prints the two environment variables that select the shimmed driver. Needs meson >= 1.4, ninja, LLVM, libclc,
# SPIRV-LLVM-Translator, libdrm, glslang, bison, flex, python3-mako/yaml/ply (Ubuntu 24.04: libclc-18-dev
# llvm-spirv-18 libllvmspirvlib-18-dev llvm-18-dev libclang-18-dev libdrm-dev glslang-tools; pip install meson).
#
# The source comes from the Ubuntu archive (orig tarball, same bytes as the Mesa release). The shim is patched to
# report the M2 Max as Asahi does (G14C B1: generation 14, variant C, revision 0x11, 38 cores) with the kernel's
# VM range, and to accept every ioctl the driver issues; the shim in Mesa <= 25.2 reports a G13G with a VM range too
# small for the driver's heaps.
set -eu
VER=${1:-26.0.3}
DIR=${2:-${MESA_SHIM_DIR:-/home/user/mesa-src}}
SRC=$DIR/mesa-$VER
mkdir -p "$DIR"
if [ ! -d "$SRC" ]; then
    curl -fsS -o "$DIR/mesa_$VER.orig.tar.xz" "http://archive.ubuntu.com/ubuntu/pool/main/m/mesa/mesa_$VER.orig.tar.xz"
    tar -C "$DIR" -xf "$DIR/mesa_$VER.orig.tar.xz"
fi
python3 - "$SRC/src/asahi/drm-shim/asahi_noop.c" <<'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
params = """static const struct drm_asahi_params_global params = {
   /* M2 Max as Asahi reports it (G14C B1); VM range as the kernel's */
   .gpu_generation = 14,
   .gpu_variant = 'C',
   .gpu_revision = 0x11,
   .chip_id = 0x6021,
   .num_dies = 1,
   .num_clusters_total = 4,
   .num_cores_per_cluster = 10,
   .max_frequency_khz = 1398000,
   .core_masks = {0x3ff, 0x3ff, 0x1ff, 0x1ff},
   .vm_start = 0x4000,
   .vm_end = 0x1000000000000ull,
   .command_timestamp_frequency_hz = 24000000,
};"""
s = re.sub(r'static const struct drm_asahi_params_global params = \{.*?\};', params, s, count=1, flags=re.S)
for ioctl in ('DRM_ASAHI_GET_TIME', 'DRM_ASAHI_GEM_BIND_OBJECT'):
    if f'[{ioctl}]' not in s:
        s = s.replace('   [DRM_ASAHI_SUBMIT] = asahi_ioctl_noop,', f'   [{ioctl}] = asahi_ioctl_noop,\n   [DRM_ASAHI_SUBMIT] = asahi_ioctl_noop,')
open(p, 'w').write(s)
EOF
if [ ! -f "$SRC/build/build.ninja" ]; then
    meson setup "$SRC/build" "$SRC" -Dbuildtype=release -Dgallium-drivers=asahi -Dvulkan-drivers=asahi -Dtools=drm-shim \
        -Dplatforms= -Dglx=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled -Dllvm=enabled -Dshared-llvm=enabled \
        -Dvalgrind=disabled -Dlibunwind=disabled >/dev/null
fi
ICD=src/asahi/vulkan/asahi_devenv_icd.$(uname -m).json
ninja -C "$SRC/build" src/asahi/vulkan/libvulkan_asahi.so src/asahi/drm-shim/libasahi_noop_drm_shim.so "$ICD" >/dev/null
echo "VK_ICD_FILENAMES=$SRC/build/$ICD"
echo "LD_PRELOAD=$SRC/build/src/asahi/drm-shim/libasahi_noop_drm_shim.so"
