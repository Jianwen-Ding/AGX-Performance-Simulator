#!/usr/bin/env python3
"""Offline check of the fragment interpolation model (`iterproj`) against a recorded hardware render.

Runs results/2026-11-14-frag-iter/iterproj_store.bin (Mesa's captured `iterproj r4_r5_r6_r7, cf1, cf0, center` followed by a
store of the four channels per lane) under `agx_run --frag`, with the coefficient planes fitted to the hardware image
(frag_w1.txt), and compares the 32 lanes (the 8x4 block at pixel (24, 28), quad layout) with the recorded pixels.
The planes come from the hardware output itself, so this verifies the plumbing and the plane evaluation (pass: <= 1e-6), not the
rasterizer's coefficient setup. Usage: frag_iter_check.py [agx_run]"""
import struct, subprocess, sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
d = root / 'results/2026-11-14-frag-iter'
run = sys.argv[1] if len(sys.argv) > 1 else str(root / 'rust/target/release/agx_run')
out = pathlib.Path('/tmp/frag_iter_out.bin')
r = subprocess.run([run, str(d / 'iterproj_store.bin'), '--grid', '32', '--wg', '32', '--frag', str(d / 'frag_w1.txt'), '--alloc', '0x1000000=1024',
                    '--uniform', '0=0x1000000', '--uniform', '1=0', '--dump', f'0x1000000=512:{out}'], capture_output=True, text=True)
if r.returncode:
    sys.exit(r.stdout + r.stderr)
sim = struct.unpack('<128f', out.read_bytes())
img = struct.unpack('<16384f', (d / 'hw_w1_64x64_rgba32f.bin').read_bytes())
worst = 0.0
for l in range(32):
    q = l // 4
    x, y = 24 + 2 * (q % 4) + (l & 1), 28 + 2 * (q // 4) + ((l >> 1) & 1)
    worst = max(worst, max(abs(a - b) for a, b in zip(img[(y * 64 + x) * 4:(y * 64 + x) * 4 + 4], sim[4 * l:4 * l + 4])))
print(f'iterproj, 32 lanes x 4 channels: max abs diff vs hardware {worst:.3g}')
sys.exit(0 if worst <= 1e-6 else 1)
