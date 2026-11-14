#!/usr/bin/env python3
"""Offline checks of the tile / coverage instructions against recorded hardware renders (results/2026-11-14-frag-iter).
1. iterproj + st_tile u8norm (Mesa's colour write) over 32 lanes vs an RGBA8 render: bytes must match (clamp + round to nearest even).
2. Mesa's discard sequence (iterproj, fcmpsel x2, or, wait_pix, sample_mask) + st_tile over the whole 64x64 target vs a float
   render with `if (va.x > 0.5) discard`: a pixel is written iff the hardware drew it (triangle pixels only).
3. st_tile then ld_tile (u8norm) returns the stored bytes as fp32 (value / 255).
Coefficient planes are fitted from the hardware output (see frag_iter_check.py), so this verifies tile conversion, sample_mask
liveness and the plumbing, not the rasterizer's coefficient setup. Usage: frag_tile_check.py [agx_run]"""
import struct, subprocess, sys, pathlib, tempfile
root = pathlib.Path(__file__).resolve().parent.parent
d = root / 'results/2026-11-14-frag-iter'
run = sys.argv[1] if len(sys.argv) > 1 else str(root / 'rust/target/release/agx_run')
tmp = pathlib.Path(tempfile.mkdtemp())
def agx(*a):
    r = subprocess.run([run, *map(str, a)], capture_output=True, text=True)
    if r.returncode: sys.exit(r.stdout + r.stderr)
def tiles(f):
    t = {}
    for l in open(f):
        w = l.split()
        if w[0] == 'tile': t[(int(w[1]), int(w[2]))] = bytes.fromhex(w[4])
    return t
ok = True
# 1
agx(d / 'iterproj_st_tile_u8norm.bin', '--grid', 32, '--wg', 32, '--frag', d / 'frag_u8.txt', '--tile-dump', tmp / 't1')
hw = (d / 'hw_u8norm_64x64_rgba8.bin').read_bytes()
t = tiles(tmp / 't1')
worst = max(max(abs(a - b) for a, b in zip(v, hw[(y * 64 + x) * 4:(y * 64 + x) * 4 + 4])) for (x, y), v in t.items())
print(f'st_tile u8norm: {len(t)} pixels, max byte diff vs hardware {worst}'); ok &= worst == 0 and len(t) == 32
# 2
std = struct.unpack('<16384f', (d / 'hw_w1_64x64_rgba32f.bin').read_bytes())
hwd = struct.unpack('<16384f', (d / 'hw_discard_64x64_rgba32f.bin').read_bytes())
base = (d / 'frag_w1.txt').read_text().replace('group 0 24 28\n', '')
n = dis = 0
for part in range(4):
    (tmp / 'setup').write_text(base + 'samples 1\n' + ''.join(f'group {g} {8 * (g % 8)} {16 * part + 4 * (g // 8)}\n' for g in range(32)))
    agx(d / 'discard_sample_mask_st_tile.bin', '--grid', 1024, '--wg', 1024, '--frag', tmp / 'setup', '--uniform', '8=255', '--tile-dump', tmp / 't2')
    w = tiles(tmp / 't2')
    for y in range(16 * part, 16 * part + 16):
        for x in range(64):
            if std[(y * 64 + x) * 4] == -777.0: continue
            n += 1; dis += (hwd[(y * 64 + x) * 4] != -777.0) != ((x, y) in w)
print(f'discard / sample_mask: {n} triangle pixels, {dis} disagree with hardware'); ok &= dis == 0 and n > 1000
# 3
agx(d / 'st_ld_tile_roundtrip.bin', '--grid', 32, '--wg', 32, '--frag', d / 'frag_u8.txt', '--alloc', '0x1000000=1024', '--uniform', '0=0x1000000',
    '--uniform', '1=0', '--dump', f'0x1000000=512:{tmp / "rt"}')
out = struct.unpack('<128f', (tmp / 'rt').read_bytes())
rt = [round(v * 255) for v in out[:4]]
print('st_tile -> ld_tile lane 0 (bytes):', rt, '(hardware pixel (24,28): [140, 0, 71, 255])'); ok &= rt == [140, 0, 71, 255]
sys.exit(0 if ok else 1)
