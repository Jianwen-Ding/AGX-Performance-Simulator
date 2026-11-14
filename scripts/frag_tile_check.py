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
# 4. packed formats vs recorded renders (same shader, same planes): rgb10a2 (one 32-bit word per pixel), srgba8 (bytes)
for name, hwf in (('rgb10a2', 'hw_rgb10a2_64x64_word.bin'), ('srgba8', 'hw_srgba8_64x64_rgba8.bin')):
    agx(d / f'iterproj_st_tile_{name}.bin', '--grid', 32, '--wg', 32, '--frag', d / 'frag_u8.txt', '--tile-dump', tmp / 't4')
    hw = (d / hwf).read_bytes()
    t = tiles(tmp / 't4')
    def comps(b):
        if name == 'srgba8': return list(b)
        w = struct.unpack('<I', b)[0]
        return [w & 1023, w >> 10 & 1023, w >> 20 & 1023, w >> 30]
    worst = max(max(abs(a - c) for a, c in zip(comps(v), comps(hw[(y * 64 + x) * 4:(y * 64 + x) * 4 + 4]))) for (x, y), v in t.items())
    print(f'st_tile {name}: {len(t)} pixels, max component diff vs hardware {worst}'); ok &= worst == 0 and len(t) == 32
# 5. explicit coordinates: every lane stores to the tile pixel in the 16-bit register pair C (here (5, 3)), not to its own pixel
agx(d / 'st_tile_explicit_coords.bin', '--grid', 32, '--wg', 32, '--frag', d / 'frag_u8.txt', '--tile-dump', tmp / 't5')
t = tiles(tmp / 't5')
print('st_tile explicit coordinates: pixels written', sorted(t), 'value', list(t.get((5, 3), b'')))
ok &= sorted(t) == [(5, 3)] and list(t[(5, 3)]) == [255, 128, 0, 255]
# 6. 4 samples, per-pixel coverage: a store to all samples writes only the covered (live) ones; sr60 / sr124 read the same mask
agx(d / 'st_tile_all_samples.bin', '--grid', 32, '--wg', 32, '--frag', d / 'frag_msaa4.txt', '--tile-dump', tmp / 't6')
got = {}
for l in open(tmp / 't6'):
    w = l.split()
    if w[0] == 'tile': got.setdefault((int(w[1]), int(w[2])), []).append(int(w[3]))
print('st_tile all-samples, coverage 0b0101 / 0b1000 / 0 / all:', {k: v for k, v in got.items() if k in ((24, 28), (25, 28), (24, 29), (26, 28))})
ok &= got.get((24, 28)) == [0, 2] and got.get((25, 28)) == [3] and (24, 29) not in got and got.get((26, 28)) == [0, 1, 2, 3]
sys.exit(0 if ok else 1)
