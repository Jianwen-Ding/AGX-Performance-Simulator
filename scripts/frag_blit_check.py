#!/usr/bin/env python3
"""Replay of Mesa's compute blit program (the shader glBlitFramebuffer runs) against recorded hardware blits (results/2026-11-14-frag-iter/blit).
The program uses explicit-coordinate `st_tile` (u8norm, tile-local coordinates from the thread position), a threadgroup barrier, `texture_sample` and
`image_write_block` (lane (0, 0) of each 32 x 32 workgroup stores the whole tile block at the workgroup's image position, clipped to the image).
The driver's state is rebuilt by hand: root table (4 scale / offset floats, destination origin), the source texture descriptor (ts0), a sampler (ss1) and the
destination PBE descriptor (ts2). Pass: every byte of every blit equals the hardware's. Usage: frag_blit_check.py [agx_run]"""
import struct, subprocess, sys, pathlib, tempfile
root = pathlib.Path(__file__).resolve().parent.parent
d = root / 'results/2026-11-14-frag-iter/blit'
run = sys.argv[1] if len(sys.argv) > 1 else str(root / 'rust/target/release/agx_run')
tmp = pathlib.Path(tempfile.mkdtemp())
def put(b, start, size, v):
    for i in range(size):
        if (v >> i) & 1: b[(start + i) // 8] |= 1 << ((start + i) % 8)
def tex_desc(dim, ch, ty, sw, w, h, addr, pitch):
    b = bytearray(24); put(b, 0, 4, dim); put(b, 6, 7, ch); put(b, 13, 3, ty)
    for i, s in enumerate(sw): put(b, 16 + 3 * i, 3, s)
    put(b, 28, 14, w - 1); put(b, 42, 14, h - 1); put(b, 66, 36, addr >> 4); put(b, 110, 18, (pitch - 16) >> 4); return bytes(b)
def pbe(ch, ty, w, h, addr, stride):
    b = bytearray(24); put(b, 6, 7, ch); put(b, 13, 3, ty); put(b, 24, 14, w - 1); put(b, 38, 14, h - 1); put(b, 64, 36, addr >> 4); put(b, 104, 21, stride - 16); return bytes(b)
def samp(filt, wrap, pixel):
    b = bytearray(8); put(b, 23, 2, filt); put(b, 25, 2, filt)
    for i in range(3): put(b, 29 + 3 * i, 3, wrap)
    put(b, 38, 1, pixel); return bytes(b)
ROOT, SRC, DST = 0x1000000, 0x2000000, 0x3000000
ok = True
for name, (sw, sh, dw, dh, lin) in {'64_64_32_32_0': (64, 64, 32, 32, 0), '40_72_36_44_0': (40, 72, 36, 44, 0), '64_64_48_40_1': (64, 64, 48, 40, 1),
                                    '48_48_100_60_0': (48, 48, 100, 60, 0), '48_48_100_60_1': (48, 48, 100, 60, 1)}.items():
    (tmp / 'root').write_bytes(struct.pack('<4f', 0.5 / dw, 0.5 / dh, 1.0 / dw, 1.0 / dh) + struct.pack('<2I', 0, 0))
    (tmp / 'setup').write_text('size 64 64\nsamples 1\ntile 32 32\n')
    r = subprocess.run([run, str(d / 'blit_main.bin'), '--grid', f'{(dw + 31) // 32 * 32},{(dh + 31) // 32 * 32}', '--wg', '32,32', '--preamble', str(d / 'blit_preamble.bin'),
                        '--frag', str(tmp / 'setup'), '--map', f'{ROOT:#x}={tmp / "root"}', '--map', f'{SRC:#x}={d / f"src_{name}.bin"}', '--alloc', f'{DST:#x}={dw * dh * 4}',
                        '--uniform', f'0={ROOT}', '--uniform', '1=0', '--tex-state', '0=' + tex_desc(2, 0x28, 0, [0, 1, 2, 3], sw, sh, SRC, sw * 4).hex(),
                        '--samp-state', '1=' + samp(lin, 0, 0).hex(), '--tex-state', '2=' + pbe(0x28, 0, dw, dh, DST, dw * 4).hex(), '--dump', f'{DST:#x}={dw * dh * 4}:{tmp / "out"}'],
                       capture_output=True, text=True)
    if r.returncode: sys.exit(r.stdout + r.stderr)
    sim, hw = (tmp / 'out').read_bytes(), (d / f'hw_dst_{name}.bin').read_bytes()
    diff = sum(a != b for a, b in zip(sim, hw))
    print(f'blit {sw}x{sh} -> {dw}x{dh} ({"linear" if lin else "nearest"}): {diff} of {len(hw)} bytes differ from hardware'); ok &= diff == 0 and len(sim) == len(hw)
sys.exit(0 if ok else 1)
