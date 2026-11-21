#!/usr/bin/env python3
"""Scene files for tests/gles_scene.c: the render-model test suite (scripts/render_check.py suite).

    python3 -I scripts/render_scene.py OUTDIR [--seed N]   # writes OUTDIR/*.txt and OUTDIR/suite.json
Families: random (random triangles, several draws), topleft (edges through pixel centers, shared edges, vertices on
corners/centers), subpixel (vertices offset by 2^-k pixel), cull (mode x front face x winding), depth (functions, write
masks, overlap), stencil, scissor, prims (strip / fan / indexed / restart / instanced), flat, discard, fragz, blend,
persp (w != 1), clip (vertices behind the eye / off screen), big (256x256 many triangles).
"""
import json, random, sys
from pathlib import Path


def px2ndc(x, n):
    return 2.0 * x / n - 1.0


def tri_px(W, H, pts, z=0.5, w=1.0, cols=None):
    cols = cols or [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    return [f'v {px2ndc(x, W) * w:.9g} {px2ndc(y, H) * w:.9g} {z * w if not isinstance(z, list) else z[i] * w:.9g} {w:.9g} '
            + ' '.join(f'{c:.6g}' for c in cols[i]) for i, (x, y) in enumerate(pts)]


def scene(W, H, body, **kw):
    lines = [f'size {W} {H}'] + [f'{k} {v}' for k, v in kw.items()] + body
    return '\n'.join(lines) + '\n'


def gen(seed=1):
    R = random.Random(seed)
    S = {}
    col = lambda: tuple(round(R.random(), 3) for _ in range(3)) + (1,)
    # random triangles
    for k in range(6):
        W, H = R.choice([(64, 64), (100, 70), (128, 96), (33, 47)])
        body, n = [], 0
        for d in range(R.randint(1, 4)):
            m = R.randint(1, 6)
            for _ in range(m):
                pts = [(R.uniform(-8, W + 8), R.uniform(-8, H + 8)) for _ in range(3)]
                body += tri_px(W, H, pts, cols=[col(), col(), col()])
            body.append(f'draw {n} {3 * m}'); n += 3 * m
        S[f'random{k}'] = scene(W, H, body, clear='0.1 0.2 0.3 1')
    # top-left rule: a square split along the diagonal, a fan around a center, edges through pixel centers
    for k, (a, b) in enumerate([(8, 40), (8.5, 40.5), (0, 64), (16.5, 16.5)]):
        body = tri_px(64, 64, [(a, a), (b, a), (a, b)], cols=[(1, 0, 0, .5)] * 3) + tri_px(64, 64, [(b, a), (b, b), (a, b)], cols=[(0, 1, 0, .5)] * 3)
        body.append('draw 0 6')
        S[f'topleft{k}'] = scene(64, 64, body, blend='add', clear='0 0 0 0')
    c = (32.5, 32.5); ring = [(32.5 + 20 * __import__('math').cos(t * 0.5236), 32.5 + 20 * __import__('math').sin(t * 0.5236)) for t in range(13)]
    body = []
    for i in range(12):
        body += tri_px(64, 64, [c, ring[i], ring[i + 1]], cols=[(0.25, 0, 0, 1)] * 3)
    body.append('draw 0 36')
    S['topleft_fan'] = scene(64, 64, body, blend='add', clear='0 0 0 0')
    # sub-pixel snapping: thin slivers whose vertices differ by 2^-k pixel
    for k in (6, 8, 9, 10, 12):
        e = 2.0 ** -k
        body = []
        for i in range(8):
            x0 = 4 + 7 * i + 0.5
            body += tri_px(64, 64, [(x0, 2.0), (x0 + e * (i + 1), 60.0), (x0 + 3.0, 31.5 + e * i)], cols=[(1, 1, 1, 1)] * 3)
        body.append('draw 0 24')
        S[f'subpixel{k}'] = scene(64, 64, body)
    # cull
    cw = tri_px(64, 64, [(10, 10), (30, 50), (50, 10)], cols=[(1, 0, 0, 1)] * 3)   # clockwise in GL window (y up)
    ccw = tri_px(64, 64, [(14, 54), (54, 54), (34, 14)], cols=[(0, 1, 0, 1)] * 3)
    for mode in ('none', 'back', 'front', 'both'):
        for ff in ('ccw', 'cw'):
            S[f'cull_{mode}_{ff}'] = scene(64, 64, cw + ccw + ['draw 0 6'], cull=mode, frontface=ff)
    # depth
    over = (tri_px(64, 64, [(4, 4), (60, 8), (20, 60)], z=[0.2, 0.8, 0.5], cols=[(1, 0, 0, 1)] * 3)
            + tri_px(64, 64, [(60, 60), (4, 40), (50, 2)], z=[0.3, 0.6, 0.9], cols=[(0, 0, 1, 1)] * 3) + ['draw 0 3', 'draw 3 3'])
    for f in ('less', 'lequal', 'greater', 'gequal', 'equal', 'notequal', 'always', 'never'):
        S[f'depth_{f}'] = scene(64, 64, over, depth=f, cleardepth='1.0' if f in ('less', 'lequal', 'equal', 'notequal', 'always', 'never') else '0.0')
    S['depth_nowrite'] = scene(64, 64, over, depth='less', depthwrite='0')
    S['depth_clear05'] = scene(64, 64, over, depth='less', cleardepth='0.5')
    # stencil: first draw replaces 1, second only where equal 1 / notequal
    for nm, sv in [('replace_eq', 'equal 1 0xff 0xff keep keep replace'), ('incr', 'always 0 0xff 0xff keep keep incr'),
                   ('invert_mask', 'always 0 0xff 0x0f keep keep invert'), ('fail_zero', 'less 2 0xff 0xff zero keep keep')]:
        S[f'stencil_{nm}'] = scene(64, 64, over, stencil=sv)
    # scissor
    big = tri_px(64, 64, [(-10, -10), (100, -10), (-10, 100)], cols=[(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]) + ['draw 0 3']
    for k, sc in enumerate(['10 12 20 30', '0 0 64 64', '33 1 7 61', '31 31 2 2']):
        S[f'scissor{k}'] = scene(64, 64, big, scissor=sc)
    # primitive topologies
    strip = []
    for i in range(8):
        strip += tri_px(64, 64, [(4 + 7 * i, 10 + 40 * (i % 2))], cols=[col()])
    S['prim_strip'] = scene(64, 64, strip + ['draw 0 8'], prim='strip')
    S['prim_fan'] = scene(64, 64, tri_px(64, 64, [c] + ring[:8], cols=[col() for _ in range(9)]) + ['draw 0 9'], prim='fan')
    S['prim_indexed'] = scene(64, 64, strip + ['i 0', 'i 1', 'i 2', 'i 2', 'i 1', 'i 3', 'i 5', 'i 4', 'i 6', 'draw 0 9'], indexed=1)
    S['prim_restart'] = scene(64, 64, strip + ['i 0', 'i 1', 'i 2', 'i 3', 'i 65535', 'i 4', 'i 5', 'i 6', 'i 7', 'draw 0 9'], indexed=1, restart=1, prim='strip')
    S['prim_instanced'] = scene(64, 64, tri_px(64, 64, [(4, 4), (20, 4), (4, 30)], cols=[col(), col(), col()]) + ['draw 0 3 5'])
    S['prim_basevertex'] = scene(64, 64, strip + ['draw 2 5'], prim='strip')
    # shading
    S['flat'] = scene(64, 64, tri_px(64, 64, [(3, 3), (60, 20), (20, 61)], cols=[col(), col(), col()]) + ['draw 0 3'], shader='flat')
    S['discard'] = scene(64, 64, big, shader='discard')
    S['fragz'] = scene(64, 64, over, shader='fragz', depth='less')
    S['blend_alpha'] = scene(64, 64, tri_px(64, 64, [(4, 4), (60, 8), (20, 60)], cols=[(1, 0, 0, .5), (0, 1, 0, .25), (0, 0, 1, .75)])
                             + tri_px(64, 64, [(60, 60), (4, 40), (50, 2)], cols=[(1, 1, 0, .5)] * 3) + ['draw 0 6'], blend='alpha', clear='0.2 0.2 0.2 1')
    S['const'] = scene(64, 64, over, shader='const')
    # perspective-correct varyings: w differs per vertex
    body = []
    for i, w in enumerate([(1, 3, 0.5), (2, 2, 7), (0.25, 1, 4)]):
        pts = [(5 + 20 * i, 5), (22 + 20 * i, 30), (5 + 20 * i, 60)]
        for (x, y), ww, cc in zip(pts, w, [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]):
            body.append(f'v {px2ndc(x, 64) * ww:.9g} {px2ndc(y, 64) * ww:.9g} {0.5 * ww:.9g} {ww:.9g} ' + ' '.join(map(str, cc)))
    S['persp'] = scene(64, 64, body + ['draw 0 9'])
    # clipping: a vertex behind the eye (w < 0) and far off screen
    S['clip_offscreen'] = scene(64, 64, tri_px(64, 64, [(-500, -300), (900, 32), (20, 700)], cols=[col(), col(), col()]) + ['draw 0 3'])
    S['clip_near'] = scene(64, 64, ['v -0.8 -0.8 0.5 1 1 0 0 1', 'v 0.8 -0.6 0.5 1 0 1 0 1', 'v 0.2 0.5 -1.5 -0.5 0 0 1 1', 'draw 0 3'])
    # big
    body = []
    for i in range(400):
        x, y = R.uniform(0, 256), R.uniform(0, 256)
        body += tri_px(256, 256, [(x, y), (x + R.uniform(-20, 20), y + R.uniform(-20, 20)), (x + R.uniform(-20, 20), y + R.uniform(-20, 20))],
                       z=[R.random() for _ in range(3)], cols=[col(), col(), col()])
    S['big_depth'] = scene(256, 256, body + ['draw 0 1200'], depth='less')
    S['big_blend'] = scene(256, 256, body + ['draw 0 1200'], blend='add', clear='0 0 0 0')
    return S


def perf(seed=2):
    """Timing sweeps (perf_*): target size, triangle count, overdraw (opaque / blend / depth), fragment ALU, vertex count with
    culled geometry, instancing, tile coverage."""
    R = random.Random(seed)
    S = {}
    quad = lambda W, H, z=0.5, cc=(0.4, 0.6, 0.8, 0.5): tri_px(W, H, [(0, 0), (W, 0), (0, H)], z=z, cols=[cc] * 3) + tri_px(W, H, [(W, 0), (W, H), (0, H)], z=z, cols=[cc] * 3)
    for n in (32, 64, 128, 256, 512, 1024, 2048):
        S[f'perf_size{n}'] = scene(n, n, quad(n, n) + ['draw 0 6'], shader='const')
    for n in (1, 4, 16, 64, 256, 1024, 4096):
        g = max(1, int((n / 2) ** 0.5)); W = 512; st = W / g; body = []
        for y in range(g):
            for x in range(g):
                body += quad_at(W, x * st, y * st, st)
        S[f'perf_tris{2 * g * g}'] = scene(W, W, body + [f'draw 0 {6 * g * g}'], shader='const')
    for L in (1, 2, 4, 8):
        body = []
        for l in range(L):
            body += quad(512, 512, z=0.9 - 0.1 * l)
        S[f'perf_over{L}_opaque'] = scene(512, 512, body + [f'draw 0 {6 * L}'], shader='alu32')
        S[f'perf_over{L}_depth'] = scene(512, 512, body + [f'draw 0 {6 * L}'], shader='alu32', depth='less')
        S[f'perf_over{L}_blend'] = scene(512, 512, body + [f'draw 0 {6 * L}'], shader='alu32', blend='alpha')
    for a in (0, 8, 32, 128, 512):
        S[f'perf_alu{a}_512'] = scene(512, 512, quad(512, 512) + ['draw 0 6'], shader=f'alu{a}')
    for a in (32, 128):
        S[f'perf_alu{a}_1024'] = scene(1024, 1024, quad(1024, 1024) + ['draw 0 6'], shader=f'alu{a}')
    for n in (3, 300, 3000, 30000):
        body = [f'v 0.1 0.1 0.5 1 1 1 1 1'] * n   # degenerate (zero-area) triangles: vertex work only
        S[f'perf_vtx{n}'] = scene(256, 256, body + [f'draw 0 {n}'], shader='const')
    for k in (1, 16, 256, 2048):
        S[f'perf_inst{k}'] = scene(256, 256, tri_px(256, 256, [(1, 1), (3, 1), (1, 3)], cols=[(1, 0, 0, 1)] * 3) + [f'draw 0 3 {k}'], shader='const')
    for frac in (1, 4, 16):
        W = 1024; t = W // frac
        S[f'perf_cover1_{frac}'] = scene(W, W, quad_at(W, 0, 0, t) + ['draw 0 6'], shader='alu32')
    return S


def quad_at(W, x0, y0, s, cc=(0.4, 0.6, 0.8, 1)):
    return tri_px(W, W, [(x0, y0), (x0 + s, y0), (x0, y0 + s)], cols=[cc] * 3) + tri_px(W, W, [(x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)], cols=[cc] * 3)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 1
    S = perf() if '--perf' in sys.argv else gen(seed)
    suite = []
    for k, v in S.items():
        (out / f'{k}.txt').write_text(v)
        suite.append({'name': k, 'scene': str(out / f'{k}.txt')})
    (out / 'suite.json').write_text(json.dumps(suite, indent=1))
    print(len(suite), 'scenes')


if __name__ == '__main__':
    main()
