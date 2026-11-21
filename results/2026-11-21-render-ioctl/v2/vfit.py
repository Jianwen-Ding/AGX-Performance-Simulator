import json, statistics as st, sys
sys.path.insert(0, '/save/workspace/AGX-Performance-Simulator/scripts')
from render_fit import nm
rows = []
for f in ['suite1','perf1','perf2','held1','held2','order','sweep','sweep2','vsweep']:
    for l in open(f'/scratch/render/v3/{f}.r2.jsonl'):
        r = json.loads(l); r['set'] = f
        if r.get('hw_vtx_ns') and r.get('sim') and r['hw_vtx_ns'] < 2.5 * r['predicted_vtx_ns']:
            rows.append(r)
HELD = ('held1', 'held2')
train = [r for r in rows if r['set'] not in HELD]; held = [r for r in rows if r['set'] in HELD]
def q(e, x): e = sorted(e); return e[int(x * (len(e) - 1))]
def base(c, s):
    v = s['vertices']
    return c[0] + c[1] * s['tiles'] + c[2] * s['draws'] + c[3] * v / (1 + v / max(1, c[4])) + c[5] * s['prim_tiles']
def inst(c, s, i0):
    a, n0, p, qq = c[i0:i0 + 4]
    t = 0.0
    for cnt, n, w in s.get('draw_shapes', []):
        if n > 1 and cnt > 0:
            t += (w / 9) ** qq * n / (1 + n / max(1, n0)) * (3 / cnt) ** abs(p)
    return a * t
def vary(c, s, k):
    return c[k] * (s['vertex_output_words'] - 5 * s['vertices']) / (1 + s['vertices'] / max(1, c[4]))
MODELS = {
    'inst': (lambda c, s: base(c, s) + vary(c, s, 6) + c[7] * s['prim_blocks'][4] + inst(c, s, 8), [6770, 1.0, 42, 2.2, 1846, 0.09, 0.3, 0.1, 10, 600, 2, 1.5]),

    'base': (base, [6770, 1.0, 42, 2.2, 1846, 0.09]),
    'outw': (lambda c, s: base(c, s) + c[6] * s['vertex_output_words'] / (1 + s['vertex_output_words'] / max(1, c[7])), [6770, 1.0, 42, 2.2, 1846, 0.09, 0.3, 5000]),
    'varyw': (lambda c, s: base(c, s) + c[6] * (s['vertex_output_words'] - 5 * s['vertices']) / (1 + s['vertices'] / max(1, c[4])), [6770, 1.0, 42, 2.2, 1846, 0.09, 0.3]),
    'outw_prims': (lambda c, s: base(c, s) + c[6] * (s['vertex_output_words'] - 5 * s['vertices']) / (1 + s['vertices'] / max(1, c[4])) + c[7] * s['rasterized'], [6770, 1.0, 42, 2.2, 1846, 0.09, 0.3, 0.5]),
    'blk': (lambda c, s: base(c, s) + c[6] * (s['vertex_output_words'] - 5 * s['vertices']) / (1 + s['vertices'] / max(1, c[4])) + c[7] * s['prim_blocks'][4], [6770, 1.0, 42, 2.2, 1846, 0.09, 0.3, 0.1]),
}
for name, (m, x0) in MODELS.items():
    f = lambda c: sum(min(abs(m(c, r['sim']) / r['hw_vtx_ns'] - 1), 1) ** 2 for r in train)
    c = nm(f, x0, iters=5000)
    et = [abs(m(c, r['sim']) / r['hw_vtx_ns'] - 1) for r in train]; eh = [abs(m(c, r['sim']) / r['hw_vtx_ns'] - 1) for r in held]
    worst = sorted(rows, key=lambda r: -abs(m(c, r['sim']) / r['hw_vtx_ns'] - 1))[:6]
    print(name, [round(x, 4) for x in c], 'train %.3f %.3f held %.3f %.3f' % (st.median(et), q(et, .9), st.median(eh), q(eh, .9)))
    print('   ', [(r['name'][:16], r['hw_vtx_ns'], round(m(c, r['sim']))) for r in worst])
