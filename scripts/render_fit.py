#!/usr/bin/env python3
"""Fit / check the render timing model (agx_sim::raster::timing) against direct-ioctl measurements.

    python3 -I scripts/render_fit.py fit CASES.jsonl [...]      # re-simulate each case's plan offline, least-squares fit, print coefficients
    python3 -I scripts/render_fit.py check CASES.jsonl [...]    # predicted (agx_render) vs measured, median / p90 error
The fit is ordinary least squares on non-negative features (pure Python normal equations), reported with leave-one-out error.
"""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_check as rc

VTX = ['one', 'tiles', 'vertices', 'primitives', 'draws']
FRAG = ['one', 'tiles_processed', 'frag_core_ns', 'bgeot_core_ns', 'translucent_samples']


def load(paths):
    out = []
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if 'hw_vtx_ns' not in r:
                continue
            plan = Path('/scratch/render/v2/runs') / r['name'] / 'plan.txt'
            sim, _ = rc.simulate(plan)
            if 'error' in sim:
                print('skip', r['name'], sim['error'], file=sys.stderr); continue
            sim['one'] = 1
            sim['bgeot_core_ns'] = sim.get('bg_core_ns', 0) + sim.get('eot_core_ns', 0)
            r['sim'] = sim
            out.append(r)
    return out


def solve(X, y):
    n = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) for j in range(n)] for i in range(n)]
    b = [sum(x[i] * t for x, t in zip(X, y)) for i in range(n)]
    for i in range(n):  # ridge on the diagonal against collinearity
        A[i][i] += 1e-9 * (A[i][i] + 1)
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]; b[c], b[p] = b[p], b[c]
        for r in range(n):
            if r != c and A[c][c]:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * q for a, q in zip(A[r], A[c])]; b[r] -= f * b[c]
    return [b[i] / A[i][i] if A[i][i] else 0.0 for i in range(n)]


def fit_rel(rows, feats, key):
    # relative-error weighting: divide each row by its measurement
    X = [[r['sim'].get(f, 0) / r[key] for f in feats] for r in rows]
    y = [1.0 for r in rows]
    return solve(X, y)


def err(rows, feats, coef, key):
    e = []
    for r in rows:
        p = sum(c * r['sim'].get(f, 0) for c, f in zip(coef, feats))
        e.append(abs(p / r[key] - 1))
    e.sort()
    return e[len(e) // 2], e[int(len(e) * 0.9)]


def main():
    mode, paths = sys.argv[1], sys.argv[2:]
    rows = load(paths)
    if mode == 'fit':
        for key, feats in (('hw_vtx_ns', VTX), ('hw_frag_ns', FRAG)):
            coef = fit_rel(rows, feats, key)
            loo = []
            for i in range(len(rows)):
                c = fit_rel(rows[:i] + rows[i + 1:], feats, key)
                p = sum(a * rows[i]['sim'].get(f, 0) for a, f in zip(c, feats))
                loo.append(abs(p / rows[i][key] - 1))
            loo.sort()
            print(key, dict(zip(feats, [round(c, 6) for c in coef])), 'fit median/p90', [round(x, 3) for x in err(rows, feats, coef, key)],
                  'LOO median/p90', round(loo[len(loo) // 2], 3), round(loo[int(len(loo) * 0.9)], 3))
    else:
        for key, pk in (('hw_vtx_ns', 'predicted_vtx_ns'), ('hw_frag_ns', 'predicted_frag_ns')):
            e = sorted(abs(r['sim'][pk] / r[key] - 1) for r in rows)
            worst = sorted(rows, key=lambda r: -abs(r['sim'][pk] / r[key] - 1))[:5]
            print(key, 'n', len(e), 'median', round(e[len(e) // 2], 3), 'p90', round(e[int(len(e) * 0.9)], 3),
                  'worst', [(r['name'], r[key], round(r['sim'][pk])) for r in worst])


if __name__ == '__main__' and sys.argv[1] in ('fit', 'check'):
    main()


# ---- nonlinear models (the ones agx_sim::raster::timing implements)
def vtx_model(c, s):
    # fixed + per screen tile (tiler setup) + vertex-shader core time + per primitive (assembly / binning) + per draw
    # vertices: saturating (the vertex stage spreads over more clusters as work grows: +0.5 us at 192, +3.7 at 3000, +5.4 at 30000)
    v = s['vertices']
    return (c[0] + c[1] * s['tiles'] + c[2] * s.get('vs_core_ns', 0) + c[3] * s['primitives'] + c[4] * s['draws']
            + c[5] * v / (1 + v / max(1.0, c[7])) + c[6] * s.get('prim_tiles', 0))


def frag_model(c, s):
    # fixed + max(per processed tile, shaded fragment core time + background / end-of-tile core time) + per translucent sample
    core = s.get('frag_core_ns', 0) + s.get('bg_core_ns', 0) + s.get('eot_core_ns', 0)
    # tail: the shading of a core's last tiles does not overlap (ratio to the core model 2.2 at 144 tiles, 1.35 at 2304):
    # c[6] tiles' worth of one tile's shading on its core = c[6] * core * 38 / tiles
    shade = c[2] * core + c[6] * core * 38 / max(1, s['tiles_processed'])
    return (c[0] + max(c[1] * s['tiles_processed'], shade) + c[3] * s.get('translucent_samples', 0) + c[4] * s.get('zls_bytes', 0)
            + c[5] * min(s.get('rasterized', 0), 1) + c[7] * s.get('prim_tiles', 0))


def nm(f, x0, iters=4000):
    n = len(x0); pts = [list(x0)] + [[x0[j] * (1.25 if j == i else 1) + (0.01 if j == i and x0[j] == 0 else 0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = sorted(range(n + 1), key=lambda i: vals[i]); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        cen = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        xr = [cen[j] + (cen[j] - pts[-1][j]) for j in range(n)]; fr = f(xr)
        if fr < vals[0]:
            xe = [cen[j] + 2 * (cen[j] - pts[-1][j]) for j in range(n)]; fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [cen[j] + 0.5 * (pts[-1][j] - cen[j]) for j in range(n)]; fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [[pts[0][j] + 0.5 * (p[j] - pts[0][j]) for j in range(n)] for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[0]


def fit_nl(rows, model, key, x0):
    loss = lambda c: sum(abs(model(c, r['sim']) / r[key] - 1) for r in rows) / len(rows) + (1e3 if min(c) < 0 else 0)
    return nm(loss, x0)


def nl_main(paths):
    held = [p for p in paths if p.startswith('held=')]
    paths = [p for p in paths if not p.startswith('held=')]
    rows = load(paths)
    hrows = load([h[5:] for h in held]) if held else []
    for key, model, x0 in (('hw_vtx_ns', vtx_model, [6700, 1.0, 0.25, 0.01, 45, 2.5, 0.1, 3000]), ('hw_frag_ns', frag_model, [3200, 11.3, 1.3, 0.001, 0.0, 750, 2.1, 0.7])):
        c = fit_nl(rows, model, key, x0)
        e = sorted(abs(model(c, r['sim']) / r[key] - 1) for r in rows)
        loo = []
        for i in range(len(rows)):
            ci = fit_nl(rows[:i] + rows[i + 1:], model, key, c)
            loo.append(abs(model(ci, rows[i]['sim']) / rows[i][key] - 1))
        loo.sort()
        worst = sorted(rows, key=lambda r: -abs(model(c, r['sim']) / r[key] - 1))[:5]
        if hrows:
            he = sorted(abs(model(c, r['sim']) / r[key] - 1) for r in hrows)
            hw_ = sorted(hrows, key=lambda r: -abs(model(c, r['sim']) / r[key] - 1))[:4]
            print(key, 'HELD-OUT n', len(he), 'median', round(he[len(he) // 2], 3), 'p90', round(he[int(len(he) * .9)], 3),
                  'worst', [(r['name'], r[key], round(model(c, r['sim']))) for r in hw_])
        print(key, [round(x, 5) for x in c], 'fit median/p90', round(e[len(e) // 2], 3), round(e[int(len(e) * .9)], 3),
              'LOO median/p90', round(loo[len(loo) // 2], 3), round(loo[int(len(loo) * .9)], 3),
              'worst', [(r['name'], r[key], round(model(c, r['sim']))) for r in worst])


if __name__ == '__main__' and sys.argv[1] == 'nl':
    nl_main(sys.argv[2:])
