import json, statistics as st, sys, math, itertools
sys.path.insert(0, '/save/workspace/AGX-Performance-Simulator/scripts')
from render_fit import nm
HELD = ('held1', 'held2')
rows = []
for f in ['suite1','perf1','perf2','held1','held2','order']:
    for l in open(f'/scratch/render/v3/{f}.resim.jsonl'):
        r = json.loads(l); r['set'] = f; rows.append(r)
for f in ['sweep','sweep2']:
    for l in open(f'/scratch/render/v3/{f}/cases.jsonl'):
        r = json.loads(l); r['set'] = f; rows.append(r)
rows = [r for r in rows if r.get('hw_frag_ns') and r.get('sim') and r['hw_vtx_ns'] < 20000]
for r in rows:
    r['tl'] = json.load(open(f"/scratch/render/v3/tl/{r['name']}.json"))
print(len(rows), 'rows')

def tile_feats(r, B):
    out = []
    for tx, ty, samp, prims, vis in r['tl']:
        blk, last = 0, None
        for p in prims:
            if last is None or p // B != last:
                blk += 1
            last = p // B
        out.append((blk, samp, len(prims), vis))
    return out
FE = {}
def feats(r, B):
    k = (r['name'], B)
    if k not in FE: FE[k] = tile_feats(r, B)
    return FE[k]

def isp_time(r, c, B, K):
    L, s, p, v = c
    chains = sorted((L * b + s * sa + p * n + v * vi for b, sa, n, vi in feats(r, B)), reverse=True)
    # tiles with no primitives still cost nothing here (the tile term covers them); LPT over 38 cores, K tiles in flight per core
    cores = [[0.0, 0.0] for _ in range(38)]   # (sum, max)
    for ch in chains:
        i = min(range(38), key=lambda j: max(cores[j][0] / K, cores[j][1]))
        cores[i][0] += ch; cores[i][1] = max(cores[i][1], ch)
    return max(max(a / K, m) for a, m in cores)

def pred(r, c, B, K):
    p = r['sim']['timing']
    F = r['predicted_frag_ns'] - max(p['frag_tiles'], p['frag_shading']) - p.get('frag_bins', 0)
    return F + max(p['frag_tiles'], p['frag_shading'], isp_time(r, c, B, K))

def errs(rs, c, B, K):
    return [abs(pred(r, c, B, K) / r['hw_frag_ns'] - 1) for r in rs]

train = [r for r in rows if r['set'] not in HELD]
held = [r for r in rows if r['set'] in HELD]
def q(e, x): e = sorted(e); return e[int(x * (len(e) - 1))]
best = None
for B in (16, 32, 64):
    for K in (1, 2, 4, 8):
        f = lambda c: sum(min(x, 1.0) ** 2 for x in errs(train, [abs(v) for v in c], B, K))
        c = [abs(v) for v in nm(f, [60.0, 0.03, 5.0, 40.0], iters=300)]
        et, eh = errs(train, c, B, K), errs(held, c, B, K)
        print(B, K, [round(v, 4) for v in c], 'train med/p90 %.3f %.3f' % (st.median(et), q(et, .9)), 'held %.3f %.3f' % (st.median(eh), q(eh, .9)), flush=True)
        if best is None or f(c) < best[0]: best = (f(c), B, K, c)
print('best', best)
_, B, K, c = best
for r in sorted(rows, key=lambda r: -abs(pred(r, c, B, K) / r['hw_frag_ns'] - 1))[:12]:
    print(r['set'], r['name'], r['hw_frag_ns'], round(pred(r, c, B, K)), round(r['predicted_frag_ns']))
