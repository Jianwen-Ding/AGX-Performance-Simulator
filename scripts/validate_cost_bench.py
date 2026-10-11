#!/usr/bin/env python3
"""Compare agx_cost predictions with the measured benchmark (offline). Same loops as run_bench:
slope between 16 and 64 steps, so loop overhead cancels as on hardware."""
import gzip, json, sys
from pathlib import Path
import agx_cost as C, run_bench as B
P = C.Params()
b = json.loads(gzip.open(C.BENCH).read())['ops']
T = 20
rows = []
for op, v in b.items():
    def slope(ch, groups):
        c = []
        for S in (16, 64):
            code = B.loop(op, ch, S, T); ins = C.decode(code, P); pcs, _ = C.trace(code)
            c.append(C.simulate(ins, pcs, P, groups))
        return C.to_ref(c[1]-c[0])/(T*48)
    pred = {'latency': slope(1, 1), 'simd_issue': slope(8, 1), 'gpu': slope(8, 4)/4}
    meas = {m: v[m]['cycles_per_step'] for m in pred}
    rows.append((op, pred, meas))
    print(f"{op:18s} " + '  '.join(f"{m[:3]} {meas[m]:6.2f}/{pred[m]:6.2f}" for m in pred), flush=True)
err = {m: [abs(p[m]-q[m])/q[m] for _, p, q in rows] for m in ('latency', 'simd_issue', 'gpu')}
for m, e in err.items():
    e = sorted(e); print(m, 'median rel err', round(e[len(e)//2], 3), 'max', round(e[-1], 3), 'within 10%:', sum(x <= .1 for x in e), '/', len(e))
