#!/usr/bin/env python3
"""Re-score agx_cost against stored random-kernel hardware measurements (offline)."""
import json, statistics as st, sys
import cost_hw as H
path = sys.argv[1] if len(sys.argv) > 1 else '/scratch/apple-m2-playground/cost/validation.json'
v = json.load(open(path))
for gm, groups in (('one', 1), ('gpu', 4)):
    errs = []
    for x in v:
        lines, body = H.kernel(x['seed'], len(x['ops']))
        p = H.predict(H.counted(body, 100), H.counted(body, 500), 400, groups)
        errs.append((p-x[gm]['measured'])/x[gm]['measured'])
    a = sorted(abs(e) for e in errs)
    print(gm, 'kernels', len(a), 'median |err|', round(a[len(a)//2], 3), 'p90', round(a[int(.9*len(a))], 3), 'max', round(a[-1], 3),
          'signed median', round(st.median(errs), 3), flush=True)
