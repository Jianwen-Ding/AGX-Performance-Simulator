import json, sys, statistics as st
import cost_hw as H, agx_cost as C, agx_loop
e = agx_loop.encode
mode = sys.argv[1]
orig = C.Params.__init__
def patched(self, *a, **k):
    orig(self, *a, **k); self.cal['issue_mode'] = mode
C.Params.__init__ = patched
v = json.load(open('/scratch/apple-m2-playground/cost/validation.json'))
for gm, groups in (('one', 1), ('gpu', 4)):
    errs = []
    for x in v:
        lines, body = H.kernel(x['seed'], 24)
        p = H.predict(H.counted(body, 100), H.counted(body, 500), 400, groups)
        errs.append((p-x[gm]['measured'])/x[gm]['measured'])
    a = sorted(abs(x) for x in errs)
    print(mode, gm, 'median', round(a[12], 3), 'p90', round(a[21], 3), 'max', round(a[-1], 3), 'bias', round(st.median(errs), 3), flush=True)
h2 = json.load(open('/scratch/apple-m2-playground/cost/hyp2.json'))
ind = lambda k: [f'iadd r{r}, r{r}, 1' for r in (7, 8, 9, 10, 11, 12, 13, 14)[:k]]
err = []
for lead, li in {'rcp': 'rcp r6, r6', 'bfi': 'bfi r6, r6, r24, r26', 'imadd': 'imadd r6, r6, r25, r24', 'floor': 'floor r6, r6', 'fadd32': 'fadd32 r6, r6, r24'}.items():
    for k in (0, 1, 2, 4):
        parts = [e(x) for x in [li]+ind(k)]*8
        p = (H.predict(H.counted(b''.join(parts), 100), H.counted(b''.join(parts), 500), 400, 1)-35.25)/8
        err.append(abs(p-h2[f'{lead}+{k}'])/h2[f'{lead}+{k}'])
print(mode, 'lead+k median', round(sorted(err)[len(err)//2], 3), 'max', round(max(err), 3))
