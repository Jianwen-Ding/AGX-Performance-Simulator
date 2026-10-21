#!/usr/bin/env python3
"""Replay the hardware simd_matrix_fmadd32 / fmadd16 records (B15, M2 Max) in agx_run and require the same D words.

results/2026-10-21-simd-matrix/arith{32,16}.json: 120 random A, B, C sets per width run on the GPU by the playground's
results/2026-10-21-b15-simd-matrix/b15_map.py (lane i loads its A, B, C pairs from table words 2i, 2i+1 at 0, 64, 128 for
fp32, or word i for fp16; D is stored at 1024 + 2i (+1), or 1024 + i). Prints one JSON line; exit 1 on any mismatch."""
import json, struct, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import coverage as cov, record_suite as RS, run_drm_compute as drm, agx_loop
cov.install()
mov = RS.mov; e = lambda *a: agx_loop.encode(*a); st = agx_loop.store

def program(w):
    if w == 32:
        b = mov(8, 0) + mov(9, 0) + mov(27, 2) + e('imadd r24, r1, r27, 0')
        for reg, base in ((12, 0), (16, 64), (20, 128)):
            b += mov(26, base) + e('iadd r25, r24, r26') + e(f'device_load 0, i32, xy, r{reg}_r{reg + 1}, u8_u9, r25, unsigned')
        b += e('wait 0') + e('simd_matrix_fmadd32 r8_r9, r12_r13, r16_r17, r20_r21')
        b += mov(26, 1024) + e('iadd r25, r24, r26') + st(8, 8, 25) + mov(26, 1025) + e('iadd r25, r24, r26') + st(9, 8, 25)
    else:
        b = mov(8, 0)
        for reg, base in ((12, 0), (16, 64), (20, 128)):
            b += mov(26, base) + e('iadd r25, r1, r26') + e(f'device_load 0, i32, x, r{reg}, u8_u9, r25, unsigned')
        b += e('wait 0') + e('simd_matrix_fmadd16 r8l_r8h, r12l_r12h, r16l_r16h, r20l_r20h')
        b += mov(26, 1024) + e('iadd r25, r1, r26') + st(8, 8, 25)
    return drm.build_main(b + e('wait 0') + e('iadd r2, r8, 0'))

res, bad = {}, 0
for w in (32, 16):
    code, words = program(w), (64 if w == 32 else 32)
    sets = json.load(open(ROOT / f'results/2026-10-21-simd-matrix/arith{w}.json'))
    mism = 0
    for s in sets:
        t = [0] * 1200
        t[0:len(s['a'])], t[64:64 + len(s['b'])], t[128:128 + len(s['c'])] = s['a'], s['b'], s['c']
        _, tab = cov.simulate(code, 32, 32, 48, 0, t)
        mism += sum(tab[1024 + i] != s['d'][i] for i in range(words))
    res[f'fp{w}'] = {'sets': len(sets), 'words': len(sets) * words, 'mismatched_words': mism}
    bad += mism
print(json.dumps(res))
sys.exit(1 if bad else 0)
