#!/usr/bin/env python3
"""Offline: fp32 boundary sweeps (safe-01 part 3) replayed in the simulator, and their IEEE-difference claims.

results/2026-10-16-float-edges/hw_samples.json.gz: per sweep of record_float_edges.py, every input (small sweeps) or
~24k samples (both ends incl. the edge set, random) with hardware words, plus the SHA-256 of the full result array.
Checks: the exact recorded kernel replayed through agx_run reproduces every retained word; and the semantics
claimed in playground novel-results/2026-10-16-conversions hold on the retained words.
"""
import gzip, hashlib, json, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import record_float_edges as FE
import record_suite as RS, run_drm_compute as drm

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = json.load(gzip.open(ROOT/'results/2026-10-16-float-edges/hw_samples.json.gz', 'rt'))
    S = FE.sweeps()
    fails = []

    def check(name, ok):
        print(('ok   ' if ok else 'FAIL ')+name, flush=True)
        if not ok: fails.append(name)

    words = {}
    for k, (text, xs, consts) in S.items():
        e = d['sweeps'][k]
        idx = np.array(e['idx']); hw = np.array(e['hw'], np.uint32); sx = xs[idx]
        code = drm.build_main(RS.alu_body(RS.e(text), FE.N))
        sim = np.concatenate([FE.simulate_chunk(code, sx, s0, consts) for s0 in range(0, len(sx), FE.N)])
        p = FE.OUT/f'{k}.npz'
        full = ''
        if p.exists():
            fh = np.load(p)['hw'].astype(np.uint32)
            full = ', full array hash ' + ('ok' if hashlib.sha256(fh.tobytes()).hexdigest() == e['hw_sha256'] else 'MISMATCH')
        check(f'{k}: {len(sx)} of {e["inputs"]} inputs replayed in agx_run match the hardware{full}',
              e['inputs'] == len(xs) and np.array_equal(sim, hw) and 'MISMATCH' not in full)
        words[k] = (sx, hw)
    f32 = lambda u: u.astype(np.uint32).view(np.float32)
    x, hw = words['f32_to_f16']; x = x[:, 0]; h = (hw[:, 0] & 0xffff).astype(np.uint16)
    ieee = f32(x).astype(np.float16).view(np.uint16); nan = np.isnan(f32(x))
    check('f32 -> f16 (fadd32 r14l, x, -0.0): IEEE round-to-nearest-even incl. fp16 denormal results; NaN -> 0x7e00',
          np.array_equal(h[~nan], ieee[~nan]) and (h[nan] == 0x7e00).all())
    x, hw = words['f16_to_f32']; x = x[:, 0].astype(np.uint16); h = hw[:, 0]
    ieee = x.view(np.float16).astype(np.float32).view(np.uint32); nan = np.isnan(x.view(np.float16))
    check('f16 -> f32 (fadd32 r14, x16, -0.0): exact for all 65,536 incl. fp16 denormals; NaN -> 0x7fc00000',
          np.array_equal(h[~nan], ieee[~nan]) and (h[nan] == 0x7fc00000).all())
    for op, fn in (('floor', np.floor), ('ceil', np.ceil), ('trunc', np.trunc), ('rint', np.rint)):
        x, hw = words[f'round_{op}']; x = x[:, 0]; h = hw[:, 0]
        den = ((x >> 23) & 0xff) == 0; nan = np.isnan(f32(x))
        ieee = fn(f32(x)).astype(np.float32).view(np.uint32)
        daz = np.where(x >> 31 != 0, 0x80000000, 0).astype(np.uint32)
        check(f'{op}: IEEE except denormal inputs read as signed zero and NaN -> 0x7fc00000',
              np.array_equal(h[~den & ~nan], ieee[~den & ~nan]) and np.array_equal(h[den], daz[den]) and (h[nan] == 0x7fc00000).all())
    for cc in ('eq', 'lt', 'gt', 'lte', 'gte', 'ltn', 'gtn'):
        x, hw = words[f'cmp_{cc}']; a, b = x[:, 0], x[:, 1]; sel = hw[:, 0] == 0x11111111
        dz = lambda u: np.where(((u >> 23) & 0xff) == 0, 0.0, f32(u).astype(np.float64))
        da, db = dz(a), dz(b); na, nb = np.isnan(da), np.isnan(db)
        want = {'eq': da == db, 'lt': da < db, 'gt': da > db, 'lte': da <= db, 'gte': da >= db,
                'ltn': np.where(na, False, np.where(nb, True, da < db)), 'gtn': np.where(na, False, np.where(nb, True, da > db))}[cc]
        check(f'fcmpsel {cc}: ' + ('IEEE compare with denormals as zero' if cc[-1] != 'n' else 'minNum / maxNum select (NaN a -> b, NaN b -> a), denormals as zero')
              + f' over all {len(a)} edge pairs', np.array_equal(sel, want) and ((hw[:, 0] == 0x11111111) | (hw[:, 0] == 0x22222222)).all())
    print(f'{len(fails)} failures')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
