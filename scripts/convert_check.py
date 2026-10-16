#!/usr/bin/env python3
"""Offline: the simulator's convert model (numpy replica of fsem::convert) against the recorded hardware results.

For every /scratch/agxsim-data/convert/<form>.npz (record_convert.py), regenerate the form's inputs, apply the model and
report mismatches grouped by category. Usage: convert_check.py [--forms REGEX] [--show N]
"""
import argparse, glob, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_convert as RC

f16_to_f64 = lambda h: h.astype(np.uint16).view(np.float16).astype(np.float64)


def f32_daz(b):
    b = b.astype(np.uint32)
    den = (b >> 23) & 0xff == 0
    return np.where(den, np.where(b >> 31 != 0, -0.0, 0.0), b.view(np.float32).astype(np.float64))


def model(m, r, d, s, x):
    w = (8, 16, 32)[m >> 2]
    signed, rte = m & 1, (m >> 1) & 1
    mask = np.uint64((1 << w)-1)
    if r == 0:
        v = f16_to_f64(x & 0xffff) if s == 'r6l' else f32_daz(x)
        lo, hi = (-(1 << (w-1)), (1 << (w-1))-1) if signed else (0, (1 << w)-1)
        with np.errstate(invalid='ignore'):
            rr = np.rint(v) if rte else np.trunc(v)
            out = np.where(np.isnan(v), 0, np.clip(np.nan_to_num(rr, posinf=hi, neginf=lo), lo, hi)).astype(np.int64)
        res = (out.astype(np.uint64) & mask).astype(np.uint64)
    else:
        src = (x & 0xffff) if s == 'r6l' else x
        v = (src.astype(np.uint64) & mask).astype(np.int64)
        if signed:
            v = np.where(v >> (w-1) & 1 == 1, v-(1 << w), v)
        if d == 'r14l':
            near = v.astype(np.float64).astype(np.float16)
            res = near.view(np.uint16).astype(np.uint64)
            if not rte:
                mag = np.abs(near.astype(np.float64))
                res = np.where(mag > np.abs(v.astype(np.float64)), res-1, res)
                res = np.where(np.abs(v) >= 1 << 16, np.where(v < 0, 0xfc00, 0x7c00), res).astype(np.uint64)
        else:
            near = v.astype(np.float32)          # numpy int64 -> float32 rounds to nearest even
            res = near.view(np.uint32).astype(np.uint64)
            if not rte:
                res = np.where(np.abs(near.astype(np.float64)) > np.abs(v.astype(np.float64)), res-1, res)
    res = res & np.uint64(0xffffffff)
    return (np.uint64(0xdead0000) | (res & np.uint64(0xffff))) if d == 'r14l' else res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--forms', default='.')
    ap.add_argument('--show', type=int, default=4)
    a = ap.parse_args()
    total = bad = 0
    for f in RC.FORMS:
        p = RC.OUT/(RC.name(*f)+'.npz')
        if not p.exists() or not re.search(a.forms, RC.name(*f)):
            continue
        hw = np.load(p)['hw'].astype(np.uint64)
        x = RC.form_inputs(f)
        mo = model(*f, x)
        miss = np.nonzero(mo != hw)[0]
        total += len(x); bad += len(miss)
        line = f'{"ok  " if not len(miss) else "FAIL"} {RC.name(*f):45s} {len(x):10d} inputs'
        if len(miss):
            line += f', {len(miss)} differ: ' + ' '.join(f'x={x[i]:#010x} model={mo[i]:#x} hw={hw[i]:#x}' for i in miss[:a.show])
        print(line, flush=True)
    print(f'{total} inputs, {bad} mismatches')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
