#!/usr/bin/env python3
"""Record raw fp32 results of one transcendental over whole binades (hardware, direct DRM).

For each requested binade (sign and biased exponent of x), every one of the 2^23 mantissas is run through
`<op> r14, r6` with record_transcendentals' straight-line kernel (each 2^18-lane chunk runs twice and must agree),
and the raw result words are saved to testsuite/transcendental/samples/<op>_<label>.bin.gz, label = the binade's
lower bound (e.g. exp2_0.25, exp2_-4). Usage: record_trans_ranges.py OP LO[,LO...]  e.g. exp2 0.25,2,-4
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import gpu_safety, run_bench as B
import record_transcendentals as T


def binade(lo):
    if lo == 0 or abs(lo) != 2.0 ** math.floor(math.log2(abs(lo))):
        raise ValueError(f'{lo}: not a power of two')
    e = int(math.log2(abs(lo))) + 127
    if not 1 <= e <= 254:
        raise ValueError(f'{lo}: not a normal binade')
    return e | (0x100 if lo < 0 else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('op', choices=['exp2', 'log2', 'sin_pt_2', 'sin_pt_1', 'rcp', 'rsqrt'])
    ap.add_argument('los')
    ap.add_argument('--out', type=Path, default=T.ROOT/'testsuite/transcendental')
    a = ap.parse_args()
    los = [float(x) for x in a.los.split(',')]
    exps = [binade(lo) for lo in los]
    gpu_safety.require_budget(1)
    R = B.Runner(Path('/scratch/apple-m2-playground/sim/transcendental'))
    for lo, e in zip(los, exps):
        path = a.out/'samples'/f'{a.op}_{lo:g}.bin.gz'
        if path.exists():
            print(f'{path} exists, skipped'); continue
        T.save(path, T.results(R, a.op, T.mantissas(e)))
        print(f'saved {path}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
