#!/usr/bin/env python3
"""Offline: the simulator (agx_run, the exact recorded kernel) against the recorded float-edge hardware words
(/scratch/agxsim-data/float_edges/<sweep>.npz from record_float_edges.py). Every chunk of the smaller sweeps is
simulated; for f32_to_f16 (4,352 chunks) every --stride-th chunk plus the first and last.
Usage: float_edges_check.py [--sweeps REGEX] [--stride N]
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_float_edges as FE
import record_suite as RS, run_drm_compute as drm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweeps', default='.')
    ap.add_argument('--stride', type=int, default=16)
    a = ap.parse_args()
    S = {k: v for k, v in FE.sweeps().items() if re.search(a.sweeps, k)}
    total = bad_all = 0
    for k, (text, xs, consts) in S.items():
        p = FE.OUT/f'{k}.npz'
        if not p.exists():
            print(f'skip {k}: not recorded'); continue
        d = np.load(p)
        if not np.array_equal(d['x'], xs):
            print(f'FAIL {k}: recorded inputs differ from the generator'); bad_all += 1; continue
        code = drm.build_main(RS.alu_body(RS.e(text), FE.N))
        chunks = list(range(0, len(xs), FE.N))
        if len(chunks) > 64:
            chunks = sorted(set(chunks[::a.stride]+[chunks[-1]]))
        bad = n = 0
        first = None
        for s0 in chunks:
            sim = FE.simulate_chunk(code, xs, s0, consts)
            hw = d['hw'][s0:s0+len(sim)]
            diff = np.nonzero((sim != hw).any(1))[0]
            bad += len(diff); n += len(sim)
            if len(diff) and first is None:
                i = diff[0]; first = (xs[s0+i].tolist(), hw[i].tolist(), sim[i].tolist())
        total += n; bad_all += bad
        print(f'{"ok  " if not bad else "FAIL"} {k:12s} {n:10d} inputs compared' + (f', {bad} differ; first x/hw/sim {first}' if bad else ''))
    print(f'{total} inputs, {bad_all} mismatches')
    return 1 if bad_all else 0


if __name__ == '__main__':
    sys.exit(main())
