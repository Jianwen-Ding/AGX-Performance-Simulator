#!/usr/bin/env python3
"""Offline: the functional simulator's fp32 transcendentals against the M2, bit for bit.

results/2026-10-16-transcendentals/hw_samples.json.gz holds hardware results recorded on 2026-10-16: exp2 (76 binades,
-2^127 .. 2^127), sin_pt_2 (60 binades, |x| < 2^127), log2 (12 binades, 2^-126 .. 2^127), 1026 inputs per binade,
and Mesa's sin / cos sequence plus the fma-split variant (chain_sin2 / chain_cos2) on 32k inputs each. Each set runs
in agx_run with the exact kernel that recorded it (results/2026-10-16-transcendentals/kernels/*.bin: load x, the op or the
chain, store) and every result word must match.
"""
import gzip, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = ROOT/'results/2026-10-16-transcendentals'
RUN = os.environ.get('AGX_RUN', str(ROOT/'rust/target/release/agx_run'))
C = 1 << 18


def sim(kernel, xs, tmp):
    out = np.zeros(len(xs), np.uint32)
    for s in range(0, len(xs), C):
        part = xs[s:s+C]
        t = np.zeros(2*C, np.uint32); t[:len(part)] = part; t.tofile(tmp/'t.bin')
        np.full(8+C+8+16, 0xcdcdcdcd, np.uint32).tofile(tmp/'o.bin')
        r = subprocess.run([RUN, str(kernel), '--grid', str(C), '--wg', '256', '--uniform', '4=0', '--uniform', f'5={C}',
                            '--uniform', '6=0x100000', '--uniform', '7=0x11', '--uniform', '8=0x800000', '--uniform', '9=0x11',
                            '--map', f'0x1100100000={tmp/"o.bin"}', '--map', f'0x1100800000={tmp/"t.bin"}',
                            '--dump', f'0x1100800000={8*C}:{tmp/"r.bin"}'], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'agx_run failed: {r.stderr[:300]}')
        out[s:s+len(part)] = np.fromfile(tmp/'r.bin', np.uint32)[C:C+len(part)]
    return out


def main():
    d = json.load(gzip.open(T/'hw_samples.json.gz', 'rt'))
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for op, v in d['ops'].items():
            xs, hw = np.array(v['x'], np.uint32), np.array(v['hw'], np.uint32)
            got = sim(T/'kernels'/f'{op}.bin', xs, Path(tmp))
            n = int(np.sum(got != hw))
            bad += n
            print(f'{"ok  " if not n else "FAIL"} {op:11s} {len(xs):7d} inputs' + (f' ({v["binades"]} binades)' if 'binades' in v else '')
                  + (f', {n} differ; first x={xs[got != hw][0]:#x}' if n else ''))
    print(f'{bad} mismatching words')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
