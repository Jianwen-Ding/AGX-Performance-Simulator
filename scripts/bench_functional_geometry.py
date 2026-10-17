#!/usr/bin/env python3
"""Alternate checked CPU geometry/barrier benchmarks; retain every paired median."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--current', type=Path, required=True)
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if min(args.pairs, args.repeats) < 1:
        ap.error('pairs and repeats must be positive')
    cases = {(t, b, p) for t in [1, 32, 256] for b in [0, 8] for p in [0, 17]}
    runs = []
    for pair in range(args.pairs):
        for build in (['baseline', 'current'] if pair % 2 == 0 else ['current', 'baseline']):
            out = subprocess.check_output([str(getattr(args, build).resolve()), '2048', str(args.repeats)], text=True, timeout=180)
            found = set()
            for line in out.splitlines():
                m = re.fullmatch(r'threads=(\d+) barriers=(\d+) placement=(\d+) workgroups=2048 repeats=(\d+) median_ms=([\d.]+) checked_instructions=(\d+)', line)
                if not m:
                    raise AssertionError(f'unexpected benchmark output: {line}')
                t, b, p = key = tuple(int(m[i]) for i in [1, 2, 3])
                if key not in cases or key in found or int(m[4]) != args.repeats or int(m[6]) != 2048 * ((t + 31) // 32) * (b + 1):
                    raise AssertionError(f'invalid benchmark case: {line}')
                found.add(key)
                runs.append(dict(pair=pair, build=build, threads=t, barriers=b, placement=p, median_ms=float(m[5]), output=line))
            if found != cases:
                raise AssertionError('missing checked cases')
    summary = []
    for t, b, p in sorted(cases):
        med = {build: statistics.median(r['median_ms'] for r in runs if (r['threads'], r['barriers'], r['placement']) == (t, b, p) and r['build'] == build) for build in ['baseline', 'current']}
        summary.append(dict(threads=t, barriers=b, placement=p, **med, speedup=med['baseline']/med['current']))
    args.output.write_text(json.dumps(dict(pairs=args.pairs, repeats=args.repeats, summary=summary, runs=runs), indent=2) + '\n')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
