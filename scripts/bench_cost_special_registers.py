#!/usr/bin/env python3
"""Paired checked cost-SR CPU timings; baseline cases are full-mask 32-bit reads only."""
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
    cases = {7, 18, 19, 80, 56, 57, 58, 59}
    runs = []
    for pair in range(args.pairs):
        for build in (['baseline', 'current'] if pair % 2 == 0 else ['current', 'baseline']):
            output = subprocess.check_output([str(getattr(args, build).resolve()), '128', '128', str(args.repeats)], text=True, timeout=180)
            found = set()
            for line in output.splitlines():
                m = re.fullmatch(r'sr=(\d+) groups=128 reads=128 repeats=(\d+) median_ms=([\d.]+) checked_lanes=4096', line)
                if not m or int(m[1]) not in cases or int(m[1]) in found or int(m[2]) != args.repeats:
                    raise AssertionError(f'unexpected benchmark result: {line}')
                found.add(int(m[1]))
                runs.append(dict(pair=pair, build=build, sr=int(m[1]), median_ms=float(m[3]), output=line))
            if found != cases:
                raise AssertionError('incomplete benchmark cases')
    summary = []
    for sr in sorted(cases):
        med = {b: statistics.median(r['median_ms'] for r in runs if r['sr'] == sr and r['build'] == b) for b in ['baseline', 'current']}
        summary.append(dict(sr=sr, **med, speedup=med['baseline']/med['current']))
    args.output.write_text(json.dumps(dict(pairs=args.pairs, repeats=args.repeats, summary=summary, runs=runs), indent=2)+'\n')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
