#!/usr/bin/env python3
"""Alternate checked functional atomic binaries; CPU execution, not GPU timing."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--current', type=Path, default=ROOT / 'rust/target/release/examples/atomic_execution')
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if min(args.pairs, args.repeats) < 1:
        ap.error('pairs and repeats must be positive')
    runs = []
    for lanes in [1, 32]:
        for pair in range(args.pairs):
            for build in (['baseline', 'current'] if pair % 2 == 0 else ['current', 'baseline']):
                exe = args.baseline if build == 'baseline' else args.current
                out = subprocess.check_output([str(exe.resolve()), '512', '128', str(lanes), str(args.repeats)],
                    text=True, timeout=120)
                found = set()
                for line in out.splitlines():
                    m = re.fullmatch(r'kind=(device|threadgroup) groups=512 steps=128 lanes=(\d+) repeats=(\d+) median_ms=([\d.]+) checked_words=(\d+) atomics=(\d+)', line)
                    if not m or m[1] in found:
                        raise AssertionError(f'unexpected benchmark output: {line}')
                    if [int(m[i]) for i in [2, 3, 5, 6]] != [lanes, args.repeats, 512 * lanes * 2, 512 * lanes * 128]:
                        raise AssertionError(f'wrong benchmark dimensions: {line}')
                    found.add(m[1])
                    runs.append(dict(lanes=lanes, pair=pair, build=build, kind=m[1], median_ms=float(m[4]), output=line))
                if found != {'device', 'threadgroup'}:
                    raise AssertionError('benchmark did not check both atomic kinds')
    summary = []
    for lanes in [1, 32]:
        for kind in ['device', 'threadgroup']:
            med = {b: statistics.median(r['median_ms'] for r in runs if r['lanes'] == lanes
                and r['kind'] == kind and r['build'] == b) for b in ['baseline', 'current']}
            summary.append(dict(lanes=lanes, kind=kind, **med, speedup=med['baseline'] / med['current']))
    report = json.dumps(dict(pairs=args.pairs, repeats=args.repeats, summary=summary, runs=runs), indent=2) + '\n'
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end='')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
