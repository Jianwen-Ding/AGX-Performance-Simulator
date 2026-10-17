#!/usr/bin/env python3
"""Alternate checked functional get_sr binaries; CPU execution, no GPU timing."""
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
    ap.add_argument('--current', type=Path, default=ROOT / 'rust/target/release/examples/special_registers')
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if min(args.pairs, args.repeats) < 1:
        ap.error('pairs and repeats must be positive')
    runs = []
    cases = {(sr, active) for sr in [7, 18, 19, 80, 56, 57, 58, 59] for active in [32, 13]}
    for pair in range(args.pairs):
        for build in (['baseline', 'current'] if pair % 2 == 0 else ['current', 'baseline']):
            exe = args.baseline if build == 'baseline' else args.current
            out = subprocess.check_output([str(exe.resolve()), '256', '128', str(args.repeats)], text=True, timeout=180)
            found = set()
            for line in out.splitlines():
                m = re.fullmatch(r'sr=(\d+) active=(\d+) groups=256 reads=128 repeats=(\d+) median_ms=([\d.]+) checked_words=8192', line)
                if not m or int(m[3]) != args.repeats:
                    raise AssertionError(f'unexpected checked output: {line}')
                key = (int(m[1]), int(m[2]))
                if key not in cases or key in found:
                    raise AssertionError(f'duplicate/unexpected case: {key}')
                found.add(key)
                runs.append(dict(pair=pair, build=build, sr=key[0], active=key[1], median_ms=float(m[4]), output=line))
            if found != cases:
                raise AssertionError('benchmark did not check every selector/mask')
    summary = []
    for sr, active in sorted(cases):
        med = {b: statistics.median(r['median_ms'] for r in runs if r['sr'] == sr and r['active'] == active and r['build'] == b)
            for b in ['baseline', 'current']}
        summary.append(dict(sr=sr, active=active, **med, speedup=med['baseline'] / med['current']))
    report = json.dumps(dict(pairs=args.pairs, repeats=args.repeats, summary=summary, runs=runs), indent=2) + '\n'
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end='')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
