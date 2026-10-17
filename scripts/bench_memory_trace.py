#!/usr/bin/env python3
"""Alternate checked memory_trace binaries; CPU tracing only, no GPU used."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--current', type=Path, default=ROOT / 'rust/target/release/examples/memory_trace')
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if min(args.pairs, args.repeats) < 1:
        ap.error('pairs and repeats must be positive')
    runs = []
    for workers in [1, 4]:
        for stride in [1, 16]:
            for pair in range(args.pairs):
                for build in (['baseline', 'current'] if pair % 2 == 0 else ['current', 'baseline']):
                    exe = args.baseline if build == 'baseline' else args.current
                    out = subprocess.check_output([str(exe.resolve()), '128', '128', str(stride), str(args.repeats)],
                        env={**os.environ, 'AGX_THREADS': str(workers)}, text=True, timeout=120)
                    found = set()
                    for line in out.splitlines():
                        m = re.fullmatch(r'kind=(cost|functional) groups=128 loads=128 stride=\d+ repeats=\d+ median_ms=([\d.]+) instructions=17280 accesses=16384', line)
                        if not m or m[1] in found:
                            raise AssertionError(f'unexpected benchmark output: {line}')
                        found.add(m[1])
                        runs.append(dict(workers=workers, stride=stride, pair=pair, build=build,
                            kind=m[1], median_ms=float(m[2]), output=line))
                    if found != {'cost', 'functional'}:
                        raise AssertionError('benchmark did not check both tracers')
    summary = []
    for workers in [1, 4]:
        for stride in [1, 16]:
            for kind in ['cost', 'functional']:
                med = {b: statistics.median(r['median_ms'] for r in runs if r['workers'] == workers
                    and r['stride'] == stride and r['kind'] == kind and r['build'] == b) for b in ['baseline', 'current']}
                summary.append(dict(workers=workers, stride=stride, kind=kind, **med, speedup=med['baseline'] / med['current']))
    text = json.dumps(dict(pairs=args.pairs, repeats=args.repeats, summary=summary, runs=runs), indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
