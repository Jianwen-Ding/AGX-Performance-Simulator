#!/usr/bin/env python3
"""Paired checked cost/functional tracing and peak RSS; CPU only, no GPU timing."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir', type=Path, required=True)
    ap.add_argument('--current-dir', type=Path, default=ROOT / 'rust/target/release/examples')
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--padding-kib', type=int, nargs='+', default=[0, 64, 1024])
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if min(args.pairs, args.repeats, args.steps) < 1 or min(args.padding_kib) < 0:
        ap.error('pairs, repeats and steps must be positive; padding must be nonnegative')
    helper = ('import json,resource,subprocess,sys; '
              's=subprocess.check_output(sys.argv[1:],text=True); '
              'print(json.dumps(dict(output=s.strip(),peak_rss_kib='
              'resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)))')
    runs = []
    for kind in ['cost', 'functional']:
        for workers in [1, 4]:
            for padding in args.padding_kib:
                for pair in range(args.pairs):
                    for build in (['baseline', 'current'] if pair % 2 == 0 else ['current', 'baseline']):
                        directory = args.baseline_dir if build == 'baseline' else args.current_dir
                        exe = (directory / f'{kind}_trace').resolve()
                        out = subprocess.check_output([sys.executable, '-c', helper, str(exe), '128', str(args.steps),
                            str(padding), str(args.repeats)], env={**os.environ, 'AGX_THREADS': str(workers)}, text=True, timeout=180)
                        row = json.loads(out)
                        ms = re.search(r'median_ms=([\d.]+)', row['output'])
                        # functional_trace has two groups/workgroup; cost_trace has one.
                        expected = 128 * (args.steps + 2) * (2 if kind == 'functional' else 1)
                        if not ms or f'instructions={expected}' not in row['output']:
                            raise AssertionError(f'unexpected checked output: {row}')
                        runs.append(dict(kind=kind, workers=workers, padding_kib=padding, pair=pair, build=build,
                            median_ms=float(ms[1]), **row))
    summary = []
    for kind in ['cost', 'functional']:
        for workers in [1, 4]:
            for padding in args.padding_kib:
                rows = [r for r in runs if r['kind'] == kind and r['workers'] == workers and r['padding_kib'] == padding]
                med = {b: statistics.median(r['median_ms'] for r in rows if r['build'] == b) for b in ['baseline', 'current']}
                rss = {b + '_rss_kib': statistics.median(r['peak_rss_kib'] for r in rows if r['build'] == b) for b in ['baseline', 'current']}
                summary.append(dict(kind=kind, workers=workers, padding_kib=padding, **med, **rss, speedup=med['baseline'] / med['current']))
    report = json.dumps(dict(pairs=args.pairs, repeats=args.repeats, steps=args.steps,
        padding_kib=args.padding_kib, summary=summary, runs=runs), indent=2) + '\n'
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end='')
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
