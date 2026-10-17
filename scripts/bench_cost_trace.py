#!/usr/bin/env python3
"""Checked before/after cost tracing and full-dispatch benchmark; no GPU used.

Build rust/examples/cost_trace.rs and agx_dispatch in both checkouts. The example
can be copied unchanged into the baseline. Alternating order limits host drift.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'rust/target/release'


def invoke(exe, args, env):
    return subprocess.check_output([str(exe), *args], env=env, text=True, timeout=180)


def measure_trace(exe, workers, padding, repeats):
    # A fresh helper has only one child, so ru_maxrss describes this benchmark,
    # not the maximum over every previously launched baseline/current process.
    helper = ('import json,resource,subprocess,sys; '
              's=subprocess.check_output(sys.argv[1:],text=True); '
              'print(json.dumps(dict(output=s.strip(),peak_rss_kib='
              'resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)))')
    out = invoke(Path(sys.executable), ['-c', helper, str(exe), '128', '128', str(padding), str(repeats)],
                 {**os.environ, 'AGX_THREADS': str(workers)})
    row = json.loads(out)
    row['median_ms'] = float(re.search(r'median_ms=([\d.]+)', row['output'])[1])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True, help='baseline cost_trace example binary')
    ap.add_argument('--baseline-dispatch', type=Path, required=True)
    ap.add_argument('--trace', type=Path, default=BIN / 'examples/cost_trace')
    ap.add_argument('--dispatch', type=Path, default=BIN / 'agx_dispatch')
    ap.add_argument('--pairs', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--compare-retained', action='store_true', help='also compare complete reports for all retained dispatch rows')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if min(args.pairs, args.repeats) < 1:
        ap.error('pairs and repeats must be positive')
    targets = {'baseline': (args.baseline.resolve(), args.baseline_dispatch.resolve()),
               'current': (args.trace.resolve(), args.dispatch.resolve())}
    result = {'pairs': args.pairs, 'repeats': args.repeats, 'trace_runs': [], 'dispatch_runs': []}
    order = lambda pair: ['baseline', 'current'] if pair % 2 == 0 else ['current', 'baseline']
    for workers, padding in [(1, 0), (1, 64), (4, 64)]:
        for pair in range(args.pairs):
            for label in order(pair):
                row = measure_trace(targets[label][0], workers, padding, args.repeats)
                result['trace_runs'].append(dict(pair=pair, build=label, workers=workers, padding_kib=padding, **row))
    env = {**os.environ, 'AGX_THREADS': '1'}
    with tempfile.TemporaryDirectory(prefix='agx-cost-bench-') as temp:
        code = Path(temp) / 'code.bin'
        invoke(targets['current'][0], ['1', '128', '0', '1'], {**env, 'AGX_BENCH_CODE': str(code)})
        job = [str(code), '--grid', '1048576', '--wg', '64', '--regs', '8', '--no-mem']
        expected = None
        for pair in range(args.pairs):
            for label in order(pair):
                samples = []
                for _ in range(args.repeats + 1):
                    start = time.perf_counter()
                    report = json.loads(invoke(targets[label][1], job, env))
                    samples.append(1000 * (time.perf_counter() - start))
                    if expected is None:
                        expected = report
                    if report != expected:
                        raise AssertionError('complete dispatch report changed')
                result['dispatch_runs'].append(dict(pair=pair, build=label,
                    median_ms=statistics.median(samples[1:]), samples_ms=samples[1:]))
        result['dispatch_report'] = expected
        if args.compare_retained:
            directory = ROOT / 'results/2026-10-11-dispatch'
            rows = json.loads((directory / 'dispatch.json').read_text())['rows']
            jobs = []
            for row in rows:
                jobs.append([str(directory / row['code']), '--grid', str(row['grid']), '--wg', str(row['wg']),
                    '--regs', str(row['regs']), '--tg-bytes', str(row.get('tg_bytes', 0)), '--uniform', '4=0',
                    '--uniform', f'5={row["grid"]}', '--uniform', '7=1', '--uniform', '9=2'])
            batch = Path(temp) / 'jobs.jsonl'
            batch.write_text('\n'.join(json.dumps(j) for j in jobs) + '\n')
            reports = {label: [json.loads(s) for s in invoke(targets[label][1], ['--batch', str(batch)], env).splitlines()]
                       for label in targets}
            if len(reports['current']) != len(jobs) or reports['baseline'] != reports['current']:
                raise AssertionError('retained dispatch report comparison failed')
            canonical = json.dumps(reports['current'], sort_keys=True, separators=(',', ':')).encode()
            result['retained_dispatch_comparison'] = dict(equal_reports=len(jobs), sha256=hashlib.sha256(canonical).hexdigest())
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    for workers, padding in [(1, 0), (1, 64), (4, 64)]:
        values = {b: statistics.median(x['median_ms'] for x in result['trace_runs']
                  if x['build'] == b and x['workers'] == workers and x['padding_kib'] == padding) for b in targets}
        print(f'trace workers={workers} padding={padding}: {values}', file=sys.stderr)
    values = {b: statistics.median(x['median_ms'] for x in result['dispatch_runs'] if x['build'] == b) for b in targets}
    print(f'full dispatch: {values}', file=sys.stderr)


if __name__ == '__main__':
    main()
