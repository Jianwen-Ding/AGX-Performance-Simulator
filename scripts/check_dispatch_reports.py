#!/usr/bin/env python3
"""Check complete prediction reports against the retained, unrebaselined SHA-256.

This checks simulator output, not fresh hardware execution. The reference hash
was established by the paired 36-dispatch comparison in the cost-tracing pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT / 'rust/target/release/agx_dispatch')
    ap.add_argument('--reference', type=Path, default=ROOT / 'results/2026-10-17-cost-tracing/benchmark.json')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    reference = json.loads(args.reference.read_text())['retained_dispatch_comparison']
    directory = ROOT / 'results/2026-10-11-dispatch'
    rows = json.loads((directory / 'dispatch.json').read_text())['rows']
    if not rows or len(rows) != reference['equal_reports']:
        raise AssertionError('retained dispatch count differs from the reference')
    jobs = []
    for row in rows:
        jobs.append([str(directory / row['code']), '--grid', str(row['grid']), '--wg', str(row['wg']),
            '--regs', str(row['regs']), '--tg-bytes', str(row.get('tg_bytes', 0)), '--uniform', '4=0',
            '--uniform', f'5={row["grid"]}', '--uniform', '7=1', '--uniform', '9=2'])
    with tempfile.TemporaryDirectory(prefix='agx-report-check-') as temp:
        batch = Path(temp) / 'jobs.jsonl'
        batch.write_text('\n'.join(json.dumps(j) for j in jobs) + '\n')
        out = subprocess.check_output([str(args.run.resolve()), '--batch', str(batch)], text=True,
            env={**os.environ, 'AGX_THREADS': os.environ.get('AGX_THREADS', '2')}, timeout=300)
    reports = [json.loads(s) for s in out.splitlines()]
    if len(reports) != len(jobs) or any('error' in r for r in reports):
        raise AssertionError('missing or failed dispatch reports')
    canonical = json.dumps(reports, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != reference['sha256']:
        raise AssertionError(f'complete report hash changed: {digest}, expected {reference["sha256"]}')
    result = dict(reports=len(reports), sha256=digest, matches_reference=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(reports)} complete prediction reports match the retained SHA-256')


if __name__ == '__main__':
    main()
