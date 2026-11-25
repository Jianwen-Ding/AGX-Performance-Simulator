#!/usr/bin/env python3
"""Summarise render_check suite records: python3 -I scripts/render_summary.py CASES.jsonl
A case counts as exact only if it has a pixel comparison and every pixel matched; errors are listed, never counted as passes."""
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
err = [r for r in rs if 'pixels' not in r]
cmp_ = [r for r in rs if 'pixels' in r]
exact = [r for r in cmp_ if r['exact'] == r['pixels']]
tol = [r for r in cmp_ if r['within_tol'] == r['pixels']]
print(json.dumps({'cases': len(rs), 'errors': len(err), 'compared': len(cmp_), 'exact': len(exact), 'within_1': len(tol),
                  'pixels': sum(r['pixels'] for r in cmp_), 'pixels_exact': sum(r['exact'] for r in cmp_),
                  'replay_eq_gl': sum(r.get('replay_vs_gl_exact') == r['pixels'] for r in cmp_)}))
for r in err:
    print('ERROR', r['name'], (r.get('sim_error') or r.get('error') or r.get('hw_error') or '')[:200])
for r in cmp_:
    if r['exact'] != r['pixels']:
        print('DIFF', r['name'], r['exact'], '/', r['pixels'], 'max', r['max_diff'], r['first_bad'][:2])
