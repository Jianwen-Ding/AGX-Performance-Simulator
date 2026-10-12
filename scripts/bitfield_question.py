#!/usr/bin/env python3
"""Decide how bfeil (and bfi / extr) with an immediate mask treat shift amounts of 32 or more, from hardware recordings
of the coverage catalog's bitfield cases (coverage/expected/alu/{bfi,bfeil,extr,shlhi,shrhi}_*).

    scripts/bitfield_question.py            report: per case, pass / fail under each simulator semantics, and a verdict

The functional simulator takes 7 bits of the shift for every form by default (fitted to the one recorded full-width
case); AGX_BFEIL_MASKED_SHIFT32=1 makes bfeil with a non-zero mask take it modulo 32. Which one the recordings match
decides whether viterbi, the three shiftfuse columns and lz4dec are compiler bugs (default right) or simulator gaps
(the alternative right): tasks/spirv-08-applications.md, notes 37-40.

Record the cases first (on the M2): scripts/coverage.py record --area alu --match 'bfeil|bfi_|extr|shlhi|shrhi'
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECK = ROOT / 'rust/target/release/suite_check'
PAT = re.compile(r'^(bfi|bfeil|extr|shlhi|shrhi)_')


def run(case, mod32):
    env = dict(os.environ)
    env.pop('AGX_BFEIL_MASKED_SHIFT32', None)
    if mod32:
        env['AGX_BFEIL_MASKED_SHIFT32'] = '1'
    r = subprocess.run([str(CHECK), str(ROOT / 'coverage/expected'), '--case', case, '-v'], capture_output=True, text=True, env=env)
    m = re.search(r'total\s+pass\s+(\d+)\s*/\s*(\d+)', r.stdout)
    if not m or m.group(2) == '0':
        return None
    return m.group(1) == m.group(2)


def main():
    d = ROOT / 'coverage/expected/alu'
    cases = sorted(p.name[:-len('.json.gz')] for p in d.glob('*.json.gz') if PAT.match(p.name)) if d.exists() else []
    if not cases:
        print(json.dumps({'status': 'PENDING', 'why': "no recorded bitfield cases: on the M2 run scripts/coverage.py record "
                                                    "--area alu --match 'bfeil|bfi_|extr|shlhi|shrhi'"}))
        return 0
    rows = [{'case': c, 'default_7bit': run(c, False), 'masked_mod32': run(c, True)} for c in cases]
    for r in rows:
        print(json.dumps(r))
    masked = [r for r in rows if r['case'].startswith('bfeil') and 'mask' in r['case']]
    other = [r for r in rows if r not in masked]
    a = all(r['default_7bit'] for r in rows)
    b = all(r['masked_mod32'] for r in masked) and all(r['default_7bit'] for r in other)
    verdict = ('A: masked bfeil takes 7 bits like the full-width form; viterbi / shiftfuse / lz4dec failures are '
               'compiler bugs (fuse_ubfe)' if a and not b else
               'B: masked bfeil takes its shift modulo 32; make AGX_BFEIL_MASKED_SHIFT32 the default' if b and not a else
               'both match (no recorded shift >= 32 reached a masked bfeil?)' if a and b else
               'neither matches: look at the failing cases (other forms or other fields differ too)')
    print(json.dumps({'cases': len(rows), 'masked_bfeil_cases': len(masked),
                      'default_fails': [r['case'] for r in rows if r['default_7bit'] is False],
                      'mod32_fails': [r['case'] for r in rows if r['masked_mod32'] is False], 'verdict': verdict}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
