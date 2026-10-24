#!/usr/bin/env python3
"""Check nested-region compilation against lane-wise integer results."""

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/region-cache-nested.ll'
LLC = os.environ.get('LLVM_LLC', 'llc')
BASELINE = os.environ.get('LLVM_LLC_BASELINE')
OBJCOPY = os.environ.get('LLVM_OBJCOPY', 'llvm-objcopy')
RUN = os.environ.get('AGX_RUN', str(ROOT / 'rust/target/release/agx_run'))
COST = os.environ.get('AGX_COST', str(ROOT / 'rust/target/release/agx_cost'))


def reference(n):
    return [sum(3 * (i + 1) if (lane + i) % 2 == 0
                else 7 if lane % 3 == 0 else 2 for i in range(n))
            for lane in range(32)]


def main():
    reports = []
    with tempfile.TemporaryDirectory(prefix='agx-region-cache-') as directory:
        work = Path(directory)
        for cpu in ['agx-g13g', 'agx-g14c']:
            variants = []
            for name, compiler in [('current', LLC)] + ([('baseline', BASELINE)]
                                                       if BASELINE else []):
                stem = work / f'{cpu}-{name}'
                obj, code = stem.with_suffix('.o'), stem.with_suffix('.bin')
                subprocess.run([compiler, '-mtriple=agx', f'-mcpu={cpu}',
                                '-verify-machineinstrs', '-filetype=obj', str(SOURCE),
                                '-o', str(obj)], check=True)
                subprocess.run([OBJCOPY, '--dump-section', f'.text={code}',
                                str(obj), str(stem.with_suffix('.copy.o'))], check=True)
                cases = []
                for n in [0, 1, 4, 9]:
                    dump = stem.with_suffix('.dump')
                    inputs = ['--grid', '32,1,1', '--wg', '32,1,1',
                              '--uniform', '0=4096', '--uniform', '1=0',
                              '--uniform', f'2={n}', '--alloc', '4096=4096']
                    result = subprocess.run([RUN, str(code), *inputs,
                                             '--dump', f'4096=128:{dump}',
                                             '--max-steps', '100000'], check=True,
                                            capture_output=True, text=True)
                    actual = list(struct.unpack('<32I', dump.read_bytes()))
                    if actual != reference(n):
                        raise RuntimeError(f'{cpu}/{name}/n={n}: '
                                           f'expected {reference(n)}, got {actual}')
                    prediction = subprocess.run([COST, str(code), '--groups', '1',
                                                 *inputs], check=True,
                                                capture_output=True, text=True)
                    cases.append({'n': n, 'outputs_checked': len(actual),
                                  'execution': json.loads(result.stdout),
                                  'cost': json.loads(prediction.stdout)})
                variants.append((code.read_bytes(), cases))
            if BASELINE and variants[0] != variants[1]:
                raise RuntimeError(f'{cpu}: cache invalidation changed code, '
                                   'execution, or predicted cost')
            reports.append({'cpu': cpu, 'code_bytes': len(variants[0][0]),
                            'baseline_checked': bool(BASELINE),
                            'cases': variants[0][1]})
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
