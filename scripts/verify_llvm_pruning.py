#!/usr/bin/env python3
"""Check LLVM's early device-function pruning against the functional and cost models."""

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/prune-before-inline.ll'
LLC = os.environ.get('LLVM_LLC', 'llc')
OBJCOPY = os.environ.get('LLVM_OBJCOPY', 'llvm-objcopy')
RUN = os.environ.get('AGX_RUN', str(ROOT / 'rust/target/release/agx_run'))
COST = os.environ.get('AGX_COST', str(ROOT / 'rust/target/release/agx_cost'))


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix='agx-pruning-') as directory:
        work = Path(directory)
        for cpu in ['agx-g13g', 'agx-g14c']:
            variants = []
            for enabled in [False, True]:
                stem = work / f'{cpu}-{enabled}'
                obj, code, dump = [stem.with_suffix(ext) for ext in ['.o', '.bin', '.dump']]
                subprocess.run([LLC, '-mtriple=agx', f'-mcpu={cpu}',
                                f'-agx-prune-before-inline={str(enabled).lower()}',
                                '-verify-machineinstrs', '-filetype=obj', str(SOURCE),
                                '-o', str(obj)], check=True)
                subprocess.run([OBJCOPY, '--dump-section', f'.text={code}',
                                str(obj), str(stem.with_suffix('.copy.o'))], check=True)
                inputs = ['--grid', '1,1,1', '--wg', '32,1,1',
                          '--uniform', '0=4096', '--uniform', '1=0', '--uniform', '2=35',
                          '--alloc', '4096=4096']
                run = subprocess.run([RUN, str(code), *inputs,
                                      '--dump', f'4096=4:{dump}', '--max-steps', '1000'],
                                     check=True, capture_output=True, text=True)
                execution = json.loads(run.stdout)
                actual = dump.read_bytes()
                if actual != struct.pack('<I', 42):
                    raise RuntimeError(f'{cpu} pruning={enabled}: expected 42, got {actual.hex()}')
                prediction = subprocess.run([COST, str(code), '--groups', '1', *inputs],
                                            check=True, capture_output=True, text=True)
                cost = json.loads(prediction.stdout)
                variants.append((code.read_bytes(), execution, cost))
            if variants[0] != variants[1]:
                raise RuntimeError(f'{cpu}: pruning changed live code, simulator execution, or predicted cost')
            results.append({'cpu': cpu, 'output': 42, 'code_bytes': len(variants[0][0]),
                            'same_live_code': True, 'same_execution': True, 'same_cost': True})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
