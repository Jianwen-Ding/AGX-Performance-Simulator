#!/usr/bin/env python3
"""Check legacy LLVM structurization's optional uniformity analysis."""

import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/structurize-analysis-lazy.ll'
LLC = os.environ.get('LLVM_LLC', 'llc')
OBJCOPY = os.environ.get('LLVM_OBJCOPY', 'llvm-objcopy')
RUN = os.environ.get('AGX_RUN', str(ROOT / 'rust/target/release/agx_run'))
COST = os.environ.get('AGX_COST', str(ROOT / 'rust/target/release/agx_cost'))


def main():
    source = SOURCE.read_text()
    reports = []
    with tempfile.TemporaryDirectory(prefix='agx-structurize-') as directory:
        work = Path(directory)
        for cpu in ['agx-g13g', 'agx-g14c']:
            for name, value in [('uniform_first', 42), ('divergent', 42),
                                ('uniform_last', 24)]:
                # Compile each entry separately so .text starts at that kernel.
                match = re.search(r'^define agx_kernel void @' + name +
                                  r'\(.*?^}', source, re.M | re.S)
                if match is None:
                    raise RuntimeError(f'missing shared LLVM fixture: {name}')
                ir = work / f'{name}.ll'
                ir.write_text('declare i32 @llvm.agx.thread.position.in.grid.x()\n' +
                              match[0] + '\n')
                variants = []
                for skip in [False, True]:
                    stem = work / f'{cpu}-{name}-{skip}'
                    obj, code = stem.with_suffix('.o'), stem.with_suffix('.bin')
                    subprocess.run([LLC, '-mtriple=agx', f'-mcpu={cpu}',
                                    f'-structurizecfg-skip-uniform-regions={str(skip).lower()}',
                                    '-verify-machineinstrs', '-filetype=obj', str(ir),
                                    '-o', str(obj)], check=True)
                    subprocess.run([OBJCOPY, '--dump-section', f'.text={code}',
                                    str(obj), str(stem.with_suffix('.copy.o'))], check=True)
                    cases = []
                    for condition in ([0] if name == 'divergent' else [0, 1]):
                        initial = work / 'initial.bin'
                        initial.write_bytes(struct.pack('<I', 99) + bytes(4092))
                        dump = stem.with_suffix('.dump')
                        inputs = ['--grid', '1,1,1', '--wg', '32,1,1',
                                  '--uniform', '0=4096', '--uniform', '1=0',
                                  '--uniform', f'2={condition}',
                                  '--map', f'4096={initial}']
                        result = subprocess.run([RUN, str(code), *inputs,
                                                 '--dump', f'4096=4:{dump}',
                                                 '--max-steps', '1000'], check=True,
                                                capture_output=True, text=True)
                        expected = value if condition == 0 else 99
                        actual = struct.unpack('<I', dump.read_bytes())[0]
                        if actual != expected:
                            raise RuntimeError(f'{cpu}/{name}/skip={skip}: '
                                               f'expected {expected}, got {actual}')
                        prediction = subprocess.run([COST, str(code), '--groups', '1',
                                                     *inputs], check=True,
                                                    capture_output=True, text=True)
                        cases.append({'condition': condition, 'output': actual,
                                      'execution': json.loads(result.stdout),
                                      'cost': json.loads(prediction.stdout)})
                    variants.append((code.read_bytes(), cases))
                if variants[0] != variants[1]:
                    raise RuntimeError(f'{cpu}/{name}: optional analysis changed code, '
                                       'execution, or predicted cost')
                reports.append({'cpu': cpu, 'kernel': name,
                                'code_bytes': len(variants[0][0]),
                                'same_code_execution_cost': True,
                                'cases': variants[0][1]})
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
