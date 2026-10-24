#!/usr/bin/env python3
"""Check lane-dependent matrix shuffles introduced during LLVM lowering."""

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/matrix-fma-lane-inputs.ll'
LLC = os.environ.get('LLVM_LLC', 'llc')
OBJCOPY = os.environ.get('LLVM_OBJCOPY', 'llvm-objcopy')
RUN = os.environ.get('AGX_RUN', str(ROOT / 'rust/target/release/agx_run'))


def reference():
    # A is distributed as two elements per lane. Gather its eight elements
    # for the output row, multiply by B=1, and add C=0 in integer arithmetic.
    values = []
    for lane in range(32):
        row = ((lane >> 4) << 2) | ((lane >> 1) & 3)
        holders = [((row >> 2) << 4) | ((j >> 1) << 3) |
                   ((row & 3) << 1) | (j & 1) for j in range(4)]
        total = sum(4 * holder + 3 for holder in holders)
        values.extend([float(total), float(total)])
    return values


def main():
    expected = reference()
    reports = []
    with tempfile.TemporaryDirectory(prefix='agx-matrix-shuffle-') as directory:
        work = Path(directory)
        for cpu, native in [('agx-g13g', False), ('agx-g14c', False), ('agx-g14c', True)]:
            tag = f'{cpu}-{native}'
            obj, code, dump = [work / f'{tag}{ext}' for ext in ['.o', '.bin', '.dump']]
            features = '+simd-matrix' if native else '-simd-matrix'
            subprocess.run([LLC, '-mtriple=agx', f'-mcpu={cpu}', f'-mattr={features}',
                            '-verify-machineinstrs', '-filetype=obj', str(SOURCE),
                            '-o', str(obj)], check=True)
            subprocess.run([OBJCOPY, '--dump-section', f'.text={code}', str(obj),
                            str(work / f'{tag}-copy.o')], check=True)
            result = subprocess.run([RUN, str(code), '--grid', '32,1,1', '--wg', '32,1,1',
                                     '--uniform', '0=4096', '--uniform', '1=0',
                                     '--alloc', '4096=4096', '--dump', f'4096=256:{dump}',
                                     '--max-steps', '10000'],
                                    check=True, capture_output=True, text=True)
            actual = list(struct.unpack('<64f', dump.read_bytes()))
            mismatches = [(i, a, e) for i, (a, e) in enumerate(zip(actual, expected)) if a != e]
            if mismatches:
                raise RuntimeError(f'{cpu}: wrong matrix elements: {mismatches[:8]}')
            reports.append({'cpu': cpu, 'matrix': 'native' if native else 'emulated', 'checked_elements': 64,
                            'code_bytes': code.stat().st_size, **json.loads(result.stdout)})
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
