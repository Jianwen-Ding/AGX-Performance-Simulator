#!/usr/bin/env python3
"""Compile LLVM's opt-in leaf-call fixture and replay its code in AGXsim."""

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/experimental-leaf-call.ll'
LLC = os.environ.get('LLVM_LLC', 'llc')
OBJCOPY = os.environ.get('LLVM_OBJCOPY', 'llvm-objcopy')
AGX_RUN = os.environ.get('AGX_RUN', str(ROOT / 'rust/target/release/agx_run'))

with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    obj, code, dump = work / 'leaf.o', work / 'leaf.bin', work / 'dump.bin'
    subprocess.run([LLC, '-mtriple=agx', '-mcpu=agx-g14c',
                    '-agx-experimental-leaf-calls', '-verify-machineinstrs',
                    '-filetype=obj', str(SOURCE), '-o', str(obj)], check=True)
    subprocess.run([OBJCOPY, '--dump-section', f'.text={code}',
                    str(obj), str(work / 'copy.o')], check=True)
    data = code.read_bytes()
    assert data[:2] == bytes.fromhex('10c0'), data[:6].hex()
    target = struct.unpack_from('<i', data, 2)[0]
    assert data[6:14] == bytes.fromhex('5216000000008800'), data[6:14].hex()
    assert data[target + 28:target + 30] == bytes.fromhex('1402'), data[target:target + 32].hex()
    env = dict(os.environ, AGX_EXPERIMENTAL_RET='buffer-pop2')
    result = subprocess.run([AGX_RUN, str(code), '--grid', '1,1,1',
                             '--wg', '32,1,1', '--alloc', '4096=4096',
                             '--dump', f'4096=4:{dump}', '--max-steps', '1000'],
                            check=True, capture_output=True, text=True, env=env)
    actual = json.loads(result.stdout)
    assert actual['instructions'] == 9 and actual['groups'] == 1, actual
    assert dump.read_bytes() == struct.pack('<I', 7), dump.read_bytes().hex()
    print('LLVM leaf call, return, reconvergence and simulator store: pass')
