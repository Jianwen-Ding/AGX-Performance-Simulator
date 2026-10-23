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
SCALAR_SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/experimental-leaf-call-scalar.ll'
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
    assert data[:8] == bytes.fromhex('42100000000010c0'), data[:8].hex()
    target = 6 + struct.unpack_from('<i', data, 8)[0]
    assert data[12:20] == bytes.fromhex('5216000000008800'), data[12:20].hex()
    assert data[target + 28:target + 30] == bytes.fromhex('1402'), data[target:target + 32].hex()
    env = dict(os.environ, AGX_EXPERIMENTAL_RET='buffer-pop2')
    result = subprocess.run([AGX_RUN, str(code), '--grid', '1,1,1',
                             '--wg', '32,1,1', '--alloc', '4096=4096',
                             '--dump', f'4096=4:{dump}', '--max-steps', '1000'],
                            check=True, capture_output=True, text=True, env=env)
    actual = json.loads(result.stdout)
    assert actual['instructions'] == 10 and actual['groups'] == 1, actual
    assert dump.read_bytes() == struct.pack('<I', 7), dump.read_bytes().hex()

    scalar_obj = work / 'scalar.o'
    scalar_code = work / 'scalar.bin'
    scalar_input = work / 'scalar-input.bin'
    scalar_dump = work / 'scalar-dump.bin'
    scalar_input.write_bytes(struct.pack('<I', 42) + bytes(4092))
    subprocess.run([LLC, '-mtriple=agx', '-mcpu=agx-g14c',
                    '-agx-experimental-leaf-calls', '-verify-machineinstrs',
                    '-filetype=obj', str(SCALAR_SOURCE), '-o', str(scalar_obj)],
                   check=True)
    subprocess.run([OBJCOPY, '--dump-section', f'.text={scalar_code}',
                    str(scalar_obj), str(work / 'scalar-copy.o')], check=True)
    result = subprocess.run([AGX_RUN, str(scalar_code), '--grid', '1,1,1',
                             '--wg', '32,1,1', '--map', f'8192={scalar_input}',
                             '--dump', f'8192=8:{scalar_dump}', '--max-steps', '1000'],
                            check=True, capture_output=True, text=True, env=env)
    actual = json.loads(result.stdout)
    assert actual['groups'] == 1 and actual['instructions'] > 10, actual
    assert scalar_dump.read_bytes() == struct.pack('<II', 42, 89), scalar_dump.read_bytes().hex()
    print('LLVM leaf calls, scalar ABI, spills and simulator stores: pass')
