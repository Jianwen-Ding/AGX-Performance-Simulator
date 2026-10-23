#!/usr/bin/env python3
"""Compile LLVM's opt-in call fixtures and replay their code in AGXsim."""

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/experimental-leaf-call.ll'
SCALAR_SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/experimental-leaf-call-scalar.ll'
AGGREGATE_SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/experimental-call-aggregate.ll'
MASKED_SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/experimental-call-masked.ll'
FRAMES_SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/experimental-call-frames.ll'
NESTED_SOURCE = ROOT.parent / 'llvm-project/llvm/test/CodeGen/AGX/experimental-nested-call.ll'
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
    push = data.find(bytes.fromhex('42100000000010c0'))
    assert push >= 0, data[:32].hex()
    target = push + 6 + struct.unpack_from('<i', data, push + 8)[0]
    assert data[push + 12:push + 20] == bytes.fromhex('5216000000008800'), data[push + 12:push + 20].hex()
    assert data.find(bytes.fromhex('1402'), target, target + 64) >= 0, data[target:target + 64].hex()
    env = dict(os.environ, AGX_EXPERIMENTAL_RET='buffer-pop2')
    result = subprocess.run([AGX_RUN, str(code), '--grid', '1,1,1',
                             '--wg', '32,1,1', '--alloc', '4096=4096',
                             '--dump', f'4096=4:{dump}', '--max-steps', '1000'],
                            check=True, capture_output=True, text=True, env=env)
    actual = json.loads(result.stdout)
    assert actual['instructions'] >= 10 and actual['groups'] == 1, actual
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

    for name, source, grid, expected in [
        ('aggregate', AGGREGATE_SOURCE, 1, [15, 23, 31]),
        ('masked', MASKED_SOURCE, 32,
         [i + 5 if i < 8 else i * 2 if i < 16 else i + 100 for i in range(32)]),
    ]:
        fixture_obj = work / f'{name}.o'
        fixture_code = work / f'{name}.bin'
        fixture_dump = work / f'{name}-dump.bin'
        subprocess.run([LLC, '-mtriple=agx', '-mcpu=agx-g14c',
                        '-agx-experimental-leaf-calls', '-verify-machineinstrs',
                        '-filetype=obj', str(source), '-o', str(fixture_obj)],
                       check=True)
        subprocess.run([OBJCOPY, '--dump-section', f'.text={fixture_code}',
                        str(fixture_obj), str(work / f'{name}-copy.o')], check=True)
        result = subprocess.run([AGX_RUN, str(fixture_code), '--grid', f'{grid},1,1',
                                 '--wg', '32,1,1', '--alloc', '8192=4096',
                                 '--dump', f'8192={4 * len(expected)}:{fixture_dump}',
                                 '--max-steps', '1000'],
                                check=True, capture_output=True, text=True, env=env)
        actual = json.loads(result.stdout)
        assert actual['groups'] == 1 and actual['instructions'] > 10, actual
        assert fixture_dump.read_bytes() == struct.pack(f'<{len(expected)}I', *expected), (name, fixture_dump.read_bytes().hex())

    frames_obj = work / 'frames.o'
    frames_code = work / 'frames.bin'
    frames_dump = work / 'frames-dump.bin'
    subprocess.run([LLC, '-mtriple=agx', '-mcpu=agx-g14c',
                    '-agx-experimental-leaf-calls', '-verify-machineinstrs',
                    '-filetype=obj', str(FRAMES_SOURCE), '-o', str(frames_obj)],
                   check=True)
    subprocess.run([OBJCOPY, '--dump-section', f'.text={frames_code}',
                    str(frames_obj), str(work / 'frames-copy.o')], check=True)
    result = subprocess.run([AGX_RUN, str(frames_code), '--grid', '32,1,1',
                             '--wg', '32,1,1', '--map', f'8192={scalar_input}',
                             '--dump', f'8192=8:{frames_dump}', '--max-steps', '1000'],
                            check=True, capture_output=True, text=True, env=env)
    actual = json.loads(result.stdout)
    assert actual['groups'] == 1 and actual['instructions'] > 10, actual
    assert frames_dump.read_bytes() == struct.pack('<II', 42, 127), frames_dump.read_bytes().hex()

    nested_obj = work / 'nested.o'
    nested_code = work / 'nested.bin'
    nested_dump = work / 'nested-dump.bin'
    subprocess.run([LLC, '-mtriple=agx', '-mcpu=agx-g14c',
                    '-agx-experimental-leaf-calls', '-verify-machineinstrs',
                    '-filetype=obj', str(NESTED_SOURCE), '-o', str(nested_obj)],
                   check=True)
    subprocess.run([OBJCOPY, '--dump-section', f'.text={nested_code}',
                    str(nested_obj), str(work / 'nested-copy.o')], check=True)
    result = subprocess.run([AGX_RUN, str(nested_code), '--grid', '1,1,1',
                             '--wg', '32,1,1', '--alloc', '8192=4096',
                             '--dump', f'8192=4:{nested_dump}', '--max-steps', '1000'],
                            check=True, capture_output=True, text=True, env=env)
    actual = json.loads(result.stdout)
    assert actual['groups'] == 1 and actual['instructions'] > 10, actual
    assert nested_dump.read_bytes() == struct.pack('<I', 13), nested_dump.read_bytes().hex()
    print('LLVM calls, aggregates, masked control, nested links and separate simulator frames: pass')
