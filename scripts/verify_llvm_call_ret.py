#!/usr/bin/env python3
"""Check that LLVM's G14C call/ret encodings execute in the measured simulator mode.

This is an offline cross-project smoke test. It does not validate an LLVM
calling convention or call timing; those are still absent from code generation.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LLVM_MC = os.environ.get('LLVM_MC', 'llvm-mc')
LLVM_OBJCOPY = os.environ.get('LLVM_OBJCOPY', 'llvm-objcopy')
AGX_RUN = os.environ.get('AGX_RUN', str(ROOT / 'rust/target/release/agx_run'))
SOURCE = 'call callee\npop_exec r0l, 2\nstop\ncallee:\nret r1\n'
EXPECTED = bytes.fromhex('10c00e00000052160000000088001402')

with tempfile.TemporaryDirectory() as temp:
    work = Path(temp)
    obj, code = work / 'call.o', work / 'call.bin'
    subprocess.run([LLVM_MC, '-triple=agx', '-mcpu=agx-g14c', '-filetype=obj',
                    '-o', str(obj)], input=SOURCE, text=True, check=True)
    subprocess.run([LLVM_OBJCOPY, '--dump-section', f'.text={code}',
                    str(obj), str(work / 'copy.o')], check=True)
    assert code.read_bytes() == EXPECTED, code.read_bytes().hex()
    env = dict(os.environ, AGX_EXPERIMENTAL_RET='buffer-pop2')
    result = subprocess.run([AGX_RUN, str(code), '--grid', '1,1,1',
                             '--wg', '32,1,1', '--max-steps', '1000'],
                            check=True, capture_output=True, text=True, env=env)
    actual = json.loads(result.stdout)
    assert actual['instructions'] == 4 and actual['groups'] == 1, actual
    print('LLVM direct call/ret bytes and simulator control flow: pass')
