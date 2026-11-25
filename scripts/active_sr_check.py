#!/usr/bin/env python3
"""Offline: sr58 (active_thread_index_in_simdgroup) / sr59 (active threads in the SIMD group) under divergence.

results/2026-11-22-active-sr: two kernels (mk.py) store sr58 / sr59 from the lanes that pass `tid < u2` or
`(tid & u3) == u2` (64 threads, 2 SIMD groups; word 4*i = sr58 of lane i, word 4*(64+i) = sr59). hw/ holds the
M2 Max readback (agx_run_hw, 2026-11-22). Each case runs in agx_run and must match byte for byte.
"""
import os, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'results/2026-11-22-active-sr'
RUN = os.environ.get('AGX_RUN', str(ROOT/'rust/target/release/agx_run'))
CASES = {'lt1': ('lt', 1, 0), 'lt5': ('lt', 5, 0), 'lt32': ('lt', 32, 0), 'lt40': ('lt', 40, 0), 'lt64': ('lt', 64, 0),
         'odd': ('msk', 1, 1), 'm8': ('msk', 3, 7), 'l31': ('msk', 31, 31)}
bad = 0
with tempfile.TemporaryDirectory() as t:
    for n, (k, u2, u3) in CASES.items():
        out = Path(t)/f'{n}.bin'
        subprocess.run([RUN, str(D/f'{k}.bin'), '--grid', '64', '--wg', '64', '--regs', '16', '--map', f'0x100000={D/"init.bin"}',
                        '--uniform', '0=0x100000', '--uniform', '1=0', '--uniform', f'2={u2}', '--uniform', f'3={u3}',
                        '--dump', f'0x100000=2048:{out}'], check=True, capture_output=True)
        ok = out.read_bytes() == (D/'hw'/f'{n}.bin').read_bytes()
        bad += not ok
        print(('ok  ' if ok else 'FAIL'), n)
print(f'{len(CASES)} cases, {bad} failures'); sys.exit(bad != 0)
