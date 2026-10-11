#!/usr/bin/env python3
"""Validate Vulkan command timestamps against native AGX work of known length.

Each variant replaces only the pinned control's `mov_imm r2` with

    mov_imm r2, PRELOAD ; <N copies of a fixed body instruction>

and the unchanged suffix stores r2. The dependent chain `iadd r2, 1, r2`
has an exact oracle (PRELOAD + N mod 2^32) for every active lane, so a
timestamp delta is only accepted for a dispatch whose results are proven.

Every dispatch is bracketed by four timestamps (top-of-pipe, compute before,
compute after, bottom-of-pipe) and host CLOCK_MONOTONIC around submit+wait.
Uses only instructions already executed in this project (mov_imm, iadd, wait).
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys

import patch_agx_sr as patcher
import run_vulkan_compute as baseline

REPO = Path(__file__).resolve().parents[1]
APPLEGPU = Path('/save/workspace/AGX_Probe/applegpu')


def mov_imm(reg, value):
    """mov_imm rN, imm32 — the control's own `6209` form with D = 2*reg."""
    return (0x62 | 2 << 7 | (2*reg) << 9).to_bytes(2, 'little')+value.to_bytes(4, 'little')


KINDS = {
    # name: (bytes for r2, width, per-step increment applied to the oracle)
    'iadd8': 8, 'fadd6': 6, 'fadd4': 4,
}


def encode_step(kind, reg, gpu):
    if kind == 'iadd8':
        raw = int.from_bytes(bytes.fromhex('0e09014064000000'), 'little')
        raw = raw & ~(0x3f << 9) & ~(0x3f << 28) | (2*reg) << 9 | (2*reg) << 28   # D bits 9..14, B bits 28..33
        size = 8
    else:
        raw = int.from_bytes(bytes.fromhex('2a094482'), 'little')                 # fadd32 r2, r2, 0.125
        raw = raw & ~(0x3f << 9) & ~(0x3f << 16) | (2*reg) << 9 | (2*reg) << 16    # D bits 9..14, A bits 16..21
        size = 4
        if kind == 'fadd6': raw |= 1 << 15; size = 6                             # same fields, long form
    b = raw.to_bytes(size, 'little')
    n = gpu.opcode_to_number(b); d = gpu.get_instruction_descriptor(n)
    text = str(d.disassemble(n)).split()
    want = (['iadd', f'r{reg},', '1,', f'r{reg}.sx'] if kind == 'iadd8' else ['fadd32', f'r{reg},', f'r{reg},', '0.125'])
    if d.decode_size(n) != size or text != want:
        raise ValueError(f'encoding check failed for {kind} r{reg}: {text}')
    return b


def fold(kind, reg, gpu):
    """r2 = r2 + rN, decoded back before use."""
    sys.path.insert(0, str(APPLEGPU)); import assemble
    b = bytes(assemble.assemble_line(f'iadd r2, r2, r{reg}' if kind == 'iadd8' else f'fadd32 r2, r2, r{reg}'))
    n = gpu.opcode_to_number(b); d = gpu.get_instruction_descriptor(n)
    if d.decode_size(n) != len(b): raise ValueError('fold encoding')
    return b


def body(n, kind, chains, preload, gpu):
    """Preload chain registers r2..r(1+chains), n chain steps round-robin, fold into r2."""
    regs = list(range(2, 2+chains))
    start = preload if kind == 'iadd8' else 0
    out = b''.join(mov_imm(r, start if r == 2 else 0) for r in regs)
    steps = [encode_step(kind, r, gpu) for r in regs]
    out += b''.join(steps[k % chains] for k in range(n))
    out += b''.join(fold(kind, r, gpu) for r in regs[1:])
    return out


def oracle(n, kind, preload):
    if kind == 'iadd8': return (preload+n) & 0xffffffff
    return struct.unpack('<I', struct.pack('<f', n*0.125))[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lengths', default='0,256,1024,4096,16384,65536')
    p.add_argument('--kind', choices=sorted(KINDS), default='iadd8')
    p.add_argument('--chains', type=int, choices=(1, 2, 4), default=1, help='independent register chains')
    p.add_argument('--count', type=int, default=32, help='active lanes (32 = one SIMD group)')
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--preload', type=lambda v: int(v, 0), default=0x10000000)
    p.add_argument('--timeout', type=int, default=30)
    p.add_argument('--output', type=Path)
    p.add_argument('--runtime', type=Path, default=Path('/scratch/apple-m2-playground/vulkan-runtime/usr/lib64'))
    p.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    p.add_argument('--include', default='/save/workspace/AGX_Probe/mesa/include')
    a = p.parse_args()
    lengths = [int(x) for x in a.lengths.split(',')]
    if not (1 <= a.repeats <= 64 and 1 <= a.count <= 65536 and all(0 <= n <= 100000 for n in lengths)):
        p.error('bounded repeats/count/lengths required')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = a.output or Path('/scratch/apple-m2-playground/timestamp-runs')/stamp
    out.mkdir(parents=True, exist_ok=False)
    state = {'passed': False, 'utc': stamp, 'count': a.count, 'repeats': a.repeats, 'preload': a.preload, 'kind': a.kind, 'chains': a.chains, 'variants': []}
    save = lambda: (out/'record.json').write_text(json.dumps(state, indent=1)+'\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('VK_', 'MESA_', 'ASAHI_', 'AGX_', 'HK_', 'LIBGL_', 'GALLIUM_', 'DRI_', 'LD_'))}
    child = env | {'VK_DRIVER_FILES': str(baseline.ICD), 'VK_ICD_FILENAMES': str(baseline.ICD),
                   'VK_LOADER_LAYERS_DISABLE': '~implicit~', 'LD_LIBRARY_PATH': str(a.runtime.resolve()),
                   'MESA_SHADER_CACHE_DISABLE': 'true'}
    exe = out/'vulkan_timestamp'
    rounded = (a.count+31)//32*32
    try:
        if baseline.sha(baseline.DRIVER) != baseline.DRIVER_HASH: raise ValueError('driver hash changed')
        r = subprocess.run(shlex.split(a.cc)+['-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', '-I'+a.include,
                           str(REPO/'tests/vulkan_timestamp.c'), '-ldl', '-o', str(exe)], env=env, capture_output=True, timeout=60)
        if r.returncode: raise RuntimeError('build failed: '+r.stderr.decode()[-400:])
        blob = bytes.fromhex(json.loads((REPO/'novel-results/2026-10-04-first-agx-instruction/patch.json').read_text())['input_hex'])
        gpu, _ = patcher.load_model(APPLEGPU)
        variants = [('unchanged', None, blob)]
        for n in lengths:
            if a.kind != 'iadd8' and n > 1 << 24: raise ValueError('float oracle exact only to 2^24 steps')
            patched, ev = patcher.patch_body(blob, body(n, a.kind, a.chains, a.preload, gpu), gprs=16)
            variants.append((f'chain-{n}', n, patched))
        for name, n, b in variants:
            path = out/(name+'.bin'); path.write_bytes(b); rb = out/(name+'.readback')
            argv = [str(exe), 'import', str(path), str(a.count), '/dev/dri/renderD128', str(rb), str(a.repeats)]
            rec = {'name': name, 'n': n, 'code_bytes': len(b)}
            state['variants'].append(rec)
            try:
                res = subprocess.run(argv, env=child, capture_output=True, timeout=a.timeout)
                rec['returncode'] = res.returncode; stdout = res.stdout.decode(errors='replace')
            except subprocess.TimeoutExpired as e:
                rec['returncode'] = None; rec['timed_out'] = True; stdout = (e.stdout or b'').decode(errors='replace')
            events = [json.loads(l) for l in stdout.splitlines() if l.startswith('{')]
            rec['limits'] = [e for e in events if 'timestamp_limits' in e or 'queue' in e]
            rec['samples'] = [e for e in events if 'rep' in e]
            if rec['returncode'] != 0:
                rec['stdout_tail'] = stdout[-800:]+res.stderr.decode(errors='replace')[-400:] if rec['returncode'] is not None else stdout[-800:]
                save(); raise RuntimeError('dispatch failed: '+name)
            w = list(struct.unpack('<'+'I'*(rounded+32), rb.read_bytes()))
            initial = [0xcdcdcdcd]*(rounded+16)+[0xa5a50000+j for j in range(16)]
            want = 0x12345678 if n is None else oracle(n, a.kind, a.preload)
            rec['oracle'] = want
            rec['values_ok'] = w[8:8+a.count] == [want]*a.count
            rec['guards_ok'] = w[:8] == initial[:8] and w[8+a.count:] == initial[8+a.count:]
            rec['all_available'] = all(all(s['avail']) for s in rec['samples'])
            rb.unlink(); path.unlink(); save()
            d = [s['ts'][2]-s['ts'][1] for s in rec['samples']]
            print(json.dumps({'variant': name, 'ok': rec['values_ok'] and rec['guards_ok'],
                              'compute_delta_ns': d, 'host_ns': [s['host_ns'] for s in rec['samples']]}), flush=True)
            if not (rec['values_ok'] and rec['guards_ok']): raise RuntimeError('oracle mismatch: '+name)
        state['passed'] = True
    except Exception as e:
        state['error'] = f'{type(e).__name__}: {e}'
    finally:
        save(); print(json.dumps({'passed': state['passed'], 'output': str(out), 'error': state.get('error')}), flush=True)
    return 0 if state['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
