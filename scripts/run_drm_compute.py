#!/usr/bin/env python3
"""Run native AGX main code through the direct DRM harness (tests/drm_compute.c).

No Mesa/Vulkan in the execution path. The main is the pinned control's own
index/bounds prefix + <body> + address/store/pop_exec/stop suffix; the body
must leave the stored value in r2. Bodies reuse run_agx_timestamps' chain
generator (exact oracles). Firmware start/end timestamps of the compute
command, DRM GET_TIME around submit/wait, and host time are recorded.
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

import agx_loop
import patch_agx_sr as patcher
import run_agx_timestamps as chains

REPO = Path(__file__).resolve().parents[1]


def control_main():
    blob = bytes.fromhex(json.loads((REPO/'novel-results/2026-10-04-first-agx-instruction/patch.json').read_text())['input_hex'])
    main = blob[patcher.CODE_START+0x80:patcher.CODE_START+0xb6]
    site = patcher.SITE-patcher.CODE_START-0x80
    if main[site:site+6] != bytes.fromhex('620978563412') or main[-2:] != bytes.fromhex('8800'):
        raise ValueError('control main layout mismatch')
    return main, site


def build_main(body):
    main, site = control_main()
    return main[:site]+bytes(body)+main[site+6:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=sorted(chains.KINDS)+['control', 'loop', 'chase', 'stream'], default='control')
    p.add_argument('--loop-kind', choices=sorted(chains.KINDS), default='iadd8', help='chain instruction inside --kind loop')
    p.add_argument('--steps', type=int, default=1, help='chain steps / dependent loads per loop iteration (loop, chase)')
    p.add_argument('--trips', type=int, default=1000, help='loop trips for --kind chase (--lengths = footprints in bytes)')
    p.add_argument('--stride', type=int, default=64, help='bytes between chased elements (--kind chase)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--chains', type=int, choices=(1, 2, 4, 8, 16), default=1, help='independent chains (8/16: chase only)')
    p.add_argument('--lengths', default='0')
    p.add_argument('--count', type=int, default=33)
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--gprs', type=int, default=16)
    p.add_argument('--preload', type=lambda v: int(v, 0), default=0x10000000)
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--device', default='/dev/dri/renderD128')
    p.add_argument('--output', type=Path)
    p.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    a = p.parse_args()
    lengths = [None] if a.kind == 'control' else [int(x) for x in a.lengths.split(',')]
    limit = {'loop': 10_000_000, 'chase': 512 << 20, 'stream': 1 << 20}.get(a.kind, 100000)
    if not (1 <= a.repeats <= 64 and 1 <= a.count <= 1 << 24 and all(n is None or 0 <= n <= limit for n in lengths)):
        p.error('bounded repeats/count/lengths required')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = a.output or Path('/scratch/apple-m2-playground/drm-compute/runs')/stamp
    out.mkdir(parents=True, exist_ok=False)
    state = {'passed': False, 'utc': stamp, 'path': 'direct-drm', 'kind': a.kind, 'chains': a.chains,
             'loop_kind': a.loop_kind if a.kind == 'loop' else None, 'steps': a.steps if a.kind == 'loop' else None,
             'trips': a.trips if a.kind == 'chase' else None, 'stride': a.stride if a.kind == 'chase' else None,
             'count': a.count, 'repeats': a.repeats, 'gprs': a.gprs, 'preload': a.preload, 'variants': []}
    save = lambda: (out/'record.json').write_text(json.dumps(state, indent=1)+'\n')
    exe = out/'drm_compute'
    rounded = (a.count+31)//32*32
    try:
        r = subprocess.run(shlex.split(a.cc)+['-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
                           str(REPO/'tests/drm_compute.c'), '-o', str(exe)], capture_output=True, timeout=60)
        if r.returncode: raise RuntimeError('build failed: '+r.stderr.decode()[-400:])
        gpu, _ = patcher.load_model(chains.APPLEGPU)
        for n in lengths:
            if a.chains > 4 and a.kind != 'chase': raise ValueError('8/16 chains are chase-only')
            if a.kind == 'chase' and a.gprs < 2*(4+a.chains): raise ValueError('gprs too small for chase registers')
            if a.kind == 'stream' and a.gprs < 32: raise ValueError('stream uses r1..r14: needs --gprs >= 32')
            if a.kind == 'loop':
                # n = trip count. Gate: emulate the identical structure (small K) before any GPU use.
                body, want = agx_loop.loop_body(n, a.steps, a.loop_kind, a.preload, a.chains)
                small = min(n, 64)
                ok, sbody, _, emu = agx_loop.check(small, a.steps, a.loop_kind, active=min(a.count, 32), preload=a.preload, chains_n=a.chains)
                diff = [i for i, (x, y) in enumerate(zip(body, sbody)) if x != y]
                if not ok or len(body) != len(sbody) or any(not 8 <= i < 12 for i in diff):
                    raise RuntimeError(f'emulator gate failed for trips={n}')
                name = f'loop-{a.loop_kind}-c{a.chains}-s{a.steps}-k{n}'
                state.setdefault('emulator', []).append({'name': name, 'emulated_trips': small, 'executed': emu['executed'],
                                                         'differs_only_in_trip_immediate': True})
            elif a.kind == 'stream':
                # n = trips; S = rounded lane count; table T[w] = w covers every address exactly once.
                S = rounded
                words = S*n*a.steps
                if words*4 > 512 << 20: raise ValueError('stream table exceeds 512 MiB')
                import array
                tpath = out/f'table-{n}.bin'; tpath.write_bytes(array.array('I', range(words)).tobytes())
                body = agx_loop.stream_body(n, a.steps, S)
                want = [agx_loop.stream_oracle(i, n, a.steps, S) for i in range(a.count)]
                ok, sbody, _, emu = agx_loop.check_stream(min(n, 4), a.steps, S, active=min(a.count, 32))
                diff = [i for i, (x, y) in enumerate(zip(body, sbody)) if x != y]
                if not ok or len(body) != len(sbody) or any(not 8 <= i < 12 for i in diff):
                    raise RuntimeError(f'emulator gate failed for stream trips={n}')
                name = f'stream-s{a.steps}-k{n}-lanes{a.count}'
                state.setdefault('emulator', []).append({'name': name, 'emulated_trips': min(n, 4), 'executed': emu['executed'],
                                                         'differs_only_in_trip_immediate': True})
            elif a.kind == 'chase':
                # n = footprint in bytes; one random cycle over n/stride elements.
                if a.stride % 4 or n % a.stride or n < 2*a.stride: raise ValueError('footprint/stride mismatch')
                table = agx_loop.chase_table(n//a.stride, a.stride//4, a.seed)
                tpath = out/f'table-{n}.bin'; tpath.write_bytes(table.tobytes())
                starts = agx_loop.chase_starts(table, a.chains)
                body = agx_loop.chase_body(a.trips, a.steps, starts=starts)
                want = agx_loop.chase_want(table, a.trips*a.steps, starts)
                small = min(a.trips, 16)
                ok, sbody, _, emu = agx_loop.check_chase(small, a.steps, table, active=min(a.count, 32), chains_n=a.chains)
                diff = [i for i, (x, y) in enumerate(zip(body, sbody)) if x != y]
                if not ok or len(body) != len(sbody) or any(not 8 <= i < 12 for i in diff):
                    raise RuntimeError(f'emulator gate failed for footprint={n}')
                name = f'chase-{n}-c{a.chains}-s{a.steps}-k{a.trips}'
                state.setdefault('emulator', []).append({'name': name, 'emulated_trips': small, 'executed': emu['executed'],
                                                         'differs_only_in_trip_immediate': True})
            elif n is None:
                name, body, want = 'control', bytes.fromhex('620978563412'), 0x12345678
            else:
                name, body, want = f'{a.kind}-c{a.chains}-{n}', chains.body(n, a.kind, a.chains, a.preload, gpu), chains.oracle(n, a.kind, a.preload)
            code = build_main(body)
            path = out/(name+'.bin'); path.write_bytes(code+bytes(16)); rb = out/(name+'.readback')
            rec = {'name': name, 'n': n, 'code_bytes': len(code), 'oracle': want if not isinstance(want, list) else 'per-lane'}
            if a.kind == 'stream': rec.update({'bytes_loaded': words*4, 'loads_per_lane': n*a.steps, 'stride_words': S})
            if a.kind == 'chase': rec.update({'footprint': n, 'stride': a.stride, 'hops': a.trips*a.steps, 'chains': a.chains, 'loads': a.trips*a.steps*a.chains, 'seed': a.seed})
            state['variants'].append(rec)
            argv = [str(exe), a.device, str(path), str(a.count), str(rb), str(a.repeats), str(a.gprs)]
            if a.kind in ('chase', 'stream'): argv.append(str(tpath))
            try:
                res = subprocess.run(argv, capture_output=True, timeout=a.timeout)
                rec['returncode'] = res.returncode; stdout = res.stdout.decode(errors='replace')
            except subprocess.TimeoutExpired as e:
                rec['returncode'] = None; rec['timed_out'] = True; stdout = (e.stdout or b'').decode(errors='replace')
            events = [json.loads(l) for l in stdout.splitlines() if l.startswith('{')]
            rec['setup'] = [e for e in events if 'params' in e or 'usc_words_hex' in e or 'cdm_hex' in e]
            rec['samples'] = [e for e in events if 'rep' in e]
            if rec['returncode'] != 0:
                rec['stdout_tail'] = stdout[-800:]; save(); raise RuntimeError('submission failed: '+name)
            w = list(struct.unpack('<'+'I'*(rounded+32), rb.read_bytes()))
            expected = [0xcdcdcdcd]*(rounded+16)+[0xa5a50000+j for j in range(16)]
            expected[8:8+a.count] = want if isinstance(want, list) else [want]*a.count
            rec['buffer_ok'] = w == expected
            rec['mismatches'] = [[i, w[i]] for i in range(len(w)) if w[i] != expected[i]][:16]
            rb.unlink(); save()
            if a.kind in ('chase', 'stream'): tpath.unlink()
            print(json.dumps({'variant': name, 'ok': rec['buffer_ok'],
                              'fw_ns': [s['fw_ts'][1]-s['fw_ts'][0] for s in rec['samples']],
                              'host_ns': [s['host_ns'] for s in rec['samples']]}), flush=True)
            if not rec['buffer_ok']: raise RuntimeError('oracle mismatch: '+name)
        state['passed'] = True
    except Exception as e:
        state['error'] = f'{type(e).__name__}: {e}'
    finally:
        save(); print(json.dumps({'passed': state['passed'], 'output': str(out), 'error': state.get('error')}), flush=True)
    return 0 if state['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
