#!/usr/bin/env python3
"""Describe the non-faulting undecoded AGX opcodes on the direct DRM path.

The 2026-10-11 scan put 49 undecoded opcodes (33 no-op, 16 ends) in two classes: no-op and ends.
This runner measures, per opcode and operand encoding:

  * instruction length: after the candidate come 8 markers
    `mov_imm r16l..r23l, 56` (bytes `62 xx 38 00`, whose upper half is itself
    `wait 0`). A candidate longer than 2 bytes swallows markers, and any desync
    lands on a whole marker or a `wait`. Layout A has the markers right after
    the candidate; layout B puts one `wait` first. Together they give the
    length up to 34 bytes:
    L=2 (0,0)  4 (1,0)  6 (1,1)  8 (2,1)  10 (2,2) ...
  * register effects: r2..r15 are preloaded with distinct values; r0..r23 are
    stored per lane after the candidate and compared with a reference body
    whose candidate is `wait 0`.
  * whether earlier stores survive (slot 0 is stored before the candidate)
  * granularity (variants): candidate run by half the lanes, by no active
    lanes, jumped over, or only in SIMD group 0 of a 2-group dispatch, each
    compared with real `stop` and `wait`.

The table index is recomputed from SR80 after the candidate (get_sr), so a
clobbered register cannot redirect a store. r1 is restored the same way
before the control suffix.

The 9 faulting opcodes (0x06 0x24 0x34 0x44 0x46 0x54 0x5c 0x64 0x74) and
0x6e are refused. The run stops at the first fault (non-normal completion or
clobbered guard), and an unchanged control must then pass. Gated by
CONFIRM_GPU_FAULTS=1 because new operand bits are untested.
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
import run_drm_compute as drm

REPO = Path(__file__).resolve().parents[1]
SENTINEL = 0xcdcdcdcd
FORBIDDEN = {0x06, 0x24, 0x34, 0x44, 0x46, 0x54, 0x5c, 0x64, 0x74, 0x6e}
NOOP = [op for op in range(3, 128, 4) if op != 0x6f]+[0x18, 0x78]  # 0x6f is decoded by applegpu
ENDS = [0x01, 0x0d, 0x1d, 0x2c, 0x2d, 0x3c, 0x3d, 0x4d, 0x50, 0x5d, 0x60, 0x6c, 0x6d, 0x70, 0x7c, 0x7d]
WAIT = bytes.fromhex('3800')
STOP = bytes.fromhex('8800')
BEFORE, AFTER = 0xb0b0b0b0, 0xafafafaf
DUMP = list(range(24))           # r0..r23 -> slots 1..24
SLOTS = 26                       # 0 before, 1..24 registers, 25 after
MARKERS = list(range(16, 24))
pre = lambda r: 0x5a000000 | r << 8 | r


def encodings(op):
    base = op | 2 << 7 | 4 << 9
    return {'base': base, 'zero': op, 'd6': op | 2 << 7 | 12 << 9, 'dt0': op | 4 << 9, 'dt1': op | 1 << 7 | 4 << 9,
            'dt3': op | 3 << 7 | 4 << 9, 'b15': base | 1 << 15, 'hi': op | 0xff80}


def marker(reg):
    return (0x62 | (2*reg) << 9).to_bytes(2, 'little')+WAIT


def get_sr(selector, reg):
    return (0x72 | 2 << 7 | (2*reg) << 9 | (selector & 63) << 16 | (selector >> 6) << 26).to_bytes(4, 'little')


def build(cand, layout='A', variant='all', stride=32):
    """Return (body, segments): segments names byte ranges for the decode check."""
    mov = agx_loop.chains.mov_imm
    enc = agx_loop.encode
    seg = []
    def add(name, b):
        seg.append((name, bytes(b)))
    add('setup', mov(25, stride)+get_sr(80, 24)+mov(26, BEFORE))
    add('store-before', agx_loop.store(26, 8, 24)+WAIT)
    add('preload', b''.join(mov(r, pre(r)) for r in range(2, 24)))
    tail = (WAIT if layout == 'B' else b'')+b''.join(marker(r) for r in MARKERS)+WAIT*8
    if variant == 'all':
        add('candidate', cand); add('markers', tail)
    else:
        limit = {'half': 16, 'none': 0, 'none-jump': 0, 'group0': 32}[variant]
        add('inner-if', enc(f'if_icmp r0l, ult, r1, {limit}, 1'))
        if variant in ('none-jump', 'group0'):
            n = 6+len(cand)+len(tail)
            add('jump', enc(f'jmp_exec_none pc+{n}'))
        add('candidate', cand); add('markers', tail)
        add('inner-pop', enc('pop_exec r0l, 1'))
    add('reindex', get_sr(80, 24)+enc('iadd r24, r24, r25'))
    out = b''
    for k, r in enumerate(DUMP):
        out += agx_loop.store(r, 8, 24)+enc('iadd r24, r24, r25')
        if k % 8 == 7: out += WAIT
    add('dump', out+mov(26, AFTER)+agx_loop.store(26, 8, 24)+WAIT)
    add('restore', get_sr(80, 1)+agx_loop.chains.mov_imm(2, 0x12345678))
    return b''.join(b for _, b in seg), seg


def timing_body(step, trips, steps):
    """Counted loop (agx_loop's shape) of `steps` copies of `step`; r2 keeps 0x12345678 for the suffix.

    `step` is a no-op candidate in exactly the byte form proven inert (layout B:
    op word + `3800 6240`), so the loop counter r3/r4 and r0l are untouched."""
    mov, enc = agx_loop.chains.mov_imm, agx_loop.encode
    head = mov(3, 0)+mov(4, trips)+mov(2, 0x12345678)+enc('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    loop = step*steps+enc('iadd r3, 1, r3.sx')+enc('while_icmp r0l, ult, r3, r4, 1')
    jump = enc(f'jmp_exec_any pc-{len(loop)}', f'jmp_exec_any pc-{len(loop)}')
    return head+loop+jump+enc('pop_exec r0l, 1')+get_sr(80, 1)


def timing_check(step_len, trips, steps):
    """Emulate the loop with each step replaced by same-length waits: exact trip count and exec restore."""
    body = timing_body(WAIT*(step_len//2), trips, steps)
    r = agx_loop.emulate(body[:-4], 32)  # get_sr r1 is not modelled; it only restores r1
    return r['counts'].get('jmp_exec_any') == trips and r['exec'] == [True]*32 and r['r2'] == [0x12345678]*32


def decode_check(seg):
    """Every non-candidate segment must decode instruction by instruction to its own end."""
    _, g = agx_loop._tools()
    for name, b in seg:
        if name in ('candidate', 'markers'):
            continue
        i = 0
        while i < len(b):
            n = g.opcode_to_number(b[i:]); d = g.get_instruction_descriptor(n)
            if d is None: raise ValueError(f'{name}: undecodable at +{i}')
            i += d.decode_size(n)
        if i != len(b): raise ValueError(f'{name}: decode overruns by {i-len(b)}')
    # markers: each must decode as a 4-byte mov_imm whose upper half decodes as wait
    for r in MARKERS:
        m = marker(r); n = g.opcode_to_number(m); d = g.get_instruction_descriptor(n)
        if d.name != 'mov_imm' or d.decode_size(n) != 4: raise ValueError('marker encoding')
        n2 = g.opcode_to_number(m[2:]+WAIT); d2 = g.get_instruction_descriptor(n2)
        if d2.name != 'wait' or d2.decode_size(n2) != 2: raise ValueError('marker tail')


def exec_check(variant):
    """Emulate the inner exec-mask structure with the candidate as `wait` and stores removed."""
    if variant == 'all':
        return True
    enc = agx_loop.encode
    _, g = agx_loop._tools()
    limit = {'half': 16, 'none': 0, 'none-jump': 0, 'group0': 32}[variant]
    body = enc(f'if_icmp r0l, ult, r1, {limit}, 1')+WAIT
    r = agx_loop.emulate(body, 32, lane_ids=True)
    want_inner = [t < limit for t in range(32)]
    r2 = agx_loop.emulate(body+enc('pop_exec r0l, 1'), 32, lane_ids=True)
    return r['exec'] == want_inner and r2['exec'] == [True]*32 and r2['r0l'] == [0]*32


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['length', 'granularity', 'both', 'timing'], default='both')
    p.add_argument('--trips', type=int, default=2000)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--repeats', type=int, default=8, help='timing: submissions per process')
    p.add_argument('--opcodes', default='all', help="'all' (49 non-faulting) or comma list")
    p.add_argument('--encodings', default='all')
    p.add_argument('--device', default='/dev/dri/renderD128')
    p.add_argument('--gprs', type=int, default=64, help='USC register count in 16-bit halves (Mesa max_reg+1); the body uses r0..r26 = 54 halves')
    p.add_argument('--timeout', type=int, default=20)
    p.add_argument('--output', type=Path)
    p.add_argument('--names', help='regex: run only jobs whose name matches (fullmatch)')
    p.add_argument('--repeat', type=int, default=1, help='run each selected job this many times (separate submissions)')
    p.add_argument('--dry-run', action='store_true', help='build and check every body; no GPU access')
    p.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    a = p.parse_args()
    ops = NOOP+ENDS if a.opcodes == 'all' else [int(o, 0) for o in a.opcodes.split(',')]
    bad = [hex(o) for o in ops if o in FORBIDDEN or o not in NOOP+ENDS]
    if bad:
        sys.exit(f'refusing faulting/unclassified opcodes: {bad}')
    for v in ('half', 'none', 'none-jump', 'group0'):
        if not exec_check(v): sys.exit(f'emulator exec-mask check failed for {v}')

    jobs = []  # (name, body, count, env)
    def job(name, cand, layout='A', variant='all', count=32, env=None):
        body, seg = build(cand, layout, variant, stride=(count+31)//32*32)
        decode_check(seg)
        jobs.append({'name': name, 'body': body, 'count': count, 'env': env or {}, 'variant': variant, 'layout': layout})
    if a.suite in ('length', 'both'):
        for layout in 'AB':
            job(f'ref-wait-{layout}', WAIT, layout)
        for op in ops:
            for en, v in encodings(op).items():
                if a.encodings != 'all' and en not in a.encodings.split(','): continue
                for layout in 'AB':
                    job(f'op-{op:02x}-{en}-{layout}', v.to_bytes(2, 'little'), layout)
    if a.suite in ('granularity', 'both'):
        gops = [o for o in ops if o in ENDS]
        cands = [('wait', WAIT), ('stop', STOP)]+[(f'{o:02x}', encodings(o)['base'].to_bytes(2, 'little')) for o in gops]
        # 0x18/0x78 end only when bit 7 is set (0x98/0xf8, like trap 0x08 -> stop 0x88); compare them with stop
        cands += [(f'{o | 0x80:02x}', (o | 0x80).to_bytes(2, 'little')) for o in ops if o in (0x18, 0x78)]
        for name, cand in cands:
            for v in ('all', 'half', 'none', 'none-jump'):
                job(f'gran-{name}-{v}', cand, 'B', v)
            for local in ('64', '32'):
                job(f'gran-{name}-group0-local{local}', cand, 'B', 'group0', 64, {'AGX_LOCAL_X': local})
    if a.suite == 'timing':
        if not (timing_check(6, 16, 4) and timing_check(2, 16, 4)):
            sys.exit('timing loop failed the emulator gate')
        steps = [('empty', b'', 0), ('wait', WAIT, a.steps), ('wait3', WAIT*3, a.steps),
                 ('mov_imm', agx_loop.chains.mov_imm(5, 0x1234), a.steps)]
        steps += [(f'{o:02x}', encodings(o)['base'].to_bytes(2, 'little')+bytes.fromhex('38006240'), a.steps)
                  for o in ops if o in NOOP and o not in (0x18, 0x78)]
        steps += [(f'{o:02x}', encodings(o)['base'].to_bytes(2, 'little'), a.steps) for o in ops if o in (0x18, 0x78)]
        for name, st, n in steps:
            jobs.append({'name': f'time-{name}', 'body': timing_body(st, a.trips, n), 'count': 32, 'env': {},
                         'variant': 'timing', 'layout': '-', 'repeats': a.repeats, 'n': a.trips*n})
    if a.names:
        import re
        jobs = [j for j in jobs if re.fullmatch(a.names, j['name'])]
    jobs = [j for j in jobs for _ in range(max(1, a.repeat))]
    print(json.dumps({'jobs': len(jobs), 'dry_run': a.dry_run}), flush=True)
    if a.dry_run:
        return 0
    if os.environ.get('CONFIRM_GPU_FAULTS') != '1':
        sys.exit('refusing to run: new operand encodings are untested. Re-run with CONFIRM_GPU_FAULTS=1.')

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = a.output or Path('/scratch/apple-m2-playground/drm-compute/opcode-describe')/stamp
    out.mkdir(parents=True, exist_ok=False)
    exe = out/'drm_compute'
    r = subprocess.run(shlex.split(a.cc)+['-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', str(REPO/'tests/drm_compute.c'), '-o', str(exe)],
                       capture_output=True, timeout=60)
    if r.returncode:
        sys.exit('build failed: '+r.stderr.decode()[-400:])
    state = {'utc': stamp, 'path': 'direct-drm', 'kind': 'opcode-describe', 'suite': a.suite, 'gprs': a.gprs,
             'slots': SLOTS, 'dump_regs': DUMP, 'markers': MARKERS, 'results': [], 'complete': False}
    save = lambda: (out/'record.json').write_text(json.dumps(state)+'\n')

    def run(jb):
        count = jb['count']; rounded = (count+31)//32*32
        code = drm.build_main(jb['body'])
        cpath = out/'code.bin'; cpath.write_bytes(code+bytes(16))
        table = out/'table.bin'; table.write_bytes(struct.pack('<I', SENTINEL)*(SLOTS*rounded))
        rb = out/'readback'; dump = out/'table.out'
        rec = {'name': jb['name'], 'variant': jb['variant'], 'layout': jb['layout'], 'count': count, 'env': jb['env'], 'n': jb.get('n')}
        env = dict(os.environ, **jb['env'], AGX_TABLE_OUT=str(dump))
        try:
            res = subprocess.run([str(exe), a.device, str(cpath), str(count), str(rb), str(jb.get('repeats', 1)), str(a.gprs), str(table)],
                                 capture_output=True, timeout=a.timeout, env=env)
            ev = [json.loads(l) for l in res.stdout.decode(errors='replace').splitlines() if l.startswith('{')]
            smp = [e for e in ev if 'rep' in e]
            rec['returncode'] = res.returncode
            rec['fw_ns'] = [e['fw_ts'][1]-e['fw_ts'][0] for e in smp]
            rec['host_ns'] = [e['host_ns'] for e in smp]
            rec['normal'] = res.returncode == 0 and bool(smp) and all(e['fw_ts'][0] and e['fw_ts'][1] and e['host_ns'] < 50_000_000 for e in smp)
            if res.returncode == 0:
                w = list(struct.unpack(f'<{rounded+32}I', rb.read_bytes()))
                want = [SENTINEL]*(rounded+16)+[0xa5a50000+j for j in range(16)]
                gi = list(range(8))+list(range(8+count, rounded+32))
                rec['guards_ok'] = all(w[i] == want[i] for i in gi)
                rec['payload'] = w[8:8+count]
                t = list(struct.unpack(f'<{SLOTS*rounded}I', dump.read_bytes()))
                rec['table'] = [t[s*rounded:s*rounded+count] for s in range(SLOTS)]
            else:
                rec['guards_ok'] = False; rec['stdout_tail'] = res.stdout.decode(errors='replace')[-300:]
        except subprocess.TimeoutExpired:
            rec.update(returncode=None, timed_out=True, normal=False, guards_ok=False)
        rec['fault'] = not (rec['normal'] and rec['guards_ok'])
        return rec

    try:
        ctrl = run({'name': 'control', 'body': bytes.fromhex('620978563412'), 'count': 32, 'env': {}, 'variant': 'control', 'layout': '-'})
        state['control_before'] = {k: ctrl.get(k) for k in ('normal', 'guards_ok', 'fw_ns')}
        if ctrl['fault'] or any(v != 0x12345678 for v in ctrl['payload']):
            raise RuntimeError('control dispatch failed before the run')
        for k, jb in enumerate(jobs):
            rec = run(jb); state['results'].append(rec)
            if k % 25 == 0 or rec['fault']: save()
            print(json.dumps({'name': rec['name'], 'fault': rec['fault'], 'fw_ns': rec.get('fw_ns'),
                              'p0': hex(rec['payload'][0]) if rec.get('payload') else None}), flush=True)
            if rec['fault']:
                chk = run({'name': 'recovery', 'body': bytes.fromhex('620978563412'), 'count': 32, 'env': {}, 'variant': 'control', 'layout': '-'})
                state['recovery'] = {k2: chk.get(k2) for k2 in ('normal', 'guards_ok', 'fault')}
                raise RuntimeError(f"fault on {rec['name']}; stopping (control after: {'ok' if not chk['fault'] else 'FAILED'})")
        state['complete'] = True
    except Exception as e:
        state['error'] = f'{type(e).__name__}: {e}'
    finally:
        save()
        print(json.dumps({'complete': state['complete'], 'n': len(state['results']), 'output': str(out), 'error': state.get('error')}), flush=True)
    return 0 if state['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
