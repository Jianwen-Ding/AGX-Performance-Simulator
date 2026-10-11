#!/usr/bin/env python3
"""What 0x18/0x78 wait on, and how the SIMD-group-ending opcodes differ (direct DRM).

Suites (each compares candidates against known references in the same run):

  barrier   One workgroup of 2 or 4 SIMD groups. Group g first runs a counted
            delay loop of 1 + g*K trips, then stores flag g+1 into table row 0.
            Every group then executes the candidate exactly once (no group can
            skip it, so a real barrier cannot deadlock), and loads the flag of
            the last (slowest) group's lane. A threadgroup barrier makes group 0
            read that flag; a no-op reads the sentinel.
  memwait   One SIMD group, counted loop: <memory op> ; candidate ; 64-step
            iadd chain ; wait 0. If the candidate waits for the memory op, the
            chain no longer overlaps it and the loop gets slower by about the
            memory latency. References: none, wait 0, memory/device barriers.
  endcost   65,536 threads (2,048 one-SIMD workgroups), body = candidate only:
            per-group cost of ending via each opcode versus stop.
  endpend   16 stores (no waits) and one pending load, then the candidate: do
            all earlier stores reach memory?
  endfollow Lanes 0-15 run the candidate, followed by 16 x stop. An opcode that
            ends by itself ends all 32 lanes; one that merely skips ahead lands
            on stop, which ends only lanes 0-15.

The faulting opcodes are refused. Stops at the first fault (non-normal
completion or clobbered guard) after an unchanged control; CONFIRM_GPU_FAULTS=1.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

import agx_loop
import patch_agx_sr as patcher
import run_drm_compute as drm
from run_drm_opcode_describe import FORBIDDEN, ENDS, WAIT, STOP, get_sr

REPO = Path(__file__).resolve().parents[1]
SENTINEL = 0xcdcdcdcd
mov, enc = agx_loop.chains.mov_imm, agx_loop.encode
H = bytes.fromhex
REFS = {'none': b'', 'wait': WAIT, 'tg_barrier': H('6800'), 'mem_barrier': H('f596'),
        'device_barrier_1': H('f59d'), 'device_barrier_2': H('f59e'), 'mem+tg_barrier': H('f5966800')}
CANDS = {'18': H('1809'), '78': H('7809'), '18z': H('1800'), '78z': H('7800'),
         'mem+18': H('f5961809'), 'mem+78': H('f5967809')}
ENDERS = {'stop': STOP, **{f'{o:02x}': (o | 2 << 7 | 4 << 9).to_bytes(2, 'little') for o in ENDS},
          '98': H('9800'), 'f8': H('f800')}
GPRS = 64   # 16-bit halves; bodies use at most r0..r24


def delay_loop(trips_reg_setup):
    """Counted loop (agx_loop shape) whose trip count r4 is computed by `trips_reg_setup`."""
    _, g = agx_loop._tools()
    gpu, _ = patcher.load_model(agx_loop.APPLEGPU)
    head = mov(3, 0)+trips_reg_setup+mov(2, 0)+enc('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    loop = b''.join(agx_loop.chains.encode_step('iadd8', 2, gpu) for _ in range(4))
    loop += enc('iadd r3, 1, r3.sx')+enc('while_icmp r0l, ult, r3, r4, 1')
    return head+loop+enc(f'jmp_exec_any pc-{len(loop)}', f'jmp_exec_any pc-{len(loop)}')+enc('pop_exec r0l, 1')


def finish():
    return get_sr(80, 1)+mov(2, 0x12345678)


def barrier_body(cand, groups, K, count):
    row = 2*count   # table rows: [0, row) flags (reads past `count` hit sentinel), [row, 2*row) results
    trips = get_sr(53, 6)+mov(7, K)+mov(8, 1)+enc('imadd r4, r6, r7, r8')
    b = get_sr(80, 24)+delay_loop(trips)
    b += enc('iadd r12, r6, 1')+agx_loop.store(12, 8, 24)            # flag[gid] = group+1
    b += cand
    b += enc(f'iadd r13, r24, {32*(groups-1)}')+enc('device_load 0, i32, x, r15, u8_u9, r13, unsigned')+WAIT
    b += mov(11, row)+enc('iadd r16, r24, r11')+agx_loop.store(15, 8, 16)+WAIT
    return b+finish()


SHARED_4K = 0x4d | 1 << 8 | 0x24 << 10 | (4096 // 256) << 24   # USC Shared: uses shared memory, vertex/compute layout, 4 KiB


def tg_mem(load, reg, idx_reg):
    """threadgroup_load/store i32, x, r<reg>, base 0, index r<idx_reg>l, packed as Mesa agx_pack.c:769-792
    (8-byte form); decode must round-trip to the expected text. The index steps one 16-byte
    4-component slot (byte = 16*index for i32; see run_drm_tg_layout.py)."""
    _, g = agx_loop._tools()
    idx16 = 2*idx_reg
    raw = ((0x69 if load else 0x29) | 1 << 8 | ((2*reg) & 63) << 9 | 1 << 15 | 0b10 << 22 | 2 << 24
           | (idx16 & 63) << 28 | 1 << 36 | (idx16 >> 6) << 48)
    b = raw.to_bytes(8, 'little'); n = g.opcode_to_number(b); d = g.get_instruction_descriptor(n)
    want = f"threadgroup_{'load' if load else 'store'} i32, x, r{reg}, 0, r{idx_reg}l"
    if d is None or d.decode_size(n) != 8 or ' '.join(str(d.disassemble(n)).split()) != want:
        raise ValueError('threadgroup memory encoding mismatch')
    return b


def tgcheck_body(groups):
    """Lane i stores 0x7e000000+i to threadgroup memory, barrier, reads lane (32*groups-1-i)."""
    n = 32*groups
    b = get_sr(80, 24)+mov(9, 0x7e000000)+enc('iadd r6, r9, r24')+tg_mem(False, 6, 24)
    b += H('6800')+mov(10, 0xffffffff)+mov(11, n-1)+enc('imadd r23, r24, r10, r11')
    b += tg_mem(True, 5, 23)+WAIT+agx_loop.store(5, 8, 24)+WAIT
    return b+finish()


def barrier_tg_body(cand, groups, K, count):
    """barrier_body with the flag exchanged through threadgroup memory instead of device memory."""
    trips = get_sr(53, 6)+mov(7, K)+mov(8, 1)+enc('imadd r4, r6, r7, r8')
    b = get_sr(80, 24)+mov(12, 0xcdcdcdcd)+tg_mem(False, 12, 24)+enc(f'iadd r13, r24, {32*groups}')+tg_mem(False, 12, 13)
    b += H('6800')   # every slot starts as the sentinel before any group delays
    b += delay_loop(trips)+enc('iadd r12, r6, 1')+tg_mem(False, 12, 24)
    b += cand
    b += enc(f'iadd r13, r24, {32*(groups-1)}')+tg_mem(True, 15, 13)+WAIT
    b += mov(11, 2*count)+enc('iadd r16, r24, r11')+agx_loop.store(15, 8, 16)+WAIT
    return b+finish()


def memwait_body(mem, cand, trips, chain=64):
    gpu, _ = patcher.load_model(agx_loop.APPLEGPU)
    memop = {'load': enc('device_load 0, i32, x, r5, u8_u9, r24, unsigned'),
             'store': agx_loop.store(6, 8, 24), 'none': b'',
             'atomic': agx_loop.atomic_add(8, 24, 6) if mem == 'atomic' else b'',
             'tgload': tg_mem(True, 5, 24) if mem == 'tgload' else b'', 'tgstore': tg_mem(False, 6, 24) if mem == 'tgstore' else b''}[mem]
    head = get_sr(80, 24)+mov(3, 0)+mov(4, trips)+mov(6, 0)+enc('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    loop = memop+cand+b''.join(agx_loop.chains.encode_step('iadd8', 6, gpu) for _ in range(chain))+WAIT
    loop += enc('iadd r3, 1, r3.sx')+enc('while_icmp r0l, ult, r3, r4, 1')
    tail = get_sr(20, 9)+mov(11, 256)+enc('iadd r10, r24, r11')+agx_loop.store(9, 8, 10)+WAIT   # table[32+gid] = SR20 core id
    return head+loop+enc(f'jmp_exec_any pc-{len(loop)}', f'jmp_exec_any pc-{len(loop)}')+enc('pop_exec r0l, 1')+WAIT+tail+finish()


def endpend_body(cand):
    b = get_sr(80, 24)+mov(25, 32)+enc('iadd r23, r24, 0')
    for k in range(16):
        b += mov(9, 0xe0000000 | k << 16)+enc('iadd r10, r9, r24')+agx_loop.store(10, 8, 23)+enc('iadd r23, r23, r25')
    b += enc('device_load 0, i32, x, r5, u8_u9, r24, unsigned')   # left pending
    return b+cand+WAIT*2+finish()


def endfollow_body(cand):
    b = get_sr(80, 24)+enc('if_icmp r0l, ult, r1, 16, 1')+cand+STOP*16+enc('pop_exec r0l, 1')
    return b+mov(9, 0xafafafaf)+agx_loop.store(9, 8, 24)+WAIT+finish()


def gates():
    """Emulate the loop skeletons with the candidate removed: trip counts and exec masks."""
    r = agx_loop.emulate(delay_loop(mov(4, 5)), 32)
    ok = r['counts'].get('jmp_exec_any') == 5 and r['exec'] == [True]*32
    body = mov(3, 0)+mov(4, 7)+enc('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    gpu, _ = patcher.load_model(agx_loop.APPLEGPU)
    loop = WAIT*4+b''.join(agx_loop.chains.encode_step('iadd8', 6, gpu) for _ in range(4))+WAIT
    loop += enc('iadd r3, 1, r3.sx')+enc('while_icmp r0l, ult, r3, r4, 1')
    body += loop+enc(f'jmp_exec_any pc-{len(loop)}', f'jmp_exec_any pc-{len(loop)}')+enc('pop_exec r0l, 1')
    r = agx_loop.emulate(body, 32)
    ok &= r['counts'].get('jmp_exec_any') == 7 and r['exec'] == [True]*32
    r = agx_loop.emulate(enc('if_icmp r0l, ult, r1, 16, 1')+WAIT+enc('pop_exec r0l, 1'), 32, lane_ids=True)
    return ok and r['exec'] == [True]*32


def decodes(body, skip):
    """Every byte outside the candidate ranges decodes; `skip` = set of byte strings allowed undecoded."""
    _, g = agx_loop._tools()
    i = 0
    while i < len(body):
        for s in skip:
            if s and body.startswith(s, i):
                i += len(s); break
        else:
            n = g.opcode_to_number(body[i:]); d = g.get_instruction_descriptor(n)
            if d is None: raise ValueError(f'undecodable at {i}: {body[i:i+4].hex()}')
            i += d.decode_size(n)
    if i != len(body): raise ValueError('decode overrun')


def jobs_for(suite, a):
    jobs = []
    unknown = set(CANDS.values()) | set(ENDERS.values())
    def add(name, body, count, env=None, repeats=1, table_words=0):
        decodes(body, unknown)
        jobs.append(dict(name=name, body=body, count=count, env=env or {}, repeats=repeats, table_words=table_words))
    if suite == 'barrier':
        for groups in (2, 4):
            count = 32*groups
            for name, c in {**{k: REFS[k] for k in ('wait', 'tg_barrier', 'mem_barrier', 'mem+tg_barrier')}, **CANDS}.items():
                add(f'barrier-g{groups}-{name}', barrier_body(c, groups, a.k, count), count,
                    {'AGX_LOCAL_X': str(count)}, 1, 4*count)
        for groups in (2, 4):
            count = 32*groups
            for name, c in [('wait', WAIT), ('tg_barrier', REFS['tg_barrier']), ('18', CANDS['18']), ('78', CANDS['78']), ('78z', CANDS['78z'])]:
                add(f'barrier-tg-g{groups}-{name}', barrier_tg_body(c, groups, a.k, count), count,
                    {'AGX_LOCAL_X': str(count), 'AGX_SHARED_WORD': str(SHARED_4K)}, 1, 4*count)
    elif suite == 'memwait':
        # The GPU clock state changes over time (2.2x), so every candidate is bracketed by the
        # same-memory-op reference ('none' candidate) and judged against both neighbours.
        for mem in ('none', 'load', 'store'):
            ref = memwait_body(mem, b'', a.trips)
            for name, c in {**REFS, **{k: CANDS[k] for k in ('18', '78')}}.items():
                if name in ('mem+tg_barrier', 'none'): continue
                add(f'memwait-{mem}-ref', ref, 32, None, a.repeats, 512)
                add(f'memwait-{mem}-{name}', memwait_body(mem, c, a.trips), 32, None, a.repeats, 512)
            add(f'memwait-{mem}-ref', ref, 32, None, a.repeats, 512)
    elif suite == 'scope':
        # Every SIMD group of one workgroup runs the same loop (uniform: no group skips the
        # candidate). A workgroup barrier's cost grows with the group count; a SIMD-local op's does not.
        for groups in (1, 2, 4, 8):
            n = 32*groups
            ref = memwait_body('none', b'', a.trips)
            for name, c in [('wait', WAIT), ('tg_barrier', REFS['tg_barrier']), ('18', CANDS['18']), ('78', CANDS['78'])]:
                add(f'scope-g{groups}-ref', ref, n, {'AGX_LOCAL_X': str(n)}, a.repeats, 512)
                add(f'scope-g{groups}-{name}', memwait_body('none', c, a.trips), n, {'AGX_LOCAL_X': str(n)}, a.repeats, 512)
            add(f'scope-g{groups}-ref', ref, n, {'AGX_LOCAL_X': str(n)}, a.repeats, 512)
        # same 8 or 64 SIMD groups as one-group workgroups (spread over cores)
        for wgs in (8, 64, 256):
            n = 32*wgs
            ref = memwait_body('none', b'', a.trips)
            add(f'scope-w{wgs}-ref', ref, n, None, a.repeats, 512+n)
            for name, c in [('wait', WAIT), ('tg_barrier', REFS['tg_barrier']), ('18', CANDS['18']), ('78', CANDS['78'])]:
                add(f'scope-w{wgs}-{name}', memwait_body('none', c, a.trips), n, None, a.repeats, 512+n)
    elif suite == 'tgcheck':
        for groups in (1, 2, 4):
            n = 32*groups
            add(f'tgcheck-g{groups}', tgcheck_body(groups), n, {'AGX_LOCAL_X': str(n), 'AGX_SHARED_WORD': str(SHARED_4K)}, 1, 512)
    elif suite == 'tgwait':
        env = {'AGX_SHARED_WORD': str(SHARED_4K)}
        for mem in ('tgload', 'tgstore', 'atomic'):
            add(f'memwait-{mem}-ref', memwait_body(mem, b'', a.trips), 32, env, a.repeats, 512)
            for name, c in [('wait', WAIT), ('tg_barrier', REFS['tg_barrier']), ('mem_barrier', REFS['mem_barrier']),
                            ('18', CANDS['18']), ('78', CANDS['78'])]:
                add(f'memwait-{mem}-{name}', memwait_body(mem, c, a.trips), 32, env, a.repeats, 512)
    elif suite == 'endcost':
        add('endcost-suffix', b'', 65536, None, a.repeats)
        for name, c in ENDERS.items():
            add(f'endcost-{name}', c, 65536, None, a.repeats)
    elif suite == 'endpend':
        for name, c in {'wait': WAIT, **ENDERS}.items():
            add(f'endpend-{name}', endpend_body(c), 32, None, 1, 17*32)
    elif suite == 'endfollow':
        for name, c in {'wait': WAIT, **ENDERS}.items():
            add(f'endfollow-{name}', endfollow_body(c), 32, None, 1, 32)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['barrier', 'memwait', 'scope', 'tgcheck', 'tgwait', 'endcost', 'endpend', 'endfollow'], required=True)
    p.add_argument('--names', help='regex (fullmatch) to select jobs')
    p.add_argument('--k', type=int, default=20000, help='barrier: delay trips per SIMD-group index')
    p.add_argument('--trips', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--times', type=int, default=1, help='run each selected job this many times (separate processes)')
    p.add_argument('--device', default='/dev/dri/renderD128')
    p.add_argument('--timeout', type=int, default=30)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    a = p.parse_args()
    for b in list(CANDS.values())+list(ENDERS.values()):
        if b[0] & 0x7f in FORBIDDEN: sys.exit('refusing a faulting opcode')
    if not gates(): sys.exit('emulator gate failed')
    jobs = jobs_for(a.suite, a)
    if a.names: jobs = [j for j in jobs if re.fullmatch(a.names, j['name'])]
    jobs = [j for _ in range(a.times) for j in jobs]   # interleaved: all jobs, then all again
    print(json.dumps({'suite': a.suite, 'jobs': len(jobs), 'dry_run': a.dry_run}), flush=True)
    if a.dry_run: return 0
    if os.environ.get('CONFIRM_GPU_FAULTS') != '1':
        sys.exit('refusing to run without CONFIRM_GPU_FAULTS=1')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = a.output or Path('/scratch/apple-m2-playground/drm-compute/sync-probe')/f'{a.suite}-{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    exe = out/'drm_compute'
    r = subprocess.run(shlex.split(a.cc)+['-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', str(REPO/'tests/drm_compute.c'), '-o', str(exe)],
                       capture_output=True, timeout=60)
    if r.returncode: sys.exit('build failed: '+r.stderr.decode()[-400:])
    state = {'utc': stamp, 'suite': a.suite, 'k': a.k, 'trips': a.trips, 'gprs': GPRS, 'results': [], 'complete': False}
    save = lambda: (out/'record.json').write_text(json.dumps(state)+'\n')

    def run(jb):
        count = jb['count']; rounded = (count+31)//32*32
        (out/'code.bin').write_bytes(drm.build_main(jb['body'])+bytes(16))
        args = [str(exe), a.device, str(out/'code.bin'), str(count), str(out/'rb'), str(jb['repeats']), str(GPRS)]
        env = dict(os.environ, **jb['env'])
        if jb['table_words']:
            (out/'table.bin').write_bytes(struct.pack('<I', SENTINEL)*jb['table_words'])
            args.append(str(out/'table.bin')); env['AGX_TABLE_OUT'] = str(out/'table.out')
        rec = {'name': jb['name'], 'count': count, 'repeats': jb['repeats']}
        try:
            res = subprocess.run(args, capture_output=True, timeout=a.timeout, env=env)
            smp = [json.loads(l) for l in res.stdout.decode(errors='replace').splitlines() if l.startswith('{"rep"')]
            rec['fw_ns'] = [e['fw_ts'][1]-e['fw_ts'][0] for e in smp]
            rec['host_ns'] = [e['host_ns'] for e in smp]
            normal = res.returncode == 0 and len(smp) == jb['repeats'] and all(e['fw_ts'][0] and e['fw_ts'][1] and e['host_ns'] < 200_000_000 for e in smp)
            if res.returncode == 0:
                w = list(struct.unpack(f'<{rounded+32}I', (out/'rb').read_bytes()))
                want = [SENTINEL]*(rounded+16)+[0xa5a50000+j for j in range(16)]
                gi = list(range(8))+list(range(8+count, rounded+32))
                normal &= all(w[i] == want[i] for i in gi)
                pay = w[8:8+count]
                rec['suffix_lanes'] = sum(v == 0x12345678 for v in pay)
                if jb['table_words']:
                    rec['table'] = list(struct.unpack(f"<{jb['table_words']}I", (out/'table.out').read_bytes()))
            else:
                rec['stdout_tail'] = res.stdout.decode(errors='replace')[-300:]
            rec['fault'] = not normal
        except subprocess.TimeoutExpired:
            rec['fault'] = True; rec['timed_out'] = True
        return rec

    control = {'name': 'control', 'body': H('620978563412'), 'count': 32, 'env': {}, 'repeats': 1, 'table_words': 0}
    try:
        c = run(control)
        if c['fault'] or c.get('suffix_lanes') != 32: raise RuntimeError('control failed before the run')
        for jb in jobs:
            rec = run(jb); state['results'].append(rec); save()
            print(json.dumps({k: rec.get(k) for k in ('name', 'fault', 'suffix_lanes')} | {'fw_us': [round(x/1000, 1) for x in rec.get('fw_ns', [])][:4]}), flush=True)
            if rec['fault']:
                c = run(control); state['recovery_ok'] = not c['fault']
                raise RuntimeError(f"fault on {rec['name']} (control after: {'ok' if not c['fault'] else 'FAILED'})")
        state['complete'] = True
    except Exception as e:
        state['error'] = f'{type(e).__name__}: {e}'
    finally:
        save()
        print(json.dumps({'complete': state['complete'], 'n': len(state['results']), 'output': str(out), 'error': state.get('error')}), flush=True)
    return 0 if state['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
