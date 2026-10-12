#!/usr/bin/env python3
"""Threadgroup (local) atomics on G14C through direct DRM: Mesa's 11 named ops only.

Same hang-avoidance rules as run_drm_atomic.py: ops 0..10 only, every encoding
built from Mesa's agx_pack.c AGX_OPCODE_LOCAL_ATOMIC formula and checked against
applegpu's `threadgroup_atomic` fields and 10-byte length, and the full body must
decode into exactly the intended instruction lengths. One workgroup of 32 lanes,
2 KiB declared threadgroup memory (USC Shared word from the addressing finding).

Mesa forces the index to immediate 0, so the 16-bit base register carries the
byte address. Modes:
  lane   : lane i stores T[i] at byte 4i, atomics it with X[i] (cmpxchg compare Y[i]),
           returns old in r2; threadgroup word read back to table[192+i].
  noret  : like lane, no destination.
  shared : all lanes store the same init at byte 1024, barrier, every lane atomics that word,
           barrier, every lane reads it back (final value).
  casrace: shared with compare = init for all lanes (one winner).
"""
import argparse, datetime, json, os, shlex, struct, subprocess, sys, time
from pathlib import Path

import agx_loop
import gpu_safety
import run_drm_compute as drm
import run_drm_atomic as DA
from run_drm_tg_layout import tg, shared_word

REPO = Path(__file__).resolve().parents[1]
OPS = DA.OPS
N = 32
BARRIER = bytes.fromhex('6800')


def mesa_local_atomic(op, R, A, S, ret):
    raw = 0x19 | (1 << 15) | (1 << 36) | (1 << 47)
    raw |= (1 << 8 if ret else 0) | (R & 63) << 9 | (1 << 15) | (A & 63) << 16 | 0 << 22
    raw |= op << 24 | 0 << 28 | (1 << 34) | (1 << 38 if ret else 0) | (A >> 6) << 58 | (R >> 6) << 60
    return raw | S << 64


def local_atomic(op, dst, base, src, ret):
    if op not in range(11):
        raise ValueError('only Mesa-named atomic ops 0..10 may run')
    _, g = agx_loop._tools()
    d = next(x for x in g.instruction_descriptors if x.name == 'threadgroup_atomic')
    x = mesa_local_atomic(op, 2*dst if ret else 0, 2*base, 2*src, ret)
    b = x.to_bytes(10, 'little')
    n = g.opcode_to_number(b)
    if g.get_instruction_descriptor(n) is not d or d.decode_size(n) != 10:
        raise ValueError(f'local atomic does not decode as 10-byte threadgroup_atomic: {b.hex()}')
    f = dict(d.decode_fields(n))
    want = {'op': op, 'Rt': int(ret), 'Ra': int(ret), 'R': 2*dst if ret else 0, 'A': 2*base, 'At': 0, 'O': 0, 'Ot': 1, 'S': 2*src}
    bad = {k: (f.get(k), v) for k, v in want.items() if f.get(k) != v}
    # applegpu re-encodes the short form: it omits the length bit 15 and the S word (bytes 8-9)
    if bad or d.encode_fields(f) | (1 << d.length_bit_pos) | (f['S'] << 64) != x:
        raise ValueError(f'applegpu/Mesa local atomic field disagreement {bad}')
    return b


def body(op, mode):
    e = agx_loop.encode; mov = agx_loop.chains.mov_imm
    ins = [mov(19, 4), mov(27, 0), e('imadd r22, r1, r19, r27'),                    # r22l = 4*lane
           e('iadd r3, r1, 0'), e('device_load 0, i32, x, r9, u8_u9, r3, unsigned'),
           e('iadd r3, r1, 64'), e('device_load 0, i32, x, r6, u8_u9, r3, unsigned'),
           e('iadd r3, r1, 128'), e('device_load 0, i32, x, r7, u8_u9, r3, unsigned'), e('wait 0'),
           mov(5, 0xdeadbeef)]
    shared = mode in ('shared', 'casrace')
    if shared:
        # every lane stores the same init value (identical writes, no branch), then barrier
        ins += [mov(23, 1024), mov(10, 0x40000000), tg(False, 'i32', 10, base=23), BARRIER]
        base = 23
    else:
        ins += [tg(False, 'i32', 9, base=22)]
        base = 22
    ret = mode != 'noret'
    ins += [local_atomic(op, 5, base, 6, ret), e('wait 0')]
    if shared: ins.append(BARRIER)
    ins += [mov(8, 0), tg(True, 'i32', 8, base=base), e('wait 0'),
            e('iadd r3, r1, 192'), agx_loop.store(8, 8, 3), e('wait 0'),
            e('iadd r2, r5, 0') if ret else mov(2, 0)]
    DA.decode_lengths(b''.join(ins), [len(i) for i in ins])
    return b''.join(ins)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ops', default=','.join(OPS))
    p.add_argument('--modes', default='noret,lane,shared,casrace')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    ops = [OPS.index(o) for o in a.ops.split(',')]
    T, X, Y = (v[:N] for v in DA.operands(a.seed))
    jobs = [{'name': f'{OPS[o]}-{m}', 'op': o, 'mode': m, 'body': body(o, m)}
            for m in a.modes.split(',') for o in ops if not (m == 'casrace' and OPS[o] != 'cmpxchg')]
    print(json.dumps({'jobs': len(jobs), 'example': jobs[0]['body'].hex()}), flush=True)
    if a.dry_run: return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = a.output or Path('/scratch/apple-m2-playground/drm-compute/tg-atomic')/stamp
    out.mkdir(parents=True, exist_ok=False)
    exe = out/'drm_compute'
    r = subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', str(REPO/'tests/drm_compute.c'), '-o', str(exe)], capture_output=True, timeout=60)
    if r.returncode: sys.exit('build failed')
    state = {'utc': stamp, 'T': T, 'X': X, 'Y': Y, 'results': [], 'complete': False}
    save = lambda: (out/'record.json').write_text(json.dumps(state)+'\n')
    init = 0x40000000

    def submit(code, words):
        (out/'c.bin').write_bytes(code+bytes(16)); (out/'t.bin').write_bytes(struct.pack(f'<{len(words)}I', *words))
        for f in ('rb', 't.out'):
            if (out/f).exists(): (out/f).unlink()
        env = dict(os.environ, AGX_LOCAL_X='32', AGX_SHARED_WORD=str(shared_word(2048)), AGX_TABLE_OUT=str(out/'t.out'))
        try:
            res = subprocess.run([str(exe), '/dev/dri/renderD128', str(out/'c.bin'), str(N), str(out/'rb'), '1', '64', str(out/'t.bin')],
                                 capture_output=True, env=env, timeout=20)
        except subprocess.TimeoutExpired:
            return None
        smp = [json.loads(l) for l in res.stdout.decode(errors='replace').splitlines() if l.startswith('{') and '"rep"' in l]
        if res.returncode or not smp or not all(e['fw_ts'][0] and e['fw_ts'][1] and e['host_ns'] < 50_000_000 for e in smp):
            return None
        w = list(struct.unpack(f'<{N+32}I', (out/'rb').read_bytes()))
        return w[8:8+N], list(struct.unpack(f'<{len(words)}I', (out/'t.out').read_bytes()))

    try:
        for jb in jobs:
            Yj = [init]*N if jb['mode'] == 'casrace' else Y
            words = T+[0]*32+X+[0]*32+Yj+[0]*32+[0xcdcdcdcd]*64
            got = submit(drm.build_main(jb['body']), words)
            rec = {'name': jb['name'], 'body': jb['body'].hex()}
            if got is None:
                rec['fault'] = True; state['results'].append(rec); time.sleep(2)
                c = submit(drm.build_main(bytes.fromhex('620978563412')), words)
                state['fault'] = {'name': jb['name'], 'control_after_ok': c is not None}
                gpu_safety.record(f"{Path(__file__).name} {jb['name']}")
                raise RuntimeError(f"fault on {jb['name']}; stopping")
            o, t = got; tgw = t[192:192+N]; notes = []
            opn = OPS[jb['op']]
            if jb['mode'] in ('lane', 'noret'):
                want = [DA.apply(opn, T[i], X[i], Y[i]) for i in range(N)]
                if tgw != want: notes.append({'tg_mismatch': [[i, T[i], X[i], tgw[i], want[i]] for i in range(N) if tgw[i] != want[i]][:6]})
                if jb['mode'] == 'lane' and o != T: notes.append({'return_mismatch': [[i, T[i], o[i]] for i in range(N) if o[i] != T[i]][:6]})
            else:
                if len(set(tgw)) != 1: notes.append('lanes read different final values')
                final = tgw[0]
                DA.N = N
                n2, _ = DA.check_shared(jb['op'], o, final, X, Yj, init); notes += n2
                DA.N = 64
                if jb['mode'] == 'casrace':
                    winners = [i for i in range(N) if o[i] == init]
                    rec['winners'] = winners
                    if len(winners) != 1 or final != X[winners[0]]: notes.append(f'winners {winners}')
            if t[:192] != words[:192]: notes.append('input table changed')
            rec.update(ok=not notes, notes=notes, out=o, tg=tgw)
            state['results'].append(rec); save()
            print(json.dumps({'name': jb['name'], 'ok': rec['ok'], 'notes': notes or None})[:500], flush=True)
        state['complete'] = True
    except Exception as e:
        state['error'] = f'{type(e).__name__}: {e}'
    finally:
        save()
        print(json.dumps({'complete': state['complete'], 'output': str(out), 'error': state.get('error'),
                          'passed': sum(1 for r in state['results'] if r.get('ok')), 'n': len(state['results'])}), flush=True)
    return 0 if state['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
