#!/usr/bin/env python3
"""Device atomics (opcode 0x15) on G14C through direct DRM: Mesa's 11 named ops only.

Hang avoidance (no unknown encodings reach the GPU):
  * op values 0..10 only (Mesa ATOMIC_OPC); 11..15 are unnamed and never run.
  * every atomic is built twice, from applegpu's field list and from Mesa's
    agx_pack.c formula (needs_g13x_coherency = NO on single-die G14), and the
    two must be bit-identical.
  * the whole body must decode with applegpu into exactly the intended
    instruction lengths (no length desync).
  * one repeat per dispatch; the first timeout or fault stops the run and an
    unchanged control is submitted to confirm recovery.

Table (u8_u9) words: [0,64) per-lane targets T, [64,128) operand X, [128,192)
compare Y, [256] contended word. Output: r2 per lane (the returned old value).

Experiments per op:
  lane   : lane i does op on T[i] with X[i] (cmpxchg: compare Y[i]), returning old.
  shared : every lane does op on word 256 with X[i], returning old.
  noret  : like lane but Rt = 0 (no destination), table only.
  casrace: shared, but every lane's compare is the initial value (cmpxchg: exactly one winner).
  count  : --count-lanes lanes add 1 to word 256; returns must be a permutation of init..init+n-1.
"""
import argparse, datetime, json, os, random, shlex, struct, subprocess, sys, time
from pathlib import Path

import agx_loop
import gpu_safety
import run_drm_compute as drm

REPO = Path(__file__).resolve().parents[1]
OPS = ['add', 'sub', 'xchg', 'cmpxchg', 'umin', 'imin', 'umax', 'imax', 'and', 'or', 'xor']
M32 = 0xffffffff
N = 64


def s32(v): return v - (1 << 32) if v & 0x80000000 else v


def apply(op, old, x, y=None):
    if op == 'add': return (old + x) & M32
    if op == 'sub': return (old - x) & M32
    if op == 'xchg': return x
    if op == 'cmpxchg': return x if old == y else old
    if op == 'umin': return min(old, x)
    if op == 'umax': return max(old, x)
    if op == 'imin': return min(old, x, key=s32)
    if op == 'imax': return max(old, x, key=s32)
    if op == 'and': return old & x
    if op == 'or': return old | x
    if op == 'xor': return old ^ x
    raise ValueError(op)


def mesa_atomic(op, R, A, O, S, Rt, uniform_base=True, scoreboard=0, unsigned_index=True):
    """Mesa agx_pack.c AGX_OPCODE_ATOMIC formula (register/uniform values in 16-bit halves)."""
    raw = 0x15 | (1 << 26) | (1 << 31) | (5 << 44)
    raw |= op << 6 | (R & 63) << 10 | (A & 15) << 16 | (O & 15) << 20
    raw |= (1 << 25 if unsigned_index else 0) | (1 << 27 if uniform_base else 0) | scoreboard << 30
    raw |= ((O >> 4) & 15) << 32 | ((A >> 4) & 15) << 36 | (R >> 6) << 40
    raw |= (1 << 47 if Rt else 0) | S << 48 | (O >> 8) << 56
    return raw


def atomic(op, dst, idx, src, ret):
    if op not in range(11):
        raise ValueError('only Mesa-named atomic ops 0..10 may run')
    _, g = agx_loop._tools()
    d = next(x for x in g.instruction_descriptors if x.name == 'atomic')
    f = {'op': op, 'g': 0, 'Rt': int(ret), 'R': 2*dst if ret else 0, 'At': 1, 'A': 16, 'Ot': 0, 'O': 2*idx, 'Ou': 1,
         'S': 2*src, 'u1': 1, 'u2': 0, 'u3': 1, 'u4': 4, 'u5': 2}
    x = d.encode_fields(f)
    m = mesa_atomic(op, 2*dst if ret else 0, 16, 2*idx, 2*src, ret)
    b = x.to_bytes(8, 'little')
    n = g.opcode_to_number(b)
    if x != m or g.get_instruction_descriptor(n) is not d or dict(d.decode_fields(n)) != f or d.decode_size(n) != 8:
        raise ValueError(f'atomic encoding disagreement: applegpu {x:016x} mesa {m:016x}')
    return b


def decode_lengths(code, expect):
    """Walk code with applegpu; lengths must equal the intended instruction list."""
    _, g = agx_loop._tools()
    pos, got = 0, []
    while pos < len(code):
        n = g.opcode_to_number(code[pos:pos+16])
        d = g.get_instruction_descriptor(n)
        if d is None: raise ValueError(f'undecodable at {pos}')
        sz = d.decode_size(n); got.append(sz); pos += sz
    if got != expect or pos != len(code):
        raise ValueError(f'length desync: {got} vs {expect}')


def body(op, mode):
    e = agx_loop.encode
    if mode == 'count':
        ins = [e('mov_imm r5, 3735928559, 0b0'), e('mov_imm r6, 1, 0b0'), e('mov_imm r3, 256, 0b0'),
               atomic(op, 5, 3, 6, True), e('wait 0'), e('iadd r2, r5, 0')]
        decode_lengths(b''.join(ins), [len(i) for i in ins])
        return b''.join(ins)
    ins = [e('iadd r3, r1, 64'), e('device_load 0, i32, x, r6, u8_u9, r3, unsigned'),
           e('iadd r3, r1, 128'), e('device_load 0, i32, x, r7, u8_u9, r3, unsigned'), e('wait 0')]
    if mode in ('shared', 'casrace'):
        ins.append(e('mov_imm r3, 256, 0b0'))
        idx = 3
    else:
        idx = 1
    ret = mode != 'noret'
    ins.append(atomic(op, 5, idx, 6, ret))
    ins.append(e('wait 0'))
    ins.append(e('iadd r2, r5, 0') if ret else e('mov_imm r2, 0, 0b0'))
    if ret:
        ins.insert(0, e('mov_imm r5, 3735928559, 0b0'))   # 0xdeadbeef: shows a missing return write
    decode_lengths(b''.join(ins), [len(i) for i in ins])
    return b''.join(ins)


def operands(seed):
    r = random.Random(seed)
    edge = [0, 1, 2, 0x7fffffff, 0x80000000, 0x80000001, M32, 0xfffffffe, 0x55555555, 0xaaaaaaaa, 0x0000ffff, 0xffff0000]
    T = [edge[i] if i < len(edge) else r.getrandbits(32) for i in range(N)]
    X = [edge[(i*5+3) % len(edge)] if i < 2*len(edge) else r.getrandbits(32) for i in range(N)]
    Y = [T[i] if i % 2 == 0 else T[i] ^ (1 << (i % 32)) for i in range(N)]
    return T, X, Y


def expect(op, mode, T, X, Y, init):
    """Return (per-lane returns or None, table updates, check function for shared)."""
    opn = OPS[op]
    if mode in ('lane', 'noret'):
        new = [apply(opn, T[i], X[i], Y[i]) for i in range(N)]
        return (T[:] if mode == 'lane' else [0]*N), {i: new[i] for i in range(N)}
    return None, None


def check_shared(op, rets, final, X, Y, init):
    """Shared word: verify the returns form a serialization consistent with the op."""
    opn = OPS[op]
    notes = []
    # Reconstruct: each returned value is the state before that lane's op. Find an order.
    # For commutative/associative ops the final value is order-independent.
    if opn in ('add', 'sub', 'and', 'or', 'xor', 'umin', 'umax', 'imin', 'imax'):
        v = init
        for x in X: v = apply(opn, v, x)
        if final != v: notes.append(f'final {final:#x} != fold {v:#x}')
    # Serialization search: an order of all lanes in which each lane's return equals the
    # state before its op, ending at the stored final value. Lanes that leave the state
    # unchanged are applied as soon as they can be (never harmful); identical lanes are
    # interchangeable, so only one per (ret, X, Y) class is tried. Backtracking, memoized.
    from collections import Counter
    order = []
    seen = set()
    def dfs(state, left):
        while True:
            idle = [k for k in left if left[k] and k[0] == state and apply(opn, state, k[1], k[2]) == state]
            if not idle: break
            for k in idle: order.extend([k]*left[k]); left[k] = 0
        if not +left: return state == final
        key = (state, frozenset((+left).items()))
        if key in seen: return False
        seen.add(key)
        for k in [k for k in left if left[k] and k[0] == state]:
            l2 = Counter(left); l2[k] -= 1; mark = len(order); order.append(k)
            if dfs(apply(opn, state, k[1], k[2]), l2): return True
            del order[mark:]
        return False
    if not dfs(init, Counter((rets[i], X[i], Y[i]) for i in range(N))):
        notes.append('no serialization of the returned values reaches the final value')
    order = [list(k) for k in order]
    state = final
    if not notes and state != final: notes.append(f'serialized end {state:#x} != final {final:#x}')
    return notes, order


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ops', default=','.join(OPS))
    p.add_argument('--modes', default='noret,lane,shared')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--count-lanes', type=int, default=4096)
    p.add_argument('--device', default='/dev/dri/renderD128')
    p.add_argument('--gprs', type=int, default=32)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    a = p.parse_args()
    ops = [OPS.index(o) for o in a.ops.split(',')]
    modes = a.modes.split(',')
    T, X, Y = operands(a.seed)
    jobs = []
    for mode in modes:
        for op in ops:
            jobs.append({'name': f'{OPS[op]}-{mode}', 'op': op, 'mode': mode, 'body': body(op, mode)})
    print(json.dumps({'jobs': len(jobs), 'atomics_hex': {j['name']: j['body'].hex() for j in jobs[:3]}}), flush=True)
    if a.dry_run: return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = a.output or Path('/scratch/apple-m2-playground/drm-compute/atomic')/stamp
    out.mkdir(parents=True, exist_ok=False)
    exe = out/'drm_compute'
    r = subprocess.run(shlex.split(a.cc)+['-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', str(REPO/'tests/drm_compute.c'), '-o', str(exe)],
                       capture_output=True, timeout=60)
    if r.returncode: sys.exit('build failed: '+r.stderr.decode()[-400:])
    state = {'utc': stamp, 'seed': a.seed, 'T': T, 'X': X, 'Y': Y, 'results': [], 'complete': False}
    save = lambda: (out/'record.json').write_text(json.dumps(state)+'\n')

    def submit(name, code, table_words, count):
        cpath = out/'code.bin'; cpath.write_bytes(code+bytes(16))
        tpath = out/'table.bin'; tpath.write_bytes(struct.pack(f'<{len(table_words)}I', *table_words))
        rb, dump = out/'readback', out/'table.out'
        for f in (rb, dump):
            if f.exists(): f.unlink()
        try:
            res = subprocess.run([str(exe), a.device, str(cpath), str(count), str(rb), '1', str(a.gprs), str(tpath)],
                                 capture_output=True, timeout=20, env=dict(os.environ, AGX_TABLE_OUT=str(dump)))
        except subprocess.TimeoutExpired:
            return {'fault': True, 'timed_out': True}
        ev = [json.loads(l) for l in res.stdout.decode(errors='replace').splitlines() if l.startswith('{')]
        smp = [e for e in ev if 'rep' in e]
        rec = {'returncode': res.returncode, 'fw_ns': [e['fw_ts'][1]-e['fw_ts'][0] for e in smp], 'host_ns': [e['host_ns'] for e in smp]}
        normal = res.returncode == 0 and smp and all(e['fw_ts'][0] and e['fw_ts'][1] and e['host_ns'] < 50_000_000 for e in smp)
        if not normal:
            rec['fault'] = True; rec['stdout_tail'] = res.stdout.decode(errors='replace')[-400:]; return rec
        rounded = (count+31)//32*32
        w = list(struct.unpack(f'<{rounded+32}I', rb.read_bytes()))
        guards = w[:8] == [0xcdcdcdcd]*8 and w[8+count:rounded+16] == [0xcdcdcdcd]*(rounded+8-count) and w[rounded+16:] == [0xa5a50000+j for j in range(16)]
        rec.update(fault=False, guards=guards, out=w[8:8+count], table=list(struct.unpack(f'<{len(table_words)}I', dump.read_bytes())))
        return rec

    ctrl = drm.build_main(bytes.fromhex('620978563412'))
    init = 0x40000000
    try:
        for jb in jobs:
            Yj = [init]*N if jb['mode'] == 'casrace' else Y
            words = T+X+Yj+[0]*64+[init]+[0]*63
            lanes = a.count_lanes if jb['mode'] == 'count' else N
            rec = submit(jb['name'], drm.build_main(jb['body']), words, lanes)
            rec.update(name=jb['name'], op=OPS[jb['op']], mode=jb['mode'], body=jb['body'].hex())
            if rec['fault']:
                state['results'].append(rec); time.sleep(2)
                c = submit('control', ctrl, words, N)
                state['fault'] = {'name': jb['name'], 'control_after_ok': not c['fault']}
                gpu_safety.record(f"{Path(__file__).name} {jb['name']}")
                raise RuntimeError(f"fault on {jb['name']}; stopping (control after: {'ok' if not c['fault'] else 'FAILED'})")
            tab = rec['table']; notes = []
            if not rec['guards']: notes.append('guard words changed')
            untouched = list(range(64, 256)) + list(range(257, 320)) + ([] if jb['mode'] not in ('shared', 'casrace', 'count') else list(range(64)))
            if jb['mode'] in ('lane', 'noret'):
                want_out, want_tab = expect(jb['op'], jb['mode'], T, X, Y, init)
                bad_t = [i for i in range(N) if tab[i] != want_tab[i]]
                if bad_t: notes.append({'table_mismatch': [[i, T[i], X[i], Y[i], tab[i], want_tab[i]] for i in bad_t[:8]], 'n': len(bad_t)})
                if jb['mode'] == 'lane':
                    bad_o = [i for i in range(N) if rec['out'][i] != T[i]]
                    if bad_o: notes.append({'return_mismatch': [[i, T[i], rec['out'][i]] for i in bad_o[:8]], 'n': len(bad_o)})
                untouched.append(256)
            elif jb['mode'] == 'count':
                if sorted(rec['out']) != list(range(init, init+lanes)): notes.append('returns are not a permutation of init..init+n-1')
                if tab[256] != init+lanes: notes.append(f'final {tab[256]:#x} != {init+lanes:#x}')
                rec['out'] = rec['out'][:64]
            else:
                n2, order = check_shared(jb['op'], rec['out'], tab[256], X, Yj, init)
                if jb['mode'] == 'casrace':
                    winners = [i for i in range(N) if rec['out'][i] == init]
                    rec['winners'] = winners
                    if len(winners) != 1 or tab[256] != X[winners[0]]: notes.append(f'cmpxchg race winners {winners}')
                notes += n2; rec['order'] = order
            bad_u = [i for i in untouched if tab[i] != words[i]]
            if bad_u: notes.append({'untouched_changed': bad_u[:8]})
            rec['ok'] = not notes; rec['notes'] = notes
            for k in ('table',): rec[k] = rec[k][:64] + [rec[k][256]]
            state['results'].append(rec); save()
            print(json.dumps({'name': jb['name'], 'ok': rec['ok'], 'notes': notes if notes else None})[:600], flush=True)
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
