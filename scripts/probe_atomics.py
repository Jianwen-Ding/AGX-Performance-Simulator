#!/usr/bin/env python3
"""Device atomic encodings outside Mesa's set, run through probe_variants (surroundings checked in agx_run).

  probe_atomics.py run   --family A|B [--only NAME] [--dry]    GPU (owner-approved unsafe probes)
  probe_atomics.py analyse DIR                                  offline: fit models to the recorded words

Family A, 64-bit min / max (applegpu: with op = cmpxchg, gap field u5 = bits 45-46 = 1 is umin64, 3 is umax64;
Mesa always emits u5 = 2): op 3, u5 in {1, 3, 0}, Rt in {0, 1}. Lane i: index r5 = 2i, source r6_r7 = S_i (64-bit).
A 4-byte index scale touches table pair i, an 8-byte scale pair 2i. Surrogate: Mesa's op 3 (32-bit cmpxchg, u5 = 2).

Family B, op values 11-15 (no name in Mesa or applegpu): 32-bit, Rt = 1, lane i on word i, r6 = value, r7 = second
operand (cmpxchg's compare slot). Operands mix floats, small integers and edge words. Surrogate: op 0 (add).
"""
import argparse, json, random, struct, sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import probe_variants as PV                               # noqa: E402

N, WG, REGS = 32, 32, 64
S_WORD = 2048                                             # source operands at table words S_WORD + 2i, + 2i + 1
TABLE_WORDS = 4096
OUT = ROOT / 'results/2026-10-17-atomic-variants'
M64 = (1 << 64) - 1


def f2u(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def u2f(x): return struct.unpack('<f', struct.pack('<I', x & 0xffffffff))[0]
def s64(x): return x - (1 << 64) if x >> 63 else x
def s32(x): return x - (1 << 32) if x >> 31 else x


def atomic_bytes(op, ret, u5=2):
    import run_drm_atomic as DA
    raw = DA.mesa_atomic(op, 2 * 14 if ret else 0, 16, 2 * 5, 2 * 6, ret)
    return (raw & ~(3 << 45) | u5 << 45).to_bytes(8, 'little')


def common(index_scale_words):
    """prefix: r6_r7 = source pair of lane i, r5 = index_scale_words * i, r14 = 0xdeadbeef. suffix: r2 = r14."""
    import record_suite as RS
    pre = RS.mov(27, 2) + RS.e('imadd r6, r1, r27, 0') + RS.mov(27, S_WORD) + RS.e('iadd r6, r6, r27')
    pre += RS.e('device_load 0, i32, xy, r6_r7, u8_u9, r6, unsigned') + RS.e('wait 0')
    pre += RS.mov(27, index_scale_words) + RS.e('imadd r5, r1, r27, 0') + RS.mov(14, 0xdeadbeef)
    return pre, RS.e('wait 0') + RS.e('iadd r2, r14, 0')


def out_ok(out, want_lanes):
    words = 8 + N + 8 + 16
    return (len(out) == words and out[:8] == [0xcdcdcdcd] * 8 and out[8:8 + N] == want_lanes
            and out[8 + N:8 + N + 8] == [0xcdcdcdcd] * 8 and out[-16:] == [0xa5a50000 + i for i in range(16)])


# ---------------------------------------------------------------------------------------------------------------
# Family A

def table_a(seed='atomic64'):
    rng = random.Random(seed)
    t = [0] * TABLE_WORDS
    for p in range(128):                       # memory pairs 0..127 (words 0..255)
        v = rng.getrandbits(64)
        t[2 * p], t[2 * p + 1] = v & 0xffffffff, v >> 32
    for i in range(N):                         # S_i relates to the pair lane i hits under each scale
        kind = i % 4
        v = (t[2 * i + 1] << 32) | t[2 * i]
        if kind == 0:   s = v ^ (1 << rng.randrange(32))                 # same high word: low word decides
        elif kind == 1: s = v ^ (1 << 63)                                 # sign flip: signed and unsigned disagree
        elif kind == 2: s = (v + rng.choice([1, -1])) & M64              # neighbour
        else:           s = rng.getrandbits(64)
        t[S_WORD + 2 * i], t[S_WORD + 2 * i + 1] = s & 0xffffffff, s >> 32
    return t


def surrogate_a_oracle(table, ret):
    """Mesa op 3 (32-bit cmpxchg, new = r6, compare = r7) at word 2i; returns (out lanes, table after)"""
    t = list(table); lanes = []
    for i in range(N):
        w = 2 * i; old = t[w]
        if old == table[S_WORD + 2 * i + 1]:
            t[w] = table[S_WORD + 2 * i]
        lanes.append(old if ret else 0xdeadbeef)
    return lanes, t


def probes_a():
    pre, suf = common(2)
    t = table_a()
    out = []
    for ret in (0, 1):
        sur = atomic_bytes(3, ret, 2)
        lanes, tab = surrogate_a_oracle(t, ret)
        for u5 in (1, 3, 0):
            out.append(PV.Probe(f'A_op3_u5{u5}_rt{ret}', pre, sur, atomic_bytes(3, ret, u5), suf, 3 << 45, N, WG, REGS, t,
                                lambda o, tb, lanes=lanes, tab=tab: out_ok(o, lanes) and tb == tab))
    return out


MODELS_64 = {
    'umin64': lambda v, s: min(v, s), 'umax64': lambda v, s: max(v, s),
    'smin64': lambda v, s: min(v, s, key=s64), 'smax64': lambda v, s: max(v, s, key=s64),
    'xchg64': lambda v, s: s, 'add64': lambda v, s: (v + s) & M64, 'none': lambda v, s: v,
}


def predict_a(table, model, scale):
    """64-bit op on the pair at byte 8*... : index 2i * scale bytes"""
    t = list(table); old = []
    for i in range(N):
        w = (2 * i * scale) // 4
        v = (t[w + 1] << 32) | t[w]; s = (table[S_WORD + 2 * i + 1] << 32) | table[S_WORD + 2 * i]
        old.append(v)
        r = MODELS_64[model](v, s)
        t[w], t[w + 1] = r & 0xffffffff, r >> 32
    return t, old


# ---------------------------------------------------------------------------------------------------------------
# Family B

def table_b(seed='atomic-op11-15'):
    rng = random.Random(seed)
    t = [0] * TABLE_WORDS
    floats = [0.0, -0.0, 1.0, -1.0, 0.5, 2.5, -3.75, 1e30, -1e30, 1e-40, float('inf'), float('-inf')]
    for i in range(N):
        k = i % 4
        if k == 0:   old, v, w = f2u(rng.uniform(-8, 8)), f2u(rng.uniform(-8, 8)), f2u(rng.uniform(-8, 8))
        elif k == 1: old, v, w = f2u(rng.choice(floats)), f2u(rng.choice(floats)), f2u(rng.choice(floats))
        elif k == 2: old, v, w = rng.randrange(16), rng.randrange(16), rng.randrange(16)   # inc / dec wrap bounds
        else:        old, v, w = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
        if i == 30: old = 0x7fc00000                               # NaN in memory
        if i == 31: v = 0x7fc00000                                 # NaN operand
        t[i], t[S_WORD + 2 * i], t[S_WORD + 2 * i + 1] = old, v, w
    return t


def probes_b():
    pre, suf = common(1)
    t = table_b()
    sur = atomic_bytes(0, 1)
    tab = list(t); lanes = []
    for i in range(N):
        lanes.append(t[i]); tab[i] = (t[i] + t[S_WORD + 2 * i]) & 0xffffffff
    return [PV.Probe(f'B_op{op}_rt1', pre, sur, atomic_bytes(op, 1), suf, 0xf << 6, N, WG, REGS, t,
                     lambda o, tb: out_ok(o, lanes) and tb == tab) for op in range(11, 16)]


def to_f32(x):
    """round a Python float to float32 bits (overflow to inf)"""
    try:
        return f2u(x)
    except OverflowError:
        return f2u(math.copysign(math.inf, x))


def fmodel(f):
    return lambda old, v, w: to_f32(f(u2f(old), u2f(v)))


def fmin(a, b, nan_wins=False):
    if math.isnan(a) or math.isnan(b):
        return math.nan if nan_wins or (math.isnan(a) and math.isnan(b)) else (b if math.isnan(a) else a)
    return a if (a < b or (a == b and math.copysign(1, a) < 0)) else b


def fmax(a, b, nan_wins=False):
    if math.isnan(a) or math.isnan(b):
        return math.nan if nan_wins or (math.isnan(a) and math.isnan(b)) else (b if math.isnan(a) else a)
    return a if (a > b or (a == b and math.copysign(1, a) > 0)) else b


MODELS_32 = {
    'fadd': fmodel(lambda a, b: a + b), 'fmin': fmodel(fmin), 'fmax': fmodel(fmax),
    'fmin_nan': fmodel(lambda a, b: fmin(a, b, True)), 'fmax_nan': fmodel(lambda a, b: fmax(a, b, True)),
    'inc_wrap': lambda o, v, w: 0 if o >= v else o + 1, 'dec_wrap': lambda o, v, w: v if o == 0 or o > v else o - 1,
    'add': lambda o, v, w: (o + v) & 0xffffffff, 'sub': lambda o, v, w: (o - v) & 0xffffffff, 'xchg': lambda o, v, w: v,
    'cmpxchg': lambda o, v, w: v if o == w else o, 'and': lambda o, v, w: o & v, 'or': lambda o, v, w: o | v,
    'xor': lambda o, v, w: o ^ v, 'nand': lambda o, v, w: ~(o & v) & 0xffffffff, 'none': lambda o, v, w: o,
    'umin': lambda o, v, w: min(o, v), 'umax': lambda o, v, w: max(o, v),
    'imin': lambda o, v, w: min(o, v, key=s32), 'imax': lambda o, v, w: max(o, v, key=s32),
}


# ---------------------------------------------------------------------------------------------------------------
# Family D: what happens after ops 11-15 (family B saw no memory change and no harness store)

D_BEFORE, D_RET, D_R1, D_REEN = 3000, 3100, 3200, 3300   # table word offsets (+ lane)


def probes_d():
    """before X: marker 0x11110000 + i -> D_BEFORE + i. After X: r14 -> D_RET + i, r1 -> D_R1 + i, then
    mov_imm r0l, 0 (the harness prologue's own instruction) and r14 -> D_REEN + i. Every index is computed before X."""
    import record_suite as RS
    st = RS.agx_loop.store
    pre, _ = common(1)
    for reg, off in ((20, D_BEFORE), (21, D_RET), (22, D_R1), (23, D_REEN)):
        pre += RS.mov(27, off) + RS.e(f'iadd r{reg}, r1, r27')
    pre += RS.mov(27, 0x11110000) + RS.e('iadd r15, r1, r27') + st(15, 8, 20) + RS.e('wait 0')
    suf = RS.e('wait 0') + st(14, 8, 21) + st(1, 8, 22) + RS.e('wait 0') + RS.e('mov_imm r0l, 0') + st(14, 8, 23)
    suf += RS.e('wait 0') + RS.e('iadd r2, r14, 0')
    t = table_b()
    tab = list(t); lanes = []
    for i in range(N):
        old = t[i]; lanes.append(old); tab[i] = (old + t[S_WORD + 2 * i]) & 0xffffffff
        tab[D_BEFORE + i], tab[D_RET + i], tab[D_R1 + i], tab[D_REEN + i] = 0x11110000 + i, old, i, old
    return [PV.Probe(f'D_op{op}_rt1', pre, atomic_bytes(0, 1), atomic_bytes(op, 1), suf, 0xf << 6, N, WG, REGS, t,
                     lambda o, tb: out_ok(o, lanes) and tb == tab) for op in range(11, 16)]


def analyse_d(r):
    t1, lanes = r['table_after'], r['out'][8:8 + N]
    seen = lambda off: sum(t1[off + i] != 0 for i in range(N))
    return {'before_marker': sum(t1[D_BEFORE + i] == 0x11110000 + i for i in range(N)), 'after_ret_stored': seen(D_RET),
            'after_r1_stored': seen(D_R1), 'r1_intact': sum(t1[D_R1 + i] == i for i in range(N)),
            'reenabled_stored': seen(D_REEN), 'harness_stored': sum(x != 0xcdcdcdcd for x in lanes),
            'ret_values': sorted({t1[D_RET + i] for i in range(N)} | {t1[D_REEN + i] for i in range(N)})[:8]}


E_N, E_WG, E_BEFORE, E_AFTER = 128, 64, 3300, 3100   # 2 workgroups x 2 SIMD groups; marker words + gid


def probes_e(ops=(11,), before=False, skip=False):
    """Scope of the op 11-15 stop: only lanes with (gid & 0x21) == 0x21 (odd lanes of the second SIMD group of each
    workgroup) execute X, inside if_icmp / pop_exec. Afterwards every lane stores 0x22220000 + gid -> E_AFTER + gid."""
    import record_suite as RS
    st = RS.agx_loop.store
    pre, _ = common(1)
    pre += RS.mov(27, E_AFTER) + RS.e('iadd r21, r1, r27') + RS.mov(27, 0x22220000) + RS.e('iadd r15, r1, r27')
    if before:   # every lane stores 0x11110000 + gid -> E_BEFORE + gid before the if
        pre += RS.mov(27, E_BEFORE) + RS.e('iadd r20, r1, r27') + RS.mov(27, 0x11110000) + RS.e('iadd r16, r1, r27')
        pre += st(16, 8, 20) + RS.e('wait 0')
    pre += RS.mov(27, 0x21) + RS.e('and r24, r1, r27') + RS.e('if_icmp r0l, ueq, r24, r27, 1')
    if skip:     # Mesa's if: SIMD groups with no active lane jump over X (8 bytes) and never issue it
        pre += RS.jnone(6 + 8)
    suf = RS.e('wait 0') + RS.e('pop_exec r0l, 1') + st(15, 8, 21) + RS.e('wait 0') + RS.e('iadd r2, r14, 0')
    t = table_b()
    for i in range(N, E_N):
        t[i], t[S_WORD + 2 * i], t[S_WORD + 2 * i + 1] = t[i % N], t[S_WORD + 2 * (i % N)], t[S_WORD + 2 * (i % N) + 1]
    tab = list(t); lanes = []
    for i in range(E_N):
        hit = i & 0x21 == 0x21
        lanes.append(t[i] if hit else 0xdeadbeef)
        if hit:
            tab[i] = (t[i] + t[S_WORD + 2 * i]) & 0xffffffff
        tab[E_AFTER + i] = 0x22220000 + i
        if before:
            tab[E_BEFORE + i] = 0x11110000 + i
    words = 8 + E_N + 8 + 16
    ok = lambda o, tb: (o[8:8 + E_N] == lanes and o[:8] == [0xcdcdcdcd] * 8 and o[-16:] == [0xa5a50000 + i for i in range(16)]
                        and len(o) == words and tb == tab)
    return [PV.Probe(f'E_op{op}_scope' + ('_before' if before else '') + ('_skip' if skip else ''), pre, atomic_bytes(0, 1), atomic_bytes(op, 1), suf, 0xf << 6, E_N, E_WG, REGS, t, ok)
            for op in ops]


def analyse_e(r):
    t1, lanes = r['table_after'], r['out'][8:8 + E_N]
    groups = {}
    for i in range(E_N):
        key = f'wg{i // E_WG}_simd{(i % E_WG) // 32}_{"exec" if i & 0x21 == 0x21 else ("odd" if i & 1 else "even")}'
        g = groups.setdefault(key, [0, 0, 0, 0])
        g[0] += 1; g[1] += t1[E_AFTER + i] == 0x22220000 + i; g[2] += lanes[i] != 0xcdcdcdcd; g[3] += t1[E_BEFORE + i] == 0x11110000 + i
    return {k: f'{v[3]}/{v[0]} before-marker, {v[1]}/{v[0]} after-marker, {v[2]}/{v[0]} harness store' for k, v in sorted(groups.items())}


# ---------------------------------------------------------------------------------------------------------------

def analyse(d):
    rows = []
    for p in sorted(Path(d).glob('*.json')):
        r = json.loads(p.read_text())
        if 'fault' in r:
            rows.append({'probe': r['name'], 'fault': r['fault'][:160]}); continue
        if r['name'].startswith('E_'):
            rows.append({'probe': r['name'], 'deterministic': r['deterministic'], **analyse_e(r)}); continue
        if r['name'].startswith('D_'):
            rows.append({'probe': r['name'], 'deterministic': r['deterministic'], **analyse_d(r)}); continue
        t0, t1, lanes = r['table_in'], r['table_after'], r['out'][8:8 + N]
        changed = [i for i in range(len(t0)) if t0[i] != t1[i]]
        row = {'probe': r['name'], 'deterministic': r['deterministic'], 'changed_words': changed[:80], 'n_changed': len(changed),
               'returned_untouched': sum(x == 0xdeadbeef for x in lanes)}
        if r['name'].startswith('A_'):
            fits = []
            for m in MODELS_64:
                for scale in (4, 8):
                    tab, old = predict_a(t0, m, scale)
                    if tab == t1:
                        ret_lo = [o & 0xffffffff for o in old]
                        fits.append({'model': m, 'scale': scale,
                                     'returns_old_low32': lanes == ret_lo, 'returns_nothing': lanes == [0xdeadbeef] * N})
            row['fits'] = fits
        else:
            per = {}
            for m, f in MODELS_32.items():
                ok_mem = [f(t0[i], t0[S_WORD + 2 * i], t0[S_WORD + 2 * i + 1]) == t1[i] for i in range(N)]
                per[m] = sum(ok_mem)
            row['memory_fit_lanes'] = dict(sorted(per.items(), key=lambda kv: -kv[1])[:5])
            row['returns_old'] = sum(lanes[i] == t0[i] for i in range(N))
            row['other_words_changed'] = [w for w in changed if w >= N][:20]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--family', choices=['A', 'B', 'D', 'E'], required=True)
    r.add_argument('--only'); r.add_argument('--dry', action='store_true'); r.add_argument('--out', type=Path, default=OUT / 'raw')
    s = sub.add_parser('analyse'); s.add_argument('dir', type=Path, nargs='?', default=OUT / 'raw')
    a = ap.parse_args()
    if a.cmd == 'run':
        PV.cov.install()
        ps = {'A': probes_a, 'B': probes_b, 'D': probes_d, 'E': lambda: probes_e() + probes_e(before=True) + probes_e(before=True, skip=True)}[a.family]()
        if a.only:
            ps = [p for p in ps if p.name == a.only]
        PV.run(ps, a.out, dry=a.dry)
    else:
        for row in analyse(a.dir):
            print(json.dumps(row))


if __name__ == '__main__':
    main()
