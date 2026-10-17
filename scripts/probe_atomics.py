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


# Family F: the other Mesa ops with u5 = 0 / 1 / 3 (is there a 64-bit add / xchg / and ...?)

F_R14, F_R15 = 3100, 3200
MESA_OPS = ['add', 'sub', 'xchg', 'cmpxchg', 'umin', 'imin', 'umax', 'imax', 'and', 'or', 'xor']
OPS64 = {'add': lambda v, s: (v + s) & M64, 'sub': lambda v, s: (v - s) & M64, 'xchg': lambda v, s: s,
         'umin': MODELS_64['umin64'], 'imin': MODELS_64['smin64'], 'umax': MODELS_64['umax64'], 'imax': MODELS_64['smax64'],
         'and': lambda v, s: v & s, 'or': lambda v, s: v | s, 'xor': lambda v, s: v ^ s}


def op32(name, old, v):
    return {'add': (old + v) & 0xffffffff, 'sub': (old - v) & 0xffffffff, 'xchg': v, 'umin': min(old, v),
            'imin': min(old, v, key=s32), 'umax': max(old, v), 'imax': max(old, v, key=s32), 'and': old & v,
            'or': old | v, 'xor': old ^ v}[name]


def probes_f(ops=None):
    """lane i: index 2i, source r6_r7 = S_i, Rt = 1 into r14 (r15 = 0xdeadbeef too); after: r14 -> F_R14 + i,
    r15 -> F_R15 + i. Surrogate: the same op with Mesa's u5 = 2 (32-bit, on word 2i)."""
    import record_suite as RS
    st = RS.agx_loop.store
    pre, _ = common(2)
    pre += RS.mov(15, 0xdeadbeef)
    for reg, off in ((21, F_R14), (22, F_R15)):
        pre += RS.mov(27, off) + RS.e(f'iadd r{reg}, r1, r27')
    suf = RS.e('wait 0') + st(14, 8, 21) + st(15, 8, 22) + RS.e('wait 0') + RS.e('iadd r2, r14, 0')
    t = table_a()
    out = []
    for op in ops or [o for o in range(11) if o != 3]:
        name = MESA_OPS[op]
        tab = list(t); lanes = []
        for i in range(N):
            old = t[2 * i]; lanes.append(old)
            tab[2 * i] = op32(name, old, t[S_WORD + 2 * i])
            tab[F_R14 + i], tab[F_R15 + i] = old, 0xdeadbeef
        for u5 in (1, 3, 0):
            out.append(PV.Probe(f'F_{name}_u5{u5}', pre, atomic_bytes(op, 1, 2), atomic_bytes(op, 1, u5), suf, 3 << 45, N, WG,
                                REGS, t, lambda o, tb, lanes=lanes, tab=tab: out_ok(o, lanes) and tb == tab))
    return out


def analyse_f(r):
    name = r['name'].split('_')[1]
    t0, t1, lanes = r['table_in'], r['table_after'], r['out'][8:8 + N]
    mem = lambda tb: [x for k, x in enumerate(tb) if k < 256]
    fits = []
    t32 = list(t0)
    for i in range(N):
        t32[2 * i] = op32(name, t0[2 * i], t0[S_WORD + 2 * i])
    if mem(t32) == mem(t1): fits.append('32-bit op on word 2i')
    tf = list(t0)
    for i in range(N):
        tf[2 * i] = fadd_ref(t0[2 * i], t0[S_WORD + 2 * i], ftz=True)
    if None not in tf and mem(tf) == mem(t1): fits.append('float32 add (round to nearest even, denormals flushed) on word 2i')
    for m, f in list(OPS64.items()) + [('none', lambda v, s: v)]:
        for scale in (4, 8):
            tb = list(t0)
            for i in range(N):
                w = (2 * i * scale) // 4
                v = (tb[w + 1] << 32) | tb[w]; s = (t0[S_WORD + 2 * i + 1] << 32) | t0[S_WORD + 2 * i]
                x = f(v, s); tb[w], tb[w + 1] = x & 0xffffffff, x >> 32
            if mem(tb) == mem(t1): fits.append(f'64-bit {m}, index x{scale} bytes')
    r14 = [t1[F_R14 + i] for i in range(N)]; r15 = [t1[F_R15 + i] for i in range(N)]
    ended = all(x == 0xcdcdcdcd for x in lanes) and all(x == 0 for x in r14)
    ret = ('SIMD group ended' if ended else 'old low word' if r14 == [t0[2 * i] for i in range(N)]
           else 'changed flag' if all(r14[i] == int(t0[2 * i:2 * i + 2] != t1[2 * i:2 * i + 2]) for i in range(N)) else 'other')
    ret15 = 'untouched' if r15 == [0xdeadbeef] * N else 'old high word' if r15 == [t0[2 * i + 1] for i in range(N)] else 'other'
    return {'memory_fits': fits, 'r14': ret, 'r15': ret15, 'n_changed_mem': sum(a != b for a, b in zip(mem(t0), mem(t1)))}


# Family G: the float add found in family F (op 1 = sub with u5 bit 45): edges, rounding, contention

G_N = 1024
F32_EDGE = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x00000002, 0x007fffff, 0x807fffff, 0x00400000, 0x00800000,
            0x80800000, 0x00800001, 0x01000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x33800000, 0x33800001, 0xb3800000,
            0x34000000, 0x3f000000, 0x4b000000, 0x4b800000, 0x7f7fffff, 0xff7fffff, 0x7f000000, 0x7f800000, 0xff800000,
            0x7fc00000, 0xffc00000, 0x7fc12345, 0x7f800001, 0x7fa00000, 0x40490fdb, 0xc0490fdb, 0x5f000000, 0x1f000000,
            0x0d000000, 0x8d000000, 0x3effffff, 0x3e800001]


def round_f32(q, sign_if_zero=0):
    """exact rational -> float32 bits, round to nearest even, overflow to inf, gradual underflow"""
    from fractions import Fraction
    if q == 0:
        return sign_if_zero << 31
    s = 1 if q < 0 else 0
    q = abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    while Fraction(2) ** e > q: e -= 1
    while Fraction(2) ** (e + 1) <= q: e += 1
    e = max(e, -126)
    quantum = Fraction(2) ** (e - 23)
    m = q / quantum
    fl = m.numerator // m.denominator; rem = m - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1): fl += 1
    if fl == 1 << 24: fl >>= 1; e += 1
    if e > 127: return s << 31 | 0x7f800000
    if fl < 1 << 23: return s << 31 | fl                       # denormal (e == -126)
    return s << 31 | (e + 127) << 23 | (fl - (1 << 23))


def fadd_ref(a, b, ftz=False):
    """IEEE float32 a + b on bits; NaN results reported as None (payload checked separately)"""
    from fractions import Fraction
    ea, eb = a >> 23 & 0xff, b >> 23 & 0xff
    if (ea == 0xff and a & 0x7fffff) or (eb == 0xff and b & 0x7fffff): return None
    if ftz:
        a = a & 0x80000000 if ea == 0 else a
        b = b & 0x80000000 if eb == 0 else b
    if ea == 0xff or eb == 0xff:
        if ea == 0xff and eb == 0xff and (a ^ b) >> 31: return None
        return a if ea == 0xff else b
    q = Fraction(u2f(a)) + Fraction(u2f(b))
    r = round_f32(q, 1 if (a >> 31 and b >> 31) else 0)
    return r & 0x80000000 if ftz and r >> 23 & 0xff == 0 else r


def table_g(part):
    rng = random.Random(f'fadd-atomic-{part}')
    pairs = [(x, y) for x in F32_EDGE for y in F32_EDGE]
    pairs = pairs[part * G_N:(part + 1) * G_N]
    while len(pairs) < G_N:    # random pairs with close exponents: rounding and ties
        x = rng.getrandbits(32) & 0x7fffffff
        if x >> 23 in (0, 0xff): continue
        y = (x & 0xff800000) ^ rng.getrandbits(23) ^ (rng.getrandbits(1) << 31)
        if rng.random() < 0.3: y = (y & 0xff800000) - (rng.randrange(1, 25) << 23) & 0xff800000 | 1 << 22 if y >> 23 & 0xff > 30 else y
        pairs.append((x, y & 0xffffffff))
    t = [0] * TABLE_WORDS
    for i, (x, y) in enumerate(pairs):
        t[i], t[S_WORD + 2 * i] = x, y
    return t[:S_WORD + 2 * G_N]


def probes_g():
    """1024 lanes, lane i: word i (index scale 1) += S_i, Rt = 1. Surrogate: Mesa's integer sub (u5 = 2)."""
    pre, suf = common(1)
    out = []
    for part in (0, 1):
        t = table_g(part)
        tab = list(t); lanes = []
        for i in range(G_N):
            lanes.append(t[i]); tab[i] = (t[i] - t[S_WORD + 2 * i]) & 0xffffffff
        words = 8 + G_N + 8 + 16
        ok = lambda o, tb, lanes=lanes, tab=tab: (len(o) == words and o[8:8 + G_N] == lanes and o[:8] == [0xcdcdcdcd] * 8
                                                 and o[-16:] == [0xa5a50000 + i for i in range(16)] and tb == tab)
        out.append(PV.Probe(f'G_fadd_edges{part}', pre, atomic_bytes(1, 1, 2), atomic_bytes(1, 1, 1), suf, 3 << 45, G_N, 64,
                            REGS, t, ok))
    return out


def analyse_g(r):
    t0, t1, lanes = r['table_in'], r['table_after'], r['out'][8:8 + G_N]
    res = {'returns_old': sum(lanes[i] == t0[i] for i in range(G_N))}
    for ftz in (False, True):
        bad = []
        for i in range(G_N):
            a, b, got = t0[i], t0[S_WORD + 2 * i], t1[i]
            want = fadd_ref(a, b, ftz)
            if want is None:
                if not (got >> 23 & 0xff == 0xff and got & 0x7fffff): bad.append(i)
            elif want != got:
                bad.append(i)
        res['ftz' if ftz else 'ieee'] = {'match': G_N - len(bad), 'mismatch': [(f'{t0[i]:08x}', f'{t0[S_WORD + 2 * i]:08x}', f'{t1[i]:08x}') for i in bad[:6]]}
    nan_out = {}
    for i in range(G_N):
        a, b, got = t0[i], t0[S_WORD + 2 * i], t1[i]
        if fadd_ref(a, b) is None:
            nan_out[f'{a:08x}+{b:08x}'] = f'{got:08x}'
    res['nan_results'] = sorted(set(nan_out.values()))
    res['nan_examples'] = dict(list(nan_out.items())[:8])
    return res


def probes_h():
    """contention: all G_N lanes on word 0 (index 0), Rt = 1. H_ones: += 1.0 from 0.0; H_mixed: += values whose sums
    round, from 1.0. Surrogate: Mesa's integer sub on the same word."""
    import record_suite as RS
    pre, suf = common(0)
    out = []
    for name, init, vals in (('H_fadd_contend_ones', 0, [f2u(1.0)] * G_N),
                             ('H_fadd_contend_mixed', f2u(1.0), [f2u(random.Random(k).uniform(0.001, 3.0)) for k in range(G_N)])):
        t = [0] * (S_WORD + 2 * G_N)
        t[0] = init
        for i, v in enumerate(vals):
            t[S_WORD + 2 * i] = v
        words = 8 + G_N + 8 + 16

        def ok(o, tb, t=t):   # surrogate: integer sub serialised in some order; check the final word and permutation
            rets = o[8:8 + G_N]
            fin = (t[0] - sum(t[S_WORD + 2 * i] for i in range(G_N))) & 0xffffffff
            return (len(o) == words and tb[0] == fin and tb[1:] == t[1:] and o[:8] == [0xcdcdcdcd] * 8
                    and o[-16:] == [0xa5a50000 + i for i in range(16)] and t[0] in rets)
        out.append(PV.Probe(name, pre, atomic_bytes(1, 1, 2), atomic_bytes(1, 1, 1), suf, 3 << 45, G_N, 64, REGS, t, ok))
    return out


def analyse_h(r):
    t0, t1, rets = r['table_in'], r['table_after'], r['out'][8:8 + G_N]
    vals = [t0[S_WORD + 2 * i] for i in range(G_N)]
    final = t1[0]
    # a valid serialisation: one chain init -> ... -> final where each lane's (ret + v) is the next lane's ret
    chain_ok, cur, used = True, t0[0], 0
    pool = {}
    for i in range(G_N):
        pool.setdefault(rets[i], []).append(i)
    while used < G_N:
        if cur not in pool or not pool[cur]:
            chain_ok = False; break
        i = pool[cur].pop(); used += 1
        cur = fadd_ref(rets[i], vals[i], ftz=True)
    return {'final': f'{final:08x} ({u2f(final)})', 'distinct_returns': len(set(rets)), 'serialisation_chain_ok': chain_ok and cur == final,
            'lanes_chained': used}


# Family I: is op 3 with u5 = 0 a 64-bit compare-and-swap? Lane kinds make the 32-bit compare (M.lo == r7) and the
# 64-bit compare (M == r8_r9, or M == r6_r7 for the swapped layout) succeed on different lanes.

I_R14, I_R15 = 3100, 3200
I_SRC = 2048                                   # r6..r9 of lane i at words I_SRC + 4i


def table_i():
    rng = random.Random('cas64')
    t = [0] * (I_R15 + N)
    for i in range(N):
        lo, hi = rng.getrandbits(32), rng.getrandbits(32)
        r6, r7, r8, r9 = (rng.getrandbits(32) for _ in range(4))
        k = i % 5
        if k == 0: r7 = lo                              # 32-bit compare true only
        elif k == 1: r8, r9 = lo, hi                    # 64-bit compare (r8_r9) true only
        elif k == 2: r7 = r8 = lo; r9 = hi              # both
        elif k == 3: r6, r7 = lo, hi                    # swapped 64-bit layout (compare in r6_r7) true
        t[2 * i], t[2 * i + 1] = lo, hi
        t[I_SRC + 4 * i:I_SRC + 4 * i + 4] = [r6, r7, r8, r9]
    return t


def probes_i():
    import record_suite as RS
    st = RS.agx_loop.store
    pre = RS.mov(27, 4) + RS.e('imadd r10, r1, r27, 0') + RS.mov(27, I_SRC) + RS.e('iadd r10, r10, r27')
    pre += RS.e('device_load 0, i32, xyzw, r6_r7_r8_r9, u8_u9, r10, unsigned') + RS.e('wait 0')
    pre += RS.mov(27, 2) + RS.e('imadd r5, r1, r27, 0') + RS.mov(14, 0xdeadbeef) + RS.mov(15, 0xdeadbeef)
    for reg, off in ((21, I_R14), (22, I_R15)):
        pre += RS.mov(27, off) + RS.e(f'iadd r{reg}, r1, r27')
    suf = RS.e('wait 0') + st(14, 8, 21) + st(15, 8, 22) + RS.e('wait 0') + RS.e('iadd r2, r14, 0')
    t = table_i()
    tab = list(t); lanes = []
    for i in range(N):
        old = t[2 * i]; lanes.append(old)
        if old == t[I_SRC + 4 * i + 1]:
            tab[2 * i] = t[I_SRC + 4 * i]
        tab[I_R14 + i], tab[I_R15 + i] = old, 0xdeadbeef
    return [PV.Probe('I_op3_u50_cas', pre, atomic_bytes(3, 1, 2), atomic_bytes(3, 1, 0), suf, 3 << 45, N, WG, REGS, t,
                     lambda o, tb: out_ok(o, lanes) and tb == tab)]


def analyse_i(r):
    t0, t1 = r['table_in'], r['table_after']
    models = {
        'cas32 (new r6, cmp r7)': lambda m, s: ((m[0] if m[0] != s[1] else s[0]), m[1]),
        'cas64 (new r6_r7, cmp r8_r9)': lambda m, s: (s[0], s[1]) if m == (s[2], s[3]) else m,
        'cas64 (cmp r6_r7, new r8_r9)': lambda m, s: (s[2], s[3]) if m == (s[0], s[1]) else m,
        'none': lambda m, s: m}
    fits = {}
    for name, f in models.items():
        fits[name] = sum(f((t0[2 * i], t0[2 * i + 1]), tuple(t0[I_SRC + 4 * i:I_SRC + 4 * i + 4])) == (t1[2 * i], t1[2 * i + 1])
                         for i in range(N))
    r14 = [t1[I_R14 + i] for i in range(N)]; r15 = [t1[I_R15 + i] for i in range(N)]
    return {'memory_fit_lanes': fits, 'r14_old_lo': sum(r14[i] == t0[2 * i] for i in range(N)),
            'r15_old_hi': sum(r15[i] == t0[2 * i + 1] for i in range(N)), 'r15_untouched': sum(x == 0xdeadbeef for x in r15)}


# ---------------------------------------------------------------------------------------------------------------

def analyse(d):
    rows = []
    for p in sorted(Path(d).glob('*.json')):
        r = json.loads(p.read_text())
        if 'fault' in r:
            rows.append({'probe': r['name'], 'fault': r['fault'][:160]}); continue
        if r['name'].startswith('I_'):
            rows.append({'probe': r['name'], 'deterministic': r['deterministic'], **analyse_i(r)}); continue
        if r['name'].startswith('H_'):
            rows.append({'probe': r['name'], 'deterministic': r['deterministic'], **analyse_h(r)}); continue
        if r['name'].startswith('G_'):
            rows.append({'probe': r['name'], 'deterministic': r['deterministic'], **analyse_g(r)}); continue
        if r['name'].startswith('F_'):
            rows.append({'probe': r['name'], 'deterministic': r['deterministic'], **analyse_f(r)}); continue
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
    r = sub.add_parser('run'); r.add_argument('--family', choices=['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I'], required=True)
    r.add_argument('--only'); r.add_argument('--dry', action='store_true'); r.add_argument('--out', type=Path, default=OUT / 'raw')
    s = sub.add_parser('analyse'); s.add_argument('dir', type=Path, nargs='?', default=OUT / 'raw')
    a = ap.parse_args()
    if a.cmd == 'run':
        PV.cov.install()
        ps = {'A': probes_a, 'B': probes_b, 'D': probes_d, 'E': lambda: probes_e() + probes_e(before=True) + probes_e(before=True, skip=True), 'F': probes_f, 'G': probes_g, 'H': probes_h, 'I': probes_i}[a.family]()
        if a.only:
            ps = [p for p in ps if p.name == a.only]
        PV.run(ps, a.out, dry=a.dry)
    else:
        for row in analyse(a.dir):
            print(json.dumps(row))


if __name__ == '__main__':
    main()
