#!/usr/bin/env python3
"""Targeted microarchitecture probes behind the cost model (direct DRM, known-safe encodings only).

    python3 scripts/cost_probes.py lead|rotation|pairs|half [--output DIR]

All bodies pass run_bench.check_body and run in the counted-loop skeleton; cycles come from the
in-dispatch clock (reference units, see the instruction-bench finding; x1.049 for real cycles).
  lead     : a leader instruction + k independent iadds (k = 0, 1, 2, 4, 8), 8 repetitions per trip
  rotation : 72 independent instructions rotating over D destination registers (D = 4..16)
  pairs    : dependent chains alternating two instructions from different units
  half     : 16-bit / 32-bit partial-register cases
Results print per test and are saved as JSON (one value: cycles per repetition, or per instruction).
"""
import argparse, json
from pathlib import Path
import agx_loop, cost_hw as H, run_bench as B, gpu_safety

e = agx_loop.encode
EMPTY_TRIP = 35.25      # measured empty-loop trip (cost-model calibration)


def per_rep(R, lines, reps=8):
    parts = [e(x) for x in lines]*reps
    body = b''.join(parts)
    if body: B.check_body(body, [len(x) for x in parts])
    m = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
    return (m['cycles']-EMPTY_TRIP)/reps, m['mhz']


def ind(k, regs=(7, 8, 9, 10, 11, 12, 13, 14)):
    return [f'iadd r{r}, r{r}, 1' for r in regs[:k]]


LEADS = {'rcp': 'rcp r6, r6', 'log2': 'log2 r6, r6', 'floor': 'floor r6, r6', 'sin_pt_1': 'sin_pt_1 r6, r6',
         'bfi': 'bfi r6, r6, r24, r26', 'convert': 'convert u32_to_f, r6, r6, rte', 'imadd': 'imadd r6, r6, r25, r24',
         'simd_shuffle': 'simd_shuffle r6, r6, r26l', 'fadd32': 'fadd32 r6, r6, r24', 'simd_iadd': 'simd_iadd r6, r6'}
PAIRS = {'iadd>iadd': ('iadd r6, r6, r24', 'iadd r6, r6, r24'), 'fadd32>iadd': ('fadd32 r6, r6, r24', 'iadd r6, r6, r24'),
         'iadd>bfi': ('iadd r6, r6, r24', 'bfi r6, r6, r24, r26'), 'fadd32>log2': ('fadd32 r6, r6, r24', 'log2 r6, r6'),
         'iadd>simd_shuffle': ('iadd r6, r6, r24', 'simd_shuffle r6, r6, r26l'), 'fadd32>convert': ('fadd32 r6, r6, r24', 'convert u32_to_f, r6, r6, rte'),
         'fadd16>fadd32': ('fadd16 r6l, r6l, r24l', 'fadd32 r6, r6, r24'), 'iadd>fmadd32': ('iadd r6, r6, r24', 'fmadd32 r6, r6, r25, r24'),
         'simd_iadd>iadd': ('simd_iadd r6, r6', 'iadd r6, r6, r24')}
HALF = {'f32 write r6 then independent f16 write r6l': ['fadd32 r6, r8, r24', 'fadd16 r6l, r7l, r24l'],
        'f16 write r6l then f32 read of r6 (merge)': ['fadd16 r6l, r6l, r24l', 'fadd32 r9, r6, r24'],
        'f16 write r6l then f16 read of r6l': ['fadd16 r6l, r6l, r24l', 'fadd16 r9l, r6l, r24l'],
        'f16 write r6l then f32 read of r7 (control)': ['fadd16 r6l, r6l, r24l', 'fadd32 r9, r7, r24'],
        'simd_iadd then dependent iadd': ['simd_iadd r6, r6', 'iadd r6, r6, r24'],
        'simd_iadd then independent iadd': ['simd_iadd r6, r6', 'iadd r7, r7, r24'],
        'simd_iadd then dependent fadd32': ['simd_iadd r6, r6', 'fadd32 r6, r6, r24'],
'f32 then f16 read of low half': ['fadd32 r6, r6, r24', 'fadd16 r6l, r6l, r24l'],
        'f16 chain': ['fadd16 r6l, r6l, r24l', 'fadd16 r6l, r6l, r24l'],
        'f16 low/high independent': ['fadd16 r6l, r6l, r24l', 'fadd16 r6h, r6h, r24l'],
        'f16 write r6l then independent f32 write r6': ['fadd16 r6l, r7l, r24l', 'fadd32 r6, r8, r24']}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['lead', 'rotation', 'pairs', 'half', 'values', 'values2', 'groups', 'memory', 'loadgrid', 'tgmem', 'memstate', 'waits', 'spaces', 'sfumix', 'sfuswitch', 'review2', 'waitdrain', 'multiload', 'sfupairs', 'sfuimad'])
    ap.add_argument('--output', type=Path, default=Path('/scratch/apple-m2-playground/cost/probes'))
    a = ap.parse_args()
    gpu_safety.require_budget(1)
    R = B.Runner(a.output/a.what)
    res = {}
    if a.what == 'lead':
        for lead, li in LEADS.items():
            for k in (0, 1, 2, 4, 8):
                res[f'{lead}+{k}'] = per_rep(R, [li]+ind(k))[0]
            print(lead, ' '.join(f"+{k}: {res[f'{lead}+{k}']:.2f}" for k in (0, 1, 2, 4, 8)), flush=True)
    elif a.what == 'rotation':
        for instr in ('iadd r{r}, r{r}, 1', 'fadd32 r{r}, r{r}, 0.125', 'fadd16 r{r}l, r{r}l, r24l'):
            for D in (4, 6, 7, 8, 9, 10, 12, 16):
                lines = [instr.format(r=5+k % D) for k in range(72)]
                c, mhz = per_rep(R, lines, reps=1)
                res[f'{instr}|{D}'] = c/72
                print(f'{instr.split()[0]:6s} D={D:2d} cycles/instr {c/72:.2f}', flush=True)
    elif a.what == 'pairs':
        for name, (x, y) in PAIRS.items():
            res[name] = per_rep(R, [x, y])[0]; print(f'{name:20s} cycles/pair {res[name]:.2f}', flush=True)
    elif a.what == 'groups':
        # latency-bound single chains: per-trip cycles with 1 SIMD group vs ~4 per core (per-core span)
        G = {'iadd chain': ['iadd r6, r6, r24'], 'fadd32 chain': ['fadd32 r6, r6, r24'], 'imadd chain': ['imadd r6, r6, r25, r24'],
             'rcp chain': ['rcp r6, r6'], 'shuffle chain': ['simd_shuffle r6, r6, r26l'], 'fmadd32 chain': ['fmadd32 r6, r6, r25, r24'],
             'iadd 2 chains': ['iadd r6, r6, r24', 'iadd r7, r7, r24'], 'mixed chain': ['iadd r6, r6, r24', 'fadd32 r6, r6, r24', 'imadd r6, r6, r25, r24', 'bfi r6, r6, r24, r26']}
        for name, lines in G.items():
            parts = [e(x) for x in lines]*8; body = b''.join(parts)
            B.check_body(body, [len(x) for x in parts])
            one = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
            four = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'gpu')
            res[name] = {'one': one['cycles'], 'four_per_core': four['cycles'], 'ratio': four['cycles']/one['cycles'], 'groups_per_core': four['groups_per_core']}
            print(f"{name:16s} 1 group {one['cycles']:7.1f}  4/core {four['cycles']:7.1f}  ratio {res[name]['ratio']:.3f}", flush=True)
    elif a.what == 'memory':
        # device memory costs. 'line 1' loads read table words 32..63 (no other writer during those tests);
        # the first variants read word 0, the CPU clock line (coherence traffic: not an ordinary hit);
        # stores write table words 32..63 (between the clock line and the timestamp regions).
        st = agx_loop.store(5, 8, 14)
        M = {'control: iadd': ['iadd r14, r1, 32'],
             'load line 1 + wait 0': ['iadd r13, r1, 32', 'device_load 0, i32, x, r6, u8_u9, r13, unsigned', 'wait 0'],
             '2 loads line 1 + one wait': ['iadd r13, r1, 32', 'device_load 0, i32, x, r6, u8_u9, r13, unsigned', 'device_load 0, i32, x, r7, u8_u9, r13, unsigned', 'wait 0'],
             'load line 1 + wait + dependent iadd': ['iadd r13, r1, 32', 'device_load 0, i32, x, r6, u8_u9, r13, unsigned', 'wait 0', 'iadd r6, r6, r24'],
             'load + wait 0 (L1 hit)': ['iadd r14, r1, 32', 'device_load 0, i32, x, r6, u8_u9, r13, unsigned', 'wait 0'],
             'load + wait 0 + dependent iadd': ['iadd r14, r1, 32', 'device_load 0, i32, x, r6, u8_u9, r13, unsigned', 'wait 0', 'iadd r6, r6, r24'],
             '2 loads + one wait 0': ['iadd r14, r1, 32', 'device_load 0, i32, x, r6, u8_u9, r13, unsigned', 'device_load 0, i32, x, r7, u8_u9, r13, unsigned', 'wait 0'],
             'store (no wait)': ['iadd r14, r1, 32', st],
             'store + wait 0': ['iadd r14, r1, 32', st, 'wait 0']}
        for name, lines in M.items():
            parts = [x if isinstance(x, bytes) else e(x) for x in lines]*8
            body = b''.join(parts)
            B.check_body(body, [len(x) for x in parts]) if False else None
            m = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
            res[name] = (m['cycles']-EMPTY_TRIP)/8
            print(f'{name:34s} cycles/rep {res[name]:7.2f}', flush=True)
    elif a.what == 'loadgrid':
        # a ALU instructions before a device load, b after it, then wait 0 (load-after-ALU finding)
        consts = (1, 0x3f800000, 32, 7)
        L = 'device_load 0, i32, x, r6, u8_u9, r26, unsigned'     # word 32 (r26 = 32, set before the loop)
        for a_ in (0, 1, 2, 4, 8):
            for b_ in (0, 4, 8, 16, 24, 32):
                pre = ['iadd r%d, r%d, 1' % (14+i % 4, 14+i % 4) for i in range(a_)]
                post = ['iadd r%d, r%d, 1' % (18+i % 4, 18+i % 4) for i in range(b_)]
                body = b''.join(e(x) for x in pre+[L]+post+['wait 0'])*8
                m = H.measure(R, H.counted(body, 100, consts=consts), H.counted(body, 500, consts=consts), 400, 'one')
                res[f'{a_},{b_}'] = (m['cycles']-EMPTY_TRIP)/8
                print(f'a={a_} b={b_:2d}: {res[f"{a_},{b_}"]:.1f}', flush=True)
    elif a.what == 'tgmem':
        # threadgroup memory (2 KiB declared; encodings from the threadgroup-addressing finding).
        from run_drm_tg_layout import tg, shared_word
        env = {'AGX_SHARED_WORD': str(shared_word(2048))}
        LD = tg(True, 'i32', 6, base=None, index=0); ST = tg(False, 'i32', 5, base=None, index=1)
        M = {'tg load, wait': [LD, e('wait 0')],
             'iadd, tg load, wait': [e('iadd r14, r14, 1'), LD, e('wait 0')],
             'tg load, wait, dependent iadd': [LD, e('wait 0'), e('iadd r6, r6, r24')],
             '2 tg loads, wait': [LD, tg(True, 'i32', 7, base=None, index=2), e('wait 0')],
             'tg store': [ST],
             'tg store, wait': [ST, e('wait 0')],
             'tg store, tg load (other slot), wait': [ST, LD, e('wait 0')]}
        for name, parts in M.items():
            body = b''.join(parts)*8
            m = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one', extra_env=env)
            res[name] = (m['cycles']-EMPTY_TRIP)/8
            print(f'{name:38s} {res[name]:6.1f} ref units/rep', flush=True)
    elif a.what == 'memstate':
        # what makes a memory instruction slow: ALU work issued while no memory instruction is outstanding
        from run_drm_tg_layout import tg, shared_word
        env = {'AGX_SHARED_WORD': str(shared_word(2048))}
        consts = (1, 0x3f800000, 32, 7)
        st = agx_loop.store(5, 8, 26); L = e('device_load 0, i32, x, r6, u8_u9, r26, unsigned')
        A = e('iadd r14, r14, 1'); W = e('wait 0'); TS = tg(False, 'i32', 5, base=None, index=1)
        I8 = b''.join(e('iadd r%d, r%d, 1' % (r, r)) for r in range(15, 23))
        M = {'store': [st], 'store, wait': [st, W], 'iadd, store': [A, st], 'iadd, store, wait': [A, st, W],
             'store, store': [st, st], 'store, 8 iadd': [st, I8], 'iadd, wait, store': [A, W, st],
             'iadd, store, store': [A, st, st], 'tg store': [TS], 'iadd, tg store': [A, TS],
             'iadd, tg store, store': [A, TS, st], 'load, wait': [L, W], 'load, iadd, wait': [L, A, W],
             'iadd, load, wait': [A, L, W], 'iadd, wait, load, wait': [A, W, L, W], 'iadd, store, load, wait': [A, st, L, W]}
        for name, parts in M.items():
            body = b''.join(parts)*8
            m = H.measure(R, H.counted(body, 100, consts=consts), H.counted(body, 500, consts=consts), 400, 'one', extra_env=env)
            res[name] = (m['cycles']-EMPTY_TRIP)/8
            print(f'{name:26s} {res[name]:6.1f} ref units/rep', flush=True)
    elif a.what == 'sfupairs':
        # throughput-bound pairs of transcendentals: 8 independent chains, two op types alternating,
        # 1 group and ~4 groups per core; is a mix slower than the sum of each op's shared-unit cost?
        OPS_ = {'log2': 'log2 r{r}, r{r}', 'floor': 'floor r{r}, r{r}', 'rcp': 'rcp r{r}, r{r}', 'exp2': 'exp2 r{r}, r{r}',
                'rsqrt': 'rsqrt r{r}, r{r}'}
        PAIRS = [('log2', 'log2'), ('rcp', 'rcp'), ('floor', 'floor'), ('log2', 'rcp'), ('floor', 'rcp'), ('log2', 'floor'),
                 ('log2', 'exp2'), ('rsqrt', 'log2'), ('rsqrt', 'rcp')]
        for a_, b_ in PAIRS:
            lines = [OPS_[a_ if k % 2 == 0 else b_].format(r=5+k) for k in range(8)]
            body = b''.join(e(x) for x in lines)*4
            B.check_body(body, [len(e(x)) for x in lines]*4)
            one = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
            four = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'gpu')
            p1 = H.predict(H.counted(body, 100), H.counted(body, 500), 400, 1)
            p4 = H.predict(H.counted(body, 100), H.counted(body, 500), 400, 4)
            name = f'{a_}+{b_}'
            res[name] = {'one': one['cycles'], 'four': four['cycles'], 'pred_one': p1, 'pred_four': p4}
            print(f"{name:12s} 1g {one['cycles']:6.1f} (pred {p1:6.1f})  4/core {four['cycles']:6.1f} (pred {p4:6.1f})", flush=True)
    elif a.what == 'sfuimad':
        # do integer-multiply/bit-field/convert results share the transcendentals' in-order writeback?
        OPS_ = {'log2': 'log2 r{r}, r{r}', 'floor': 'floor r{r}, r{r}', 'rcp': 'rcp r{r}, r{r}', 'imadd': 'imadd r{r}, r{r}, r25, r24',
                'bitrev': 'bitrev r{r}, r{r}', 'convert': 'convert u32_to_f, r{r}, r{r}, rte', 'fadd32': 'fadd32 r{r}, r{r}, r24'}
        PAIRS = [('imadd', 'imadd'), ('bitrev', 'bitrev'), ('log2', 'imadd'), ('log2', 'bitrev'), ('log2', 'convert'), ('floor', 'imadd'),
                 ('rcp', 'imadd'), ('log2', 'fadd32'), ('convert', 'convert')]
        for a_, b_ in PAIRS:
            lines = [OPS_[a_ if k % 2 == 0 else b_].format(r=5+k) for k in range(8)]
            body = b''.join(e(x) for x in lines)*4
            B.check_body(body, [len(e(x)) for x in lines]*4)
            one = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
            four = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'gpu')
            p1 = H.predict(H.counted(body, 100), H.counted(body, 500), 400, 1)
            p4 = H.predict(H.counted(body, 100), H.counted(body, 500), 400, 4)
            name = f'{a_}+{b_}'
            res[name] = {'one': one['cycles'], 'four': four['cycles'], 'pred_one': p1, 'pred_four': p4}
            print(f"{name:16s} 1g {one['cycles']:6.1f} (pred {p1:6.1f})  4/core {four['cycles']:6.1f} (pred {p4:6.1f})", flush=True)
    elif a.what == 'multiload':
        # several device loads in flight, ALU work between them, and loads writing the same register
        consts = (1, 0x3f800000, 32, 7)
        L = lambda d, i=26: e(f'device_load 0, i32, x, r{d}, u8_u9, r{i}, unsigned')
        A = lambda k=1: b''.join(e('iadd r%d, r%d, 1' % (5+i % 8, 5+i % 8)) for i in range(k))
        W = e('wait 0')
        M = {'load13, load14, wait': [L(13), L(14), W], 'iadd, load13, iadd, load14, wait': [A(), L(13), A(), L(14), W],
             'iadd, load13, iadd, load13, wait (WAW)': [A(), L(13), A(), L(13), W], 'iadd, load13, load13, wait (WAW)': [A(), L(13), L(13), W],
             'load13, 4 iadd, load14, 4 iadd, wait': [L(13), A(4), L(14), A(4), W],
             'iadd, load13, iadd, load14, iadd, load15, iadd, wait': [A(), L(13), A(), L(14), A(), L(15), A(), W],
             '3 loads, wait': [L(13), L(14), L(15, 27), W]}
        for name, parts in M.items():
            body = b''.join(parts)*8
            m = H.measure(R, H.counted(body, 100, consts=consts), H.counted(body, 500, consts=consts), 400, 'one')
            p_ = (H.predict(H.counted(body, 100, consts=consts), H.counted(body, 500, consts=consts), 400, 1)-EMPTY_TRIP)/8
            res[name] = {'measured': (m['cycles']-EMPTY_TRIP)/8, 'predicted': p_}
            print(f"{name:52s} meas {res[name]['measured']:6.1f} pred {p_:6.1f}", flush=True)
    elif a.what == 'waitdrain':
        # does an idle wait drain in-flight ALU results? X alone vs X followed by wait 0 (independent X)
        X = {'iadd': 'iadd r{r}, r{r}, 1', 'imadd': 'imadd r{r}, r{r}, r25, r24', 'rcp': 'rcp r{r}, r{r}',
             'simd_shuffle': 'simd_shuffle r{r}, r{r}, r26l', 'simd_iadd': 'simd_iadd r{r}, r{r}', 'fadd32': 'fadd32 r{r}, r{r}, r24'}
        for name, tmpl in X.items():
            for with_wait in (False, True):
                parts = []
                for k in range(8):
                    parts.append(e(tmpl.format(r=5+k % 8)))
                    if with_wait: parts.append(e('wait 0'))
                body = b''.join(parts)
                m = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
                res[f'{name}|{with_wait}'] = (m['cycles']-EMPTY_TRIP)/8
            print(f"{name:13s} alone {res[f'{name}|False']:6.2f}  + wait {res[f'{name}|True']:6.2f}  (wait adds {res[f'{name}|True']-res[f'{name}|False']:5.2f})", flush=True)
    elif a.what == 'review2':
        # decisive probes requested by the second review (all archived)
        from run_drm_tg_layout import tg, shared_word
        env = {'AGX_SHARED_WORD': str(shared_word(2048))}
        consts = (1, 0x3f800000, 32, 7)
        st = agx_loop.store(5, 8, 26); L = e('device_load 0, i32, x, r6, u8_u9, r26, unsigned')
        A = e('iadd r14, r14, 1'); W0 = e('wait 0'); W1 = e('wait 1')
        TL = tg(True, 'i32', 7, base=None, index=0)
        I = lambda k: b''.join(e('iadd r%d, r%d, 1' % (15+i % 8, 15+i % 8)) for i in range(k))
        M = {}
        for k in (1, 2, 4, 8, 16): M[f'A store, {k} iadd'] = [st, I(k)]                 # count vs time
        for k in (0, 1, 2, 3, 4, 6, 8):
            M[f'B iadd, {k} wait1, load, wait0'] = [A, W1*k, L, W0]
            M[f'B iadd, {k} wait1, store'] = [A, W1*k, st]
        M['C tg load, iadd, wait'] = [TL, A, W0]; M['C tg load, 8 iadd, wait'] = [TL, I(8), W0]
        M['C tg load, wait'] = [TL, W0]
        M['D 8 wait + 6 iadd (532 B, 112 instr)'] = [W0*8, I(6)]
        M['D 8 wait + 4 iadd (404 B, 96 instr)'] = [W0*8, I(4)]
        M['E iadd, store, load, wait'] = [A, st, L, W0]; M['E store, load, wait'] = [st, L, W0]
        M['E load, store, wait'] = [L, st, W0]
        for name, parts in M.items():
            body = b''.join(parts)*8
            m = H.measure(R, H.counted(body, 100, consts=consts), H.counted(body, 500, consts=consts), 400, 'one', extra_env=env)
            res[name] = (m['cycles']-EMPTY_TRIP)/8
            print(f'{name:42s} {res[name]:6.1f} units/rep', flush=True)
    elif a.what == 'sfuswitch':
        # one SIMD group: independent transcendentals of one function vs mixed functions
        V = {'log2 x3': ['log2 r6, r6', 'log2 r7, r7', 'log2 r8, r8'], 'floor x3': ['floor r6, r6', 'floor r7, r7', 'floor r8, r8'],
             'rcp x3': ['rcp r6, r6', 'rcp r7, r7', 'rcp r8, r8'], 'log2, floor, rcp': ['log2 r6, r6', 'floor r7, r7', 'rcp r8, r8'],
             'log2, exp2, log2, exp2': ['log2 r6, r6', 'exp2 r7, r7', 'log2 r8, r8', 'exp2 r9, r9'],
             'log2 x4': ['log2 r6, r6', 'log2 r7, r7', 'log2 r8, r8', 'log2 r9, r9'],
             'floor, rint, floor, rint': ['floor r6, r6', 'rint r7, r7', 'floor r8, r8', 'rint r9, r9']}
        for name, lines in V.items():
            body = b''.join(e(x) for x in lines)*8
            one = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
            p1 = H.predict(H.counted(body, 100), H.counted(body, 500), 400, 1)
            res[name] = {'one': one['cycles'], 'pred_one': p1}
            print(f"{name:26s} 1g {(one['cycles']-35.25)/8:6.2f}/rep (pred {(p1-35.25)/8:6.2f})", flush=True)
    elif a.what == 'sfumix':
        # transcendental contention between SIMD groups: 1 group vs ~4 per core, with model predictions
        V = {'rcp chain + 3 iadd': ['rcp r6, r6', 'iadd r7, r7, 1', 'iadd r8, r8, 1', 'iadd r9, r9, 1'],
             'log2, floor, rcp (independent)': ['log2 r6, r6', 'floor r7, r7', 'rcp r8, r8'],
             '4 independent rcp': ['rcp r6, r6', 'rcp r7, r7', 'rcp r8, r8', 'rcp r9, r9'],
             'rcp + fadd32 + imadd': ['rcp r6, r6', 'fadd32 r7, r7, r24', 'imadd r8, r8, r25, r24'],
             '8 independent floor': ['floor r%d, r%d' % (r, r) for r in range(5, 13)],
             'control: 4 iadd chains': ['iadd r6, r6, 1', 'iadd r7, r7, 1', 'iadd r8, r8, 1', 'iadd r9, r9, 1']}
        for name, lines in V.items():
            body = b''.join(e(x) for x in lines)*8
            B.check_body(body, [len(e(x)) for x in lines]*8)
            one = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
            four = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'gpu')
            p1 = H.predict(H.counted(body, 100), H.counted(body, 500), 400, 1)
            p4 = H.predict(H.counted(body, 100), H.counted(body, 500), 400, 4)
            res[name] = {'one': one['cycles'], 'four': four['cycles'], 'pred_one': p1, 'pred_four': p4}
            print(f"{name:32s} 1g {one['cycles']:6.1f} (pred {p1:6.1f})  4/core {four['cycles']:6.1f} (pred {p4:6.1f})  ratio {four['cycles']/one['cycles']:.2f}", flush=True)
    elif a.what == 'spaces':
        # memory-space transitions with no ALU work in the loop body
        from run_drm_tg_layout import tg, shared_word
        env = {'AGX_SHARED_WORD': str(shared_word(2048))}
        consts = (1, 0x3f800000, 32, 7)
        DS = agx_loop.store(5, 8, 26); DL = e('device_load 0, i32, x, r6, u8_u9, r26, unsigned')
        TS = tg(False, 'i32', 5, base=None, index=1); TL = tg(True, 'i32', 7, base=None, index=0); W = e('wait 0')
        M = {'dev store, dev store': [DS, DS], 'tg store, tg store': [TS, TS], 'dev store, tg store': [DS, TS],
             'tg store, dev store': [TS, DS], 'dev load, wait, dev load, wait': [DL, W, DL, W],
             'tg load, wait, tg load, wait': [TL, W, TL, W], 'dev load, wait, tg load, wait': [DL, W, TL, W],
             'tg load, wait, dev load, wait': [TL, W, DL, W], 'dev store, tg load, wait': [DS, TL, W],
             'tg store, dev load, wait': [TS, DL, W]}
        for name, parts in M.items():
            body = b''.join(parts)*8
            m = H.measure(R, H.counted(body, 100, consts=consts), H.counted(body, 500, consts=consts), 400, 'one', extra_env=env)
            res[name] = (m['cycles']-EMPTY_TRIP)/8
            print(f'{name:34s} {res[name]:6.1f} units/rep', flush=True)
    elif a.what == 'waits':
        # cost of wait instructions whose slot has nothing pending (no memory instructions in the loop)
        for slot in (0, 1):
            for k in (1, 2, 4, 8, 16, 24, 28, 30, 31, 32, 40):
                body = (e(f'wait {slot}')*k)*8
                m = H.measure(R, H.counted(body, 100), H.counted(body, 500), 400, 'one')
                res[f'{slot},{k}'] = (m['cycles']-EMPTY_TRIP)/8
                print(f'wait {slot} x {k:2d} per rep: {res[f"{slot},{k}"]:7.1f} units/rep  ({res[f"{slot},{k}"]/k:5.2f} per wait)', flush=True)
    elif a.what == 'values2':
        # identical instruction sequences; only the preloaded constants differ (r26/r27 set before the loop)
        one, near1, half, zero = 0x3f800000, 0x3f800347, 0x3f000000, 0
        V = {'fmul32 x*r27, r27=1.0': (['fmul32 r6, r6, r27'], (1, one, 5, one)),
             'fmul32 x*r27, r27=1.0001': (['fmul32 r6, r6, r27'], (1, one, 5, near1)),
             'fmadd32 x*r27+r26, 1.0/0.0': (['fmadd32 r6, r6, r27, r26'], (1, one, zero, one)),
             'fmadd32 x*r27+r26, 1.0001/0.5': (['fmadd32 r6, r6, r27, r26'], (1, one, half, near1)),
             'fadd32 x+r26, r26=0.0': (['fadd32 r6, r6, r26'], (1, one, zero, one)),
             'fadd32 x+r26, r26=0.5': (['fadd32 r6, r6, r26'], (1, one, half, one))}
        for name, (lines, consts) in V.items():
            parts = [e(x) for x in lines]*8; body = b''.join(parts)
            B.check_body(body, [len(x) for x in parts])
            m = H.measure(R, H.counted(body, 100, consts=consts), H.counted(body, 500, consts=consts), 400, 'one')
            res[name] = (m['cycles']-EMPTY_TRIP)/8; print(f'{name:34s} cycles/rep {res[name]:.2f}', flush=True)
    elif a.what == 'values':
        # same instruction sequences with different data: benchmark chains (NaN/inf/trivial) vs finite values.
        # r24 = 1 (int) / 0.0f, r25 = 1.0f; the loop preloads r5..r12 with 1.5f (cost_hw.counted).
        V = {'log2 chain (1.5 -> NaN after 3 steps)': ['log2 r6, r6'],
             'exp2 chain (1.5 -> +inf after 4 steps)': ['exp2 r6, r6'],
             'log2/exp2 alternating (stays ~1.5)': ['log2 r6, r6', 'exp2 r6, r6'],
             'fmul32 x*1.0 (trivial)': ['fmul32 r6, r6, r25'],
             'fmul32 x*1.0001 (r26 = 1.0001f)': ['mov_imm r26, 1065354055, 0b0', 'fmul32 r6, r6, r26'],
             'fmadd32 x*1+0 (trivial)': ['fmadd32 r6, r6, r25, r24'],
             'fmadd32 x*1.0001+0.5': ['mov_imm r26, 1065354055, 0b0', 'mov_imm r27, 1056964608, 0b0', 'fmadd32 r6, r6, r26, r27'],
             'rcp chain (1.5 <-> 0.667)': ['rcp r6, r6'],
             'rcp of 0 -> inf -> 0': ['mov_imm r6, 0, 0b0', 'rcp r6, r6', 'rcp r6, r6']}
        for name, lines in V.items():
            res[name] = per_rep(R, lines)[0]; print(f'{name:42s} cycles/rep {res[name]:.2f}', flush=True)
    else:
        for name, lines in HALF.items():
            res[name] = per_rep(R, lines)[0]; print(f'{name:45s} cycles/pair {res[name]:.2f}', flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/f'{a.what}.json').write_text(json.dumps(res, indent=1)+'\n')


if __name__ == '__main__':
    main()
