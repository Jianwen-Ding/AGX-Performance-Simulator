#!/usr/bin/env python3
"""Hardware calibration and validation of agx_cost (direct DRM, known-safe encodings only).

calibrate : taken-loop overhead per trip (empty body) and get_sr cost, in cycles, with the
            in-dispatch clock of run_bench (REF slope = ns per cycle).
validate  : random kernels: L instructions drawn from the benchmarked templates with random
            register dependencies among r5..r12 (constants r24..r26), run in a counted loop.
            Cycles per trip = slope between two trip counts, measured with 1 SIMD group and
            with ~4 groups per core (per-core span aggregation), and compared with agx_cost.
Every body passes run_bench.check_body (decoded lengths, no control flow, dest >= r5).
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path
import agx_loop, gpu_safety, run_bench as B, agx_cost as C
from bench_ops import OPS
from run_drm_opcode_describe import get_sr

e, mov = agx_loop.encode, B.mov
POOL = [k for k, (t, c) in OPS.items() if c not in ('int64',) and '{d1}' not in t]


def rand_instr(rng, ops):
    op = rng.choice(ops)
    t, cls = OPS[op]
    d = rng.randint(5, 12); s = rng.randint(5, 12)
    t = t.replace('r20', 'r24').replace('r21', 'r25').replace('r22', 'r26')
    first = t.index('{d}')
    second = t.find('{d}', first+3)
    if second >= 0: t = t[:second]+'{s}'+t[second+3:]
    return op, t.format(d=d, s=s)


def kernel(seed, length, ops=POOL):
    rng = random.Random(seed)
    lines = [rand_instr(rng, ops) for _ in range(length)]
    parts = [e(x) for _, x in lines]
    body = b''.join(parts)
    B.check_body(body, [len(x) for x in parts])
    return lines, body


MEM_CONSTS = (1, 0x3f800000, 32, 7)     # r26 = 32 (table word 32, the only store target), r27 = 7 (read-only line)


def mem_kernel(seed, length, p_mem=0.15):
    """Random kernel mixing benchmarked ALU forms with memory instructions. ALU instructions use r5..r12;
    loads write r13..r15, which nothing reads, so no load result is consumed without its wait. Addresses:
    device loads from words 32 / 7 (r26 / r27), device stores only to word 32, threadgroup loads/stores at
    immediate slots 0..7 (2 KiB declared). Random wait 0 instructions drain the loads/stores."""
    from run_drm_tg_layout import tg
    rng = random.Random(seed)
    lines, parts = [], []
    for _ in range(length):
        x = rng.random()
        if x < p_mem:
            kind = rng.choice(['dload', 'dload', 'dstore', 'tgload', 'tgstore'])
            d = rng.randint(13, 15); src = rng.randint(5, 12); slot = rng.randint(0, 7)
            if kind == 'dload':
                idx = rng.choice([26, 27]); t = f'device_load 0, i32, x, r{d}, u8_u9, r{idx}, unsigned'; b = e(t)
            elif kind == 'dstore':
                b = agx_loop.store(src, 8, 26); t = f'device_store r{src} -> word 32'
            elif kind == 'tgload':
                b = tg(True, 'i32', d, base=None, index=slot); t = f'threadgroup_load r{d} <- slot {slot}'
            else:
                b = tg(False, 'i32', src, base=None, index=slot); t = f'threadgroup_store r{src} -> slot {slot}'
        elif x < p_mem+0.06:
            b = e('wait 0'); t = 'wait 0'
        else:
            _, t = rand_instr(rng, POOL); b = e(t)
        lines.append(t); parts.append(b)
    lines.append('wait 0'); parts.append(e('wait 0'))
    body = b''.join(parts)
    mem_gate(body, [len(x) for x in parts])
    return lines, body


def mem_gate(body, lengths):
    """Decode-check a mixed body: exact lengths; only known ALU / memory / wait instructions; ALU
    destinations >= r5; load destinations r13..r15; no control flow."""
    _, g = agx_loop._tools()
    p = 0
    for want in lengths:
        n = g.opcode_to_number(body[p:p+16]); d = g.get_instruction_descriptor(n)
        if d is None or d.decode_size(n) != want: raise ValueError(f'length mismatch at {p}')
        n &= (1 << 8*want)-1; f = dict(d.decode_fields(n))
        if d.name in ('device_load', 'threadgroup_load'):
            if not 26 <= f['R'] <= 31: raise ValueError(f'load destination outside r13..r15 at {p}')
        elif d.name in ('device_store', 'threadgroup_store', 'wait'):
            pass
        elif d.name in B.CONTROL:
            raise ValueError(f'control flow in body: {d.name}')
        else:
            if f.get('D', 0) < 10: raise ValueError(f'{d.name} writes below r5')
        p += want


def counted(body, trips, init=True, consts=(1, 0x3f800000, 5, 7)):
    """Counted loop around `body`; r24..r27 preloaded with `consts`, r5..r12 with 1.5f."""
    head = mov(3, 0)+mov(4, trips)
    if init: head += b''.join(mov(24+k, v) for k, v in enumerate(consts))+b''.join(mov(r, 0x3fc00000) for r in range(5, 13))
    head += e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    lp = body+e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
    lp += e(f'jmp_exec_any pc-{len(lp)}', f'jmp_exec_any pc-{len(lp)}')
    return head+lp+e('pop_exec r0l, 1')+e('iadd r2, r5, 0')


def program(phase_a, phase_b, count, ref_trips=1000, warm=20000):
    from run_drm_opcode_describe import get_sr as gs
    core = gs(20, 28)+mov(27, B.TS_BASE+6*count)+e('iadd r27, r1, r27')+agx_loop.store(28, 8, 27)+e('wait 0')
    ph = [B.loop(B.REF, 1, 16, ref_trips), B.loop(B.REF, 1, 64, ref_trips), phase_a, phase_b, B.loop(B.REF, 1, 16, ref_trips)]
    b = core+B.loop('fadd32', 8, 64, warm)+B.clock_read(0, count)
    for k, x in enumerate(ph): b += x+B.clock_read(k+1, count)
    return b


def measure(R, phase_a, phase_b, units, groups_mode, ref_trips=1000, extra_env=None):
    """Cycles per unit of (phase_b - phase_a). groups_mode 'one' (32 lanes) or 'gpu' (4/core)."""
    count = 32 if groups_mode == 'one' else B.CORES*4*32
    body = program(phase_a, phase_b, count, ref_trips)
    for attempt in range(1, 9):
        fw, ts = R.run(body, count, 2+attempt, 32, clock=True, extra_env=extra_env)
        if groups_mode == 'one':
            t = ts[0]; d = [((t[k+1]-t[k]) & 0xffffffff)*B.TICK_NS for k in range(5)]
            npc = (d[1]-d[0])/(ref_trips*48)/2
            if abs(d[4]-d[0])/d[0] < 0.02:
                return {'cycles': (d[3]-d[2])/npc/units, 'mhz': 1e3/npc, 'attempts': attempt}
        else:
            from collections import defaultdict, Counter
            by = defaultdict(list)
            for t in ts: by[t[6]].append(t)
            span = lambda g, a, b: (max((x[b]-g[0][0]) & 0xffffffff for x in g)-min((x[a]-g[0][0]) & 0xffffffff for x in g))*B.TICK_NS
            vals = []
            nmode = Counter(len(g) for g in by.values()).most_common(1)[0][0]
            for g in by.values():
                if len(g) != nmode: continue
                npc = (span(g, 1, 2)-span(g, 0, 1))/(ref_trips*48)/2
                vals.append(((span(g, 3, 4)-span(g, 2, 3))/npc/units, 1e3/npc))
            if len(vals) >= 10:
                return {'cycles': statistics.median(v[0] for v in vals), 'mhz': statistics.median(v[1] for v in vals),
                        'groups_per_core': nmode, 'cores': len(vals), 'attempts': attempt}
    return None


_P = None


def predict(phase_a, phase_b, units, groups, P=None):
    global _P
    if P is None:
        if _P is None: _P = C.Params(); C.fit_branch(_P)     # cached per process (branch fit is slow)
        P = _P
    c = []
    for code in (phase_a, phase_b):
        ins = C.decode(code, P); pcs, _ = C.trace(code); c.append(C.simulate(ins, pcs, P, groups))
    return C.to_ref(c[1]-c[0])/units      # reference units, like the stored measurements


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('what', choices=['calibrate', 'validate'])
    ap.add_argument('--kernels', type=int, default=24)
    ap.add_argument('--length', type=int, default=24)
    ap.add_argument('--seed-base', type=int, default=1000)
    ap.add_argument('--name', default='validation.json')
    ap.add_argument('--memory', action='store_true', help='random kernels with memory instructions (mem_kernel)')
    ap.add_argument('--output', type=Path, default=Path('/scratch/apple-m2-playground/cost'))
    a = ap.parse_args()
    gpu_safety.require_budget(1)
    R = B.Runner(a.output/a.what)
    if a.what == 'calibrate':
        cal = {}
        T1, T2 = 200, 1000
        m = measure(R, counted(b'', T1), counted(b'', T2), T2-T1, 'one')
        cal['loop_trip_cycles_empty'] = m['cycles']; print('empty loop trip', m, flush=True)
        # get_sr: issue cost in a loop of 16 independent get_sr (sr80) vs empty
        gs16 = b''.join(get_sr(80, 5+k % 8) for k in range(16))
        m2 = measure(R, counted(b'', 400), counted(gs16, 400), 400*16, 'one')
        cal['get_sr_issue_cycles'] = m2['cycles']; print('get_sr issue', m2, flush=True)
        dep = b''.join(get_sr(80, 5)+e('iadd r6, r5, r6') for _ in range(8))
        nodep = b''.join(e('iadd r6, r7, r6') for _ in range(8))
        m3 = measure(R, counted(nodep, 400), counted(dep, 400), 400*8, 'one')
        cal['get_sr_extra_dependent_cycles'] = m3['cycles']; print('get_sr dependent extra', m3, flush=True)
        (a.output/'calibration.json').write_text(json.dumps(cal, indent=1)+'\n'); print(json.dumps(cal))
    else:
        out = []
        for k in range(a.kernels):
            if a.memory:
                lines, body = mem_kernel(a.seed_base+k, a.length)
                pa, pb = counted(body, 100, consts=MEM_CONSTS), counted(body, 500, consts=MEM_CONSTS)
            else:
                lines, body = kernel(a.seed_base+k, a.length)
                pa, pb = counted(body, 100), counted(body, 500)
            row = {'seed': a.seed_base+k, 'length': a.length, 'memory': a.memory,
                   'ops': lines if a.memory else [x for _, x in lines]}
            for gm, groups in (('one', 1), ('gpu', 4)):
                env = None
                if a.memory:
                    from run_drm_tg_layout import shared_word
                    env = {'AGX_SHARED_WORD': str(shared_word(2048))}
                m = measure(R, pa, pb, 400, gm, extra_env=env)
                p = predict(pa, pb, 400, groups)
                if gm == 'gpu' and m: p = p   # core cycles per trip with 4 groups each doing one trip
                row[gm] = {'measured': m and m['cycles'], 'predicted': p, 'mhz': m and m['mhz']}
            out.append(row)
            print(json.dumps({'seed': row['seed'], 'one': [round(row['one']['measured'] or -1, 1), round(row['one']['predicted'], 1)],
                              'gpu': [round(row['gpu']['measured'] or -1, 1), round(row['gpu']['predicted'], 1)]}), flush=True)
            (a.output/a.name).write_text(json.dumps(out)+'\n')
        for gm in ('one', 'gpu'):
            err = sorted(abs(r[gm]['predicted']-r[gm]['measured'])/r[gm]['measured'] for r in out if r[gm]['measured'])
            print(gm, 'median rel err', round(err[len(err)//2], 3), 'p90', round(err[int(.9*len(err))], 3), 'max', round(err[-1], 3))


if __name__ == '__main__':
    main()
