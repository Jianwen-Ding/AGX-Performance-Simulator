#!/usr/bin/env python3
"""Throughput beyond 4 SIMD groups per core (direct DRM, known-safe encodings only).

run_bench.py measures at most 4 SIMD groups per core: with more, every group also runs the dependent
`iadd` reference chain, the reference needs the core's whole issue rate and the inferred clock is wrong
(notes/failed-approaches.md, "8 SIMD groups per core breaks the in-dispatch reference"). Here the
reference runs in ONE SIMD group per core while the others idle at a barrier:

* 38 workgroups of G SIMD groups (32*G lanes); each lands on its own core (checked per workgroup with
  SR20: all of its groups must report one core, and no core may hold two workgroups).
* Every group runs the same program, but the reference loops take their trip count from a register
  that is `ref_trips` in the workgroup's first SIMD group (SR53 = 0) and 0 elsewhere (a do-while loop: 1 trip).
* Phases (BAR = threadgroup_barrier 0x68, Tk = CPU-timer read stored per group):
      core-id, WARM, BAR, T0 REF(s1) T1 REF(s2) T2, BAR, OP(s1), BAR, T3, BAR, OP(s2), BAR, T4 REF(s1) T5
  Group 0's T2..T4 bracket OP(s1) and OP(s2) of all G groups between the same barriers, so the slope
  (T4-T3)-(T3-T2) over trips*(s2-s1) is the core's time for one step of every group. REF's slope
  gives ns per cycle at that moment (2 reference units per step); REF(s1) at both ends bounds drift.
* Results are in reference units (dependent 8-byte iadd step = 2), like run_bench; real = x1.049.

Usage: scale_probes.py ops [--ops a,b] [--groups 1,2,4,8,16,32] [--chains 1,8]
"""
import argparse, json, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
import agx_loop, gpu_safety, run_bench as B
from bench_ops import OPS

e, mov = agx_loop.encode, B.mov
BAR = bytes.fromhex('6800')          # threadgroup_barrier (whole workgroup; barrier finding)
CORES = 38
R_LOCAL, R_REFTRIPS = 30, 31         # untouched by loop()/clock_read()/core-id (they use r3..r27, r28)


def group_setup(G, ref_trips, warp=None):
    """r30 = SIMD-group index in the workgroup (SR53; `warp` substitutes a constant for the emulator gate),
    r31 = ref trips in group 0, else 0."""
    w = B.get_sr(53, R_LOCAL) if warp is None else mov(R_LOCAL, warp)
    return (w+mov(R_REFTRIPS, ref_trips)
            + e(f'icmpsel ult, r{R_REFTRIPS}, r{R_LOCAL}, 1, r{R_REFTRIPS}, 0'))


R_OPTRIPS = 29


def loop_reg(op, nchains, steps, reg=R_REFTRIPS):
    """run_bench.loop with the trip count taken from a register (per-group)."""
    b = B.loop(op, nchains, steps, 1)
    m = mov(4, 1)
    i = b.index(m)
    return b[:i]+e(f'iadd r4, r{reg}, 0')+b[i+len(m):]


def op_trips_setup(trips, idle):
    """r29 = OP trips, or 1 in the SIMD groups listed in `idle` (by SR53 index)."""
    b = mov(R_OPTRIPS, trips)
    for s in idle:
        b += e(f'icmpsel seq, r{R_OPTRIPS}, r{R_LOCAL}, {s}, 1, r{R_OPTRIPS}')
    return b


def skew_loop(k):
    """k*SR53 + 1 trips of 8 dependent iadds on r28 (desynchronises the groups before a timed phase)."""
    if not k: return b''
    head = mov(28, k)+e(f'imadd r28, r{R_LOCAL}, r28, 0')+e('iadd r28, r28, 1')+mov(3, 0)+e('iadd r4, r28, 0')
    head += e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    body = b''.join(e('iadd r27, r27, 1') for _ in range(8))+e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
    return head+body+e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}')+e('pop_exec r0l, 1')


def program(op, chains, G, s1, s2, trips, ref_trips, count, warm, idle=(), phase=None, skew=0):
    """`phase(s)`: optional builder of the timed loop with s steps (trip count from r29); default the bench
    loop of `op` with `chains` chains."""
    phase = phase or (lambda s: loop_reg(op, chains, s, R_OPTRIPS))
    # region 6: (SR20 core << 8) | SR17 hardware SIMD-group slot
    core = (B.get_sr(20, 28)+B.get_sr(17, 29)+mov(27, 256)+e('imadd r28, r28, r27, r29')
            + mov(27, B.TS_BASE+6*count)+e('iadd r27, r1, r27')+agx_loop.store(28, 8, 27)+e('wait 0'))
    ck = lambda k: B.clock_read(k, count)
    b = group_setup(G, ref_trips)+core+op_trips_setup(trips, idle)+(B.loop('fadd32', 8, 64, warm) if warm else b'')+BAR
    b += ck(0)+loop_reg(B.REF, 1, s1)+ck(1)+loop_reg(B.REF, 1, s2)+ck(2)+BAR
    b += skew_loop(skew)+phase(s1)+BAR+ck(3)+BAR+skew_loop(skew)+phase(s2)+BAR+ck(4)
    b += loop_reg(B.REF, 1, s1)+ck(5)
    return b


def gate():
    """Emulate the per-group reference loop for a first group (trips) and another group (1 trip)."""
    for warp, want in ((0, 7), (3, 1)):
        body = group_setup(4, 7, warp)+loop_reg(B.REF, 1, 2)+e('iadd r2, r3, 0')
        r = agx_loop.emulate(body, 32, lane_ids=True, num_registers=32)
        if r['r2'] != [want]*32 or r['counts'].get('jmp_exec_any') != want:
            raise RuntimeError(f'gate failed: want {want} trips, got {r["r2"][:2]} / {r["counts"].get("jmp_exec_any")}')


def analyse(ts, G, trips, ref_trips, s1, s2, max_drift=0.03):
    """Per workgroup: rows of its first group's timestamps; SR20 of all its groups must agree."""
    wgs = defaultdict(list)
    ts = [t[:6]+[t[6] >> 8, t[6] & 255] for t in ts]
    for i, t in enumerate(ts): wgs[i//G].append(t)
    per_core = Counter()
    rows, bad = [], 0
    for w, grp in wgs.items():
        cores = {t[6] for t in grp}
        if len(cores) != 1: bad += 1; continue
        per_core[cores.pop()] += 1
    for w, grp in wgs.items():
        c = grp[0][6]
        if len({t[6] for t in grp}) != 1 or per_core[c] != 1: continue
        t = grp[0]
        d = [((t[k+1]-t[k]) & 0xffffffff)*B.TICK_NS for k in range(5)]
        npc = (d[1]-d[0])/(ref_trips*(s2-s1))/2
        op = (d[3]-d[2])/(trips*(s2-s1))
        if min(d) <= 0: continue
        drift = (d[4]-d[0])/d[0]
        if npc <= 0 or op <= 0 or abs(drift) > max_drift: continue
        rows.append({'core': c, 'mhz': 1e3/npc, 'core_cycles_per_step': op/npc, 'slots': [x[7] for x in grp]})
    out = {'workgroups': len(wgs), 'split_workgroups': bad, 'shared_cores': sum(1 for v in per_core.values() if v > 1),
           'stable_cores': len(rows)}
    if rows:
        cps = statistics.median(r['core_cycles_per_step'] for r in rows)
        out.update({'core_cycles_per_step': cps, 'instr_per_cycle': G/cps, 'lanes_per_cycle': 32*G/cps,
                    'mhz': statistics.median(r['mhz'] for r in rows),
                    'slots': [r['slots'] for r in rows][:6],
                    'spread': [min(r['core_cycles_per_step'] for r in rows), max(r['core_cycles_per_step'] for r in rows)]})
    return out


def estimate_trips(op, chains, G, s):
    """Trips so that one OP(s2) phase lasts ~1-2 ms at ~1 GHz (bench table: issue / core rate)."""
    return 2000


def groups_spec(text):
    """'4,8,5/1+2' -> [(4, ()), (8, ()), (5, (1, 2))]: G, then the SIMD groups (SR53) given no OP work."""
    out = []
    for x in text.split(','):
        g, _, idle = x.partition('/')
        out.append((int(g), tuple(int(v) for v in idle.split('+')) if idle else ()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('what', choices=['ops', 'gate'])
    ap.add_argument('--ops', default='iadd_imm,mov_imm,fadd32_imm,fadd32,fmadd32,fmadd32_imm,imadd,bfi,convert_f2i,rcp,'
                    'log2,rsqrt,floor,simd_shuffle,quad_shuffle,simd_iadd,simd_prefix_iadd,quad_iadd,icmp_ballot,'
                    'fadd16,fmadd16,iadd64,icmpsel')
    ap.add_argument('--groups', default='1,2,4,8,12,16,24,32')
    ap.add_argument('--chains', default='1,8')
    ap.add_argument('--trips', type=int, default=0, help='OP trips (default 2000)')
    ap.add_argument('--ref-trips', type=int, default=2000)
    ap.add_argument('--warm', type=int, default=4000)
    ap.add_argument('--steps', default='32,128', help='body lengths s1,s2')
    ap.add_argument('--skew', type=int, default=0, help='before each timed phase, group g runs g*SKEW+1 trips of 8 iadds')
    ap.add_argument('--name', default='scale.json')
    ap.add_argument('--output', type=Path, default=Path('/scratch/apple-m2-playground/sim/scale'))
    a = ap.parse_args()
    gate()
    if a.what == 'gate': print('gate ok'); return
    gpu_safety.require_budget(1)
    R = B.Runner(a.output)
    S1, S2 = map(int, a.steps.split(','))
    out = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'steps': [S1, S2], 'ref_trips': a.ref_trips,
           'units': 'reference (x1.049 = real cycles)', 'rows': []}
    path = a.output/a.name
    for op in a.ops.split(','):
        R.run(B.straight(op), 32)
        for ch in map(int, a.chains.split(',')):
            for G, idle in groups_spec(a.groups):
                trips = a.trips or estimate_trips(op, ch, G, S2)
                count = CORES*32*G
                body = program(op, ch, G, S1, S2, trips, a.ref_trips, count, a.warm, idle, skew=a.skew)
                best = None
                for attempt in range(1, 6):
                    fw, ts = R.run(body, count, 2+attempt, 32*G, gprs=64, clock=True)
                    m0 = fw
                    m = analyse(ts, G, trips, a.ref_trips, S1, S2)
                    if m['stable_cores'] >= 10: best = m; break
                    best = best or m
                row = {'op': op, 'chains': ch, 'groups': G, 'idle': idle, 'skew': a.skew, 'trips': trips, 'attempts': attempt, **best}
                out['rows'].append(row)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'spread'}), flush=True)
                path.write_text(json.dumps(out, indent=0)+'\n')


if __name__ == '__main__':
    main()


def barrier_loop(steps, code=BAR, filler=b''):
    """Counted loop (trips in r29) of `steps` barriers (each optionally preceded by `filler`): every SIMD group
    of the workgroup executes the same number, so it cannot deadlock."""
    head = mov(3, 0)+e(f'iadd r4, r{R_OPTRIPS}, 0')+e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    body = (filler+code)*steps+e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
    return head+body+e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}')+e('pop_exec r0l, 1')


def barriers(groups, trips=200, ref_trips=500, s=(8, 32), out=Path('/scratch/apple-m2-playground/sim/barrier'), fills=(0, 4),
             codes=None, name='barrier.json', fill_kind='indep'):
    """Core cycles per barrier (all groups) for 0x68 and 0x78, alone and after 4 independent iadds."""
    R = B.Runner(out)
    rows = []
    for bname, code in (codes or (('0x68', BAR), ('0x78', bytes.fromhex('7809')))):
        for fill in fills:
            if fill_kind == 'indep':
                filler = b''.join(e(f'iadd r{5+k % 16}, r{5+k % 16}, 1') for k in range(fill))
            else:   # one dependent chain
                filler = b''.join(e('iadd r5, r5, 1') for k in range(fill))
            for G in groups:
                if bname == '0x78' and G > 4: continue      # 0x78 only covers aligned blocks of 4 groups
                count = CORES*32*G
                body = program(None, 0, G, s[0], s[1], trips, ref_trips, count, 2000, phase=lambda n: barrier_loop(n, code, filler))
                for att in range(1, 5):
                    fw, ts = R.run(body, count, 2+att, 32*G, gprs=64, clock=True)
                    m = analyse(ts, G, trips, ref_trips, s[0], s[1])
                    if m['stable_cores'] >= 10: break
                row = {'barrier': bname, 'filler_iadds': fill, 'fill_kind': fill_kind, 'groups': G, **{k: v for k, v in m.items() if k != 'slots'}}
                rows.append(row)
                print(json.dumps({'barrier': bname, 'fill': fill, 'kind': fill_kind, 'G': G, 'cycles_per_barrier': round(m.get('core_cycles_per_step', 0), 2),
                                  'stable': m['stable_cores']}), flush=True)
    (out/name).write_text(json.dumps({'steps': list(s), 'rows': rows})+'\n')
