#!/usr/bin/env python3
"""Per-instruction benchmark suite on G14C (direct DRM): latency, per-SIMD issue rate,
and GPU-wide throughput expressed as lanes per cycle per core.

Method (all timing inside one dispatch, so DVFS clock changes cancel)
  Every SIMD group reads the CPU-written 24 MHz timer (device_barrier_1 + coherent load, see the
  2026-10-11 in-shader clock finding) around five phases: REF(16) REF(64) OP(16) OP(64) REF(16).
  REF is the dependent 32-bit `iadd r, r, 1` chain (2 cycles/step, 2026-10-10 loop finding); its slope is
  ns per cycle at that moment, the OP slope divided by it is cycles per step, and the repeated
  REF(16) bounds clock drift during the dispatch. Each case runs Mesa's counted-loop skeleton (push_exec / while_icmp / jmp_exec_any /
  pop_exec, as in agx_loop.loop_body) with a body of S steps of one instruction over C
  register chains. Two body lengths S1 < S2 at the same trip count K give a slope that
  removes dispatch and loop overhead:  ns_per_step = (T(S2) - T(S1)) / (K * (S2 - S1)).
  The reference is the dependent 32-bit iadd chain (2 cycles per step, see the
  2026-10-10 loop/ALU finding); its slope gives ns per cycle, i.e. the clock.
    latency      : 1 SIMD group, 1 chain           -> cycles per dependent step
    simd_issue   : 1 SIMD group, 8 chains          -> cycles per instruction per SIMD group
    gpu          : 38 cores x 16 groups, 8 chains  -> lanes per cycle per core
  "units" = lanes/cycle/core / 32: SIMD-group instructions the core completes per cycle.

Safety (no GPU hang possible from the body)
  * Only instructions whose base form already ran on hardware (bench_ops.py).
  * Every body instruction is decoded: exact length, not control flow, destination
    register >= r5 (r0l exec, r3 counter, r4 trip count are never written).
  * Each instruction runs once straight-line before it is looped; trip counts are
    sized so every dispatch stays far below the ~0.5 s firmware timeout; any fault
    stops the suite and is recorded in the per-boot budget (gpu_safety.py).
"""
import argparse, json, os, statistics, struct, subprocess, sys, time
from pathlib import Path
import agx_loop, gpu_safety, run_drm_compute as drm
import run_agx_timestamps as chains
from bench_ops import OPS

REPO = Path(__file__).resolve().parents[1]
mov = chains.mov_imm
e = agx_loop.encode
CORES = 38
CONTROL = {'if_icmp', 'if_fcmp', 'else_icmp', 'else_fcmp', 'while_icmp', 'while_fcmp', 'pop_exec', 'jmp_exec_any',
           'jmp_exec_none', 'stop', 'trap', 'call', 'ret', 'wait', 'threadgroup_barrier', 'memory_barrier'}
CONST = {'int': (1, 3, 5, 7), 'int64': (1, 0, 5, 7), 'bitfield': (0x5a5a1234, 3, 5, 7), 'select': (100, 7, 5, 7),
         'move': (1, 3, 5, 7), 'float': (0, 0x3f800000, 5, 7), 'half': (0, 0x3c00, 5, 7), 'convert': (1, 3, 5, 7),
         'sfu': (1, 3, 5, 7), 'simd': (1, 3, 5, 7)}
INIT = {'float': 0x3fc00000, 'half': 0x3e00, 'sfu': 0x3fc00000, 'convert': 0x3fc00000}


def chain_regs(cls, n):
    step = 2 if cls == 'int64' else 1
    regs = [5+step*k for k in range(n)]
    assert regs[-1]+step-1 < 24, 'chain registers overlap constants'
    return regs


def step_bytes(op, d):
    t, cls = OPS[op]
    t = t.replace('r20', 'r24').replace('r21', 'r25').replace('r22', 'r26')
    return e(t.format(d=d, d1=d+1))


def check_body(body, nsteps_bytes):
    """Decode every instruction: lengths as intended, no control flow, dest >= r5."""
    _, g = agx_loop._tools()
    p = 0
    for want in nsteps_bytes:
        n = g.opcode_to_number(body[p:p+16]); d = g.get_instruction_descriptor(n)
        if d is None or d.decode_size(n) != want: raise ValueError(f'length mismatch at {p}')
        n &= (1 << 8*want)-1
        if d.name in CONTROL: raise ValueError(f'control-flow instruction in body: {d.name}')
        f = dict(d.decode_fields(n))
        if 'D' not in f: raise ValueError(f'no destination field: {d.name}')
        if f['D'] < 10: raise ValueError(f'{d.name} writes below r5 (D={f["D"]})')
        p += want


def loop(op, nchains, steps, trips):
    t, cls = OPS[op]
    regs = chain_regs(cls, nchains)
    c = CONST[cls]
    head = mov(3, 0)+mov(4, trips)+b''.join(mov(24+k, v) for k, v in enumerate(c))
    head += b''.join(mov(r, INIT.get(cls, 0x01234567+r)) for r in regs)
    if cls == 'int64': head += b''.join(mov(r+1, 0) for r in regs)
    head += e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    parts = [step_bytes(op, regs[k % nchains]) for k in range(steps)]
    body = b''.join(parts)
    check_body(body, [len(x) for x in parts])
    lp = body+e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
    lp += e(f'jmp_exec_any pc-{len(lp)}', f'jmp_exec_any pc-{len(lp)}')
    return head+lp+e('pop_exec r0l, 1')+e('iadd r2, r5, 0')


def straight(op):
    t, cls = OPS[op]
    c = CONST[cls]
    b = b''.join(mov(24+k, v) for k, v in enumerate(c))+mov(5, INIT.get(cls, 7))+mov(6, 0)
    s = step_bytes(op, 5)
    check_body(s, [len(s)])
    return b+s+e('iadd r2, r5, 0')


from run_drm_cpu_clock import load as clk_load
from run_drm_opcode_describe import get_sr

REF = 'iadd_imm'        # dependent chain, 2 cycles/step, one register read (stays below the read-port limit)
TS_BASE = 64            # table word where timestamp regions start; CPU timer at word 1


def clock_read(k, count):
    """Timestamp k: invalidate, coherent load of the CPU timer, store it to table[TS_BASE + k*count + lane]."""
    return (bytes.fromhex('f59d')+mov(22, 1)+clk_load(28, 22, True)+e('wait 0')
            + mov(27, TS_BASE+k*count)+e('iadd r27, r1, r27')+agx_loop.store(28, 8, 27)+e('wait 0'))


def program(op, chains_op, s1, s2, trips, count, warm=20000):
    """WARM, then T0 REF(s1) T1 REF(s2) T2 OP(s1) T3 OP(s2) T4 REF(s1) T5, all in one dispatch.
    WARM (8 independent fadd32 chains, `warm` trips of 64) lets the clock settle under this load."""
    ph = [loop(REF, 1, s1, trips), loop(REF, 1, s2, trips), loop(op, chains_op, s1, trips),
          loop(op, chains_op, s2, trips), loop(REF, 1, s1, trips)]
    # region 6: SR20 core id of each lane (per-core aggregation in GPU mode)
    core = get_sr(20, 28)+mov(27, TS_BASE+6*count)+e('iadd r27, r1, r27')+agx_loop.store(28, 8, 27)+e('wait 0')
    b = core+(loop('fadd32', 8, 64, warm) if warm else b'')+clock_read(0, count)
    for k, x in enumerate(ph):
        b += x+clock_read(k+1, count)
    return b


class Runner:
    def __init__(self, out):
        self.out = out; out.mkdir(parents=True, exist_ok=True)
        self.exe = out/'drm_compute'
        subprocess.run(['cc', '-std=c11', '-O2', str(REPO/'tests/drm_compute.c'), '-o', str(self.exe)], check=True)
        self.n = 0

    def run(self, body, count, repeats=1, local_x=32, gprs=64, clock=False, extra_env=None):
        (self.out/'c.bin').write_bytes(drm.build_main(body)+bytes(16))
        env = {'PATH': '/usr/bin', 'AGX_LOCAL_X': str(local_x), **(extra_env or {})}
        argv = [str(self.exe), '/dev/dri/renderD128', str(self.out/'c.bin'), str(count), str(self.out/'rb'), str(repeats), str(gprs)]
        if clock:
            words = TS_BASE+7*count
            (self.out/'t.bin').write_bytes(bytes(4*words)); argv.append(str(self.out/'t.bin'))
            env.update(AGX_CPU_CLOCK='0', AGX_TABLE_OUT=str(self.out/'t.out'))
        try:
            r = subprocess.run(argv, capture_output=True, timeout=30, env=env)
        except subprocess.TimeoutExpired:
            gpu_safety.record('run_bench timeout'); raise RuntimeError('host timeout')
        smp = [json.loads(l) for l in r.stdout.decode().splitlines() if '"rep"' in l]
        if not (r.returncode == 0 and len(smp) == repeats and all(x['fw_ts'][0] and x['fw_ts'][1] for x in smp)):
            gpu_safety.record('run_bench fault'); raise RuntimeError('dispatch fault: '+r.stdout.decode()[-200:])
        if max(x['host_ns'] for x in smp) > 200_000_000: raise RuntimeError('dispatch too long; stopping')
        self.n += 1
        fw = [x['fw_ts'][1]-x['fw_ts'][0] for x in smp]
        if not clock: return fw, None
        t = struct.unpack(f'<{TS_BASE+7*count}I', (self.out/'t.out').read_bytes())
        ts = [[t[TS_BASE+k*count+i] for k in range(7)] for i in range(0, count, 32)]    # per SIMD group: T0..T5, core
        return fw, ts


MODES = {   # name: (op chains, lanes, local_x)
    'latency': (1, 32, 32),
    'simd_issue': (8, 32, 32),
    'gpu': (8, CORES*4*32, 32),     # analysed per core (SR20), see analyse_gpu
    'gpu8': (8, CORES*8*32, 32),    # 8 groups/core: tests whether more groups add throughput
}
TICK_NS = 1e9/24e6
# GPU DVFS levels (Asahi t602x-common.dtsi opp-table-gpu). Clocks inferred with the 2-unit iadd reference
# come out low by REAL_CYCLES_PER_UNIT; record the nearest level as a sanity check of each measurement.
DVFS_MHZ = (444, 612, 808, 968, 1110, 1236, 1338, 1398)
REAL_CYCLES_PER_UNIT = 1.049


def nearest_level(mhz):
    real = mhz*REAL_CYCLES_PER_UNIT
    lvl = min(DVFS_MHZ, key=lambda f: abs(f-real))
    return lvl, abs(real-lvl)/lvl      # CNTVCT is 24 MHz


def analyse(ts, trips, s1, s2, max_drift=0.02):
    """Per SIMD group: ns/cycle from the reference slope, cycles/step for the op; medians over the
    groups whose clock was stable (first and last REF(16) phases within max_drift)."""
    rows = []
    for t in ts:
        d = [((t[k+1]-t[k]) & 0xffffffff)*TICK_NS for k in range(5)]
        ref_slope = (d[1]-d[0])/(trips*(s2-s1))           # ns per reference step (2 cycles)
        npc = ref_slope/2
        op_slope = (d[3]-d[2])/(trips*(s2-s1))
        drift = (d[4]-d[0])/d[0]
        rows.append({'ns_per_cycle': npc, 'op_ns_per_step': op_slope, 'op_cycles_per_step': op_slope/npc, 'drift': drift})
    allg = len(rows)
    rows = [r for r in rows if abs(r['drift']) < max_drift]
    if not rows: return {'groups': allg, 'stable_groups': 0}
    med = lambda k: statistics.median(r[k] for r in rows)
    out = {'groups': allg, 'stable_groups': len(rows), 'ns_per_cycle': med('ns_per_cycle'), 'mhz': 1e3/med('ns_per_cycle'),
            'cycles_per_step': med('op_cycles_per_step'), 'ns_per_step': med('op_ns_per_step'),
            'cycles_spread': [min(r['op_cycles_per_step'] for r in rows), max(r['op_cycles_per_step'] for r in rows)],
            'max_abs_drift': max(abs(r['drift']) for r in rows)}
    return out


def analyse_gpu(ts, trips, s1, s2):
    """GPU mode, per core (SR20). Contention is per core and SIMD groups sharing a busy unit are
    not scheduled fairly, so a core is measured as a whole: each phase's core span runs from the
    first of its groups entering to the last leaving. Span slopes (64-step minus 16-step) remove
    loop overhead: REF gives ns per cycle (2 cycles per step), OP gives ns per step for all of the
    core's groups together, i.e. lane-ops per ns = groups * 32 / slope."""
    from collections import defaultdict, Counter
    by = defaultdict(list)
    for t in ts: by[t[6]].append(t)
    span = lambda g, a, b: max((x[b]-g[0][0]) & 0xffffffff for x in g)*TICK_NS - min((x[a]-g[0][0]) & 0xffffffff for x in g)*TICK_NS
    cores = []
    for c, g in by.items():
        npc = (span(g, 1, 2)-span(g, 0, 1))/(trips*(s2-s1))/2
        op = (span(g, 3, 4)-span(g, 2, 3))/(trips*(s2-s1))
        if npc <= 0 or op <= 0: continue
        align = (max((x[2]-g[0][0]) & 0xffffffff for x in g)-min((x[2]-g[0][0]) & 0xffffffff for x in g))*TICK_NS
        cores.append({'core': c, 'groups': len(g), 'lanes_per_cycle': len(g)*32/op*npc, 'mhz': 1e3/npc,
                      'core_cycles_per_step': op/npc, 'op16_start_spread_ns': align})
    if not cores: return {'cores_used': 0, 'cores_seen': len(by)}
    mode_n = Counter(c['groups'] for c in cores).most_common(1)[0][0]
    sel = [c for c in cores if c['groups'] == mode_n]
    med = lambda k: statistics.median(c[k] for c in sel)
    return {'cores_seen': len(by), 'cores_used': len(sel), 'groups_per_core': mode_n,
            'group_count_histogram': dict(Counter(len(g) for g in by.values())),
            'lanes_per_cycle_per_core': med('lanes_per_cycle'),
            'lanes_spread': [min(c['lanes_per_cycle'] for c in sel), max(c['lanes_per_cycle'] for c in sel)],
            'cycles_per_step': med('core_cycles_per_step')/mode_n,     # per group-step, comparable to other modes
            'mhz': med('mhz'), 'max_align_ns': max(c['op16_start_spread_ns'] for c in sel), 'stable_groups': mode_n*len(sel), 'groups': len(ts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ops', default=','.join(OPS))
    ap.add_argument('--modes', default='latency,simd_issue,gpu')
    ap.add_argument('--trips', type=int, default=1000)
    ap.add_argument('--warm', type=int, default=20000, help='warm-up trips (x64 fadd32) before the timed phases')
    ap.add_argument('--output', type=Path, default=Path('/scratch/apple-m2-playground/bench/run'))
    a = ap.parse_args()
    gpu_safety.require_budget(1)
    R = Runner(a.output)
    rec = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'modes': {k: MODES[k] for k in a.modes.split(',')},
           'trips': a.trips, 'warm': a.warm, 'steps': [16, 64], 'ops': {}}
    save = lambda: (a.output/'record.json').write_text(json.dumps(rec)+'\n')
    S1, S2 = 16, 64
    try:
        for op in a.ops.split(','):
            R.run(straight(op), 32)                               # stage 1: one straight-line execution
            res = {'class': OPS[op][1], 'template': OPS[op][0]}
            for mode in a.modes.split(','):
                ch, count, lx = MODES[mode]
                body = program(op, ch, S1, S2, a.trips, count, a.warm)
                accepted = False
                for attempt in range(1, 9):
                    # back-to-back submits warm the clock; the table holds the last one
                    fw, ts = R.run(body, count, 2+attempt, lx, clock=True)
                    if mode in ('gpu', 'gpu8'):
                        m = analyse_gpu(ts, a.trips, S1, S2)
                        # accept when two independent dispatches agree within 5% (a clock change during
                        # the OP phase would show as disagreement)
                        if m.get('cores_used', 0) >= 10:
                            prev = res.get('_prev_'+mode)
                            if prev and abs(prev-m['lanes_per_cycle_per_core']) <= 0.05*m['lanes_per_cycle_per_core']:
                                m['previous_dispatch_lanes'] = prev; accepted = True; break
                            res['_prev_'+mode] = m['lanes_per_cycle_per_core']
                    else:
                        m = analyse(ts, a.trips, S1, S2)
                        if m['stable_groups'] >= max(1, m['groups']//2): accepted = True; break
                m['fw_ns'] = fw; m['attempts'] = attempt; m['accepted'] = accepted
                if 'mhz' in m: m['dvfs_level_mhz'], m['dvfs_mismatch'] = nearest_level(m['mhz'])
                if not accepted or 'cycles_per_step' not in m:
                    m.pop('cycles_per_step', None)
                    res[mode] = m; print(json.dumps({'op': op, 'mode': mode, 'unstable_clock': True})); continue
                res[mode] = m
            if 'gpu' in res and 'cycles_per_step' in res['gpu']:
                g = res['gpu']; ch, count, lx = MODES['gpu']
                # every SIMD group issues one op per chain-step for 32 lanes: lanes/cycle/core
                res['lanes_per_cycle_per_core'] = g['lanes_per_cycle_per_core']
                res['units_per_core'] = g['lanes_per_cycle_per_core']/32
            for k in [k for k in res if k.startswith('_prev_')]: res.pop(k)
            rec['ops'][op] = res; save()
            summary = {'op': op}
            for mode in ('latency', 'simd_issue', 'gpu'):
                if mode in res and 'cycles_per_step' in res[mode]: summary[mode] = [round(res[mode]['cycles_per_step'], 2), round(res[mode]['mhz']), res[mode]['stable_groups'], res[mode]['attempts']]
            if 'gpu8' in res and 'lanes_per_cycle_per_core' in res['gpu8']: summary['gpu8 lanes/clk/core'] = [round(res['gpu8']['lanes_per_cycle_per_core'], 1), round(res['gpu8']['mhz'])]
            if 'gpu' in res and 'groups_per_core' in res['gpu']: summary['gpu_cores'] = [res['gpu']['cores_used'], res['gpu']['groups_per_core'], round(res['gpu']['max_align_ns'])]
            if 'units_per_core' in res: summary['lanes/clk/core'] = round(res['lanes_per_cycle_per_core'], 1)
            print(json.dumps(summary), flush=True)
        rec['complete'] = True
    except Exception as ex:
        rec['error'] = str(ex); print('STOPPED:', ex)
    finally:
        rec['dispatches'] = R.n; save()


if __name__ == '__main__':
    main()
