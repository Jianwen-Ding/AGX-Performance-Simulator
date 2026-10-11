#!/usr/bin/env python3
"""Loop-body size probe (direct DRM, known-safe encodings). One SIMD group times, inside one
dispatch, counted loops whose body is S dependent steps of one instruction, for several S, with
the CPU-written timer. ns/trip vs S shows where per-step cost changes with body size (bytes).
Drift check: the first phase is repeated at the end."""
import json, sys
from pathlib import Path
import agx_loop, run_bench as B, cost_hw as H, gpu_safety
e = agx_loop.encode

def phases_program(codes, count=32, warm=20000):
    b = B.loop('fadd32', 8, 64, warm)+B.clock_read(0, count)
    for k, c in enumerate(codes): b += c+B.clock_read(k+1, count)
    return b

def run(R, codes):
    n = len(codes)
    body = phases_program(codes)
    old = B.TS_BASE
    # Runner reads 7 regions; phases beyond 6 need more regions: run in chunks of 5
    fw, ts = R.run(body, 32, 3, 32, clock=True)
    t = ts[0]
    return [((t[k+1]-t[k]) & 0xffffffff)*B.TICK_NS for k in range(n)]

def chain(instr, steps, trips):
    step = e(instr)
    body = step*steps
    B.check_body(body, [len(step)]*steps)
    return H.counted(body, trips)

if __name__ == '__main__':
    gpu_safety.require_budget(1)
    R = B.Runner(Path('/scratch/apple-m2-playground/cost/fetch'))
    T = 300
    out = {}
    for instr in sys.argv[1:] or ['iadd r6, r6, 1', 'fadd32 r6, r6, 0.125']:
        L = len(e(instr))
        sizes = [16, 32, 48, 56, 60, 62, 64, 66, 68, 72, 80, 96, 128] if L == 8 else [32, 64, 96, 112, 120, 124, 128, 132, 136, 144, 160, 192, 256]
        rows = []
        for i in range(0, len(sizes), 4):     # 4 sizes + repeated first per dispatch (5 timed phases)
            grp = sizes[i:i+4]
            codes = [chain(instr, s, T) for s in grp]+[chain(instr, grp[0], T)]
            d = run(R, codes)
            drift = abs(d[-1]-d[0])/d[0]
            for s, x in zip(grp, d): rows.append({'steps': s, 'body_bytes': s*L+18, 'ns_per_trip': x/T, 'drift': drift})
        out[instr] = rows
        for a, b in zip(rows, rows[1:]):
            print(f"{instr:22s} S {a['steps']:3d}->{b['steps']:3d} body {a['body_bytes']:4d}->{b['body_bytes']:4d} B  ns/step {(b['ns_per_trip']-a['ns_per_trip'])/(b['steps']-a['steps']):6.3f}  drift {b['drift']:.3f}", flush=True)
    Path('/scratch/apple-m2-playground/cost/fetch.json').write_text(json.dumps(out)+'\n')
