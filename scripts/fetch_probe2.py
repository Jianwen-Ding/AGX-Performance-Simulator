#!/usr/bin/env python3
"""Instruction fetch / loop buffer probe (direct DRM, known-safe encodings).
Bodies of N independent instructions (8 rotating registers r5..r12, so issue-bound at 1/cycle when
fetch keeps up) for N spanning body sizes 128 B .. 2 KiB, 8-byte `iadd rX, rX, 1` vs 4-byte
`fadd32 rX, rX, 0.125`. Each dispatch times 4 sizes plus a repeat of the first (drift check);
consecutive dispatches overlap by one size so per-instruction slopes are always taken within one
dispatch. Reported: ns per instruction between neighbouring sizes and the body bytes."""
import json, sys
from pathlib import Path
import agx_loop, run_bench as B, cost_hw as H, gpu_safety
from fetch_probe import phases_program
e = agx_loop.encode
T = 200

def body(instr, n):
    parts = [e(instr.format(r=5+k % 8)) for k in range(n)]
    B.check_body(b''.join(parts), [len(p) for p in parts])
    return H.counted(b''.join(parts), T)

if __name__ == '__main__':
    gpu_safety.require_budget(1)
    R = B.Runner(Path('/scratch/apple-m2-playground/cost/fetch2'))
    out = {}
    for instr, L in (('iadd r{r}, r{r}, 1', 8), ('fadd32 r{r}, r{r}, 0.125', 4)):
        target_bytes = [128, 256, 384, 448, 480, 512, 544, 576, 640, 768, 1024, 1536, 2048]
        ns = [b//L for b in target_bytes]
        rows = []
        for i in range(0, len(ns)-1, 3):
            grp = ns[i:i+4]
            codes = [body(instr, n) for n in grp]+[body(instr, grp[0])]
            fw, ts = R.run(phases_program(codes), 32, 3, 32, clock=True)
            t = ts[0]; d = [((t[k+1]-t[k]) & 0xffffffff)*B.TICK_NS/T for k in range(len(codes))]
            drift = abs(d[-1]-d[0])/d[0]
            for a, b, da, db in zip(grp, grp[1:], d, d[1:]):
                r = {'from_bytes': a*L, 'to_bytes': b*L, 'ns_per_instr': (db-da)/(b-a), 'drift': drift}
                rows.append(r)
                print(f"{instr.split()[0]:7s} {L}B  body {a*L:5d}->{b*L:5d} B  ns/instr {r['ns_per_instr']:6.3f}  drift {drift:.3f}", flush=True)
        out[instr] = rows
    Path('/scratch/apple-m2-playground/cost/fetch2.json').write_text(json.dumps(out)+'\n')
