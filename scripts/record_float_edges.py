#!/usr/bin/env python3
"""Subnormal / boundary behaviour of the safe fp32 forms PTX's default (non-.ftz) instructions need (safe-01 part 3).

Every kernel is record_suite.alu_body(<instruction>): the exact harness the coverage suite recorded these forms with
(r6..r13 = table[8*gid ..]; r14 = r15 = 0xdeadbeef; instruction; r14, r15 stored); only the input data is new.
Sweeps (results: /scratch/agxsim-data/float_edges/<sweep>.npz with the inputs and hardware words):
  cmp_<cc>    fcmpsel <cc>, r14, r6, r7, r8, r9 over every pair of the edge set E (r8 / r9 = 0x11111111 / 0x22222222)
  round_<op>  floor / ceil / trunc / rint r14, r6 over E and edge-dense binades 2^-3 .. 2^23, both signs
  f32_to_f16  fadd32 r14l, r6, r7 (r7 = -0.0: Mesa's f2f16) over every fp32 in binades 2^-26 .. 2^-13 and
              2^14 .. 2^16, both signs, plus E
  f16_to_f32  fadd32 r14, r6l, r7 (r7 = -0.0) over all 65,536 fp16
Each 2^16-lane dispatch runs twice and must agree. Usage: record_float_edges.py [--sweeps REGEX]
"""
import argparse, re, struct, sys
from pathlib import Path
import numpy as np
import coverage as COV
COV.install()
import gpu_safety, run_bench as B, record_suite as RS, run_drm_compute as drm
import record_convert as RC

OUT = Path('/scratch/agxsim-data/float_edges')
N = 1 << 16
f2b = lambda v: struct.unpack('<I', struct.pack('<f', v))[0]
NEG0 = 0x80000000


def edge_set():
    v = [0, 1, 2, 0x400000, 0x7ffffe, 0x7fffff, 0x800000, 0x800001, 0xffffff, 0x1000000, f2b(2.0**-125),
         f2b(1.0), f2b(1.0)+1, f2b(1.0)-1, f2b(1.5), f2b(0.5), f2b(2.5), f2b(-2.5), f2b(65504.0), f2b(65520.0), f2b(65519.0),
         f2b(2.0**-14), f2b(2.0**-24), f2b(2.0**-25), f2b(1.5*2.0**-25), f2b(2.0**-15)+1, f2b(8388607.5), f2b(8388608.0),
         0x7f7fffff, 0x7f800000, 0x7fc00000, 0x7fa00000, 0x7f800001]
    v = sorted(set(v))
    return np.array(v+[x | 0x80000000 for x in v], np.uint32)


def sweeps():
    E = edge_set()
    rng = np.random.default_rng(31)
    out = {}
    a, b = np.meshgrid(E, E, indexing='ij')
    pairs = np.stack([a.ravel(), b.ravel()], 1)
    for cc in ('eq', 'lt', 'gt', 'lte', 'gte', 'ltn', 'gtn'):
        out[f'cmp_{cc}'] = (f'fcmpsel {cc}, r14, r6, r7, r8, r9', pairs, {8: 0x11111111, 9: 0x22222222})
    rparts = [E]
    for k in range(-3, 24):
        ms = RC.mant_set(k, rng, [0, 1])
        for sg in (0, 1):
            rparts.append((np.uint32((k+127) << 23) | ms) | np.uint32(sg << 31))
    rin = np.concatenate(rparts)[:, None]
    for op in ('floor', 'ceil', 'trunc', 'rint'):
        out[f'round_{op}'] = (f'{op} r14, r6', rin, {})
    mant = np.arange(1 << 23, dtype=np.uint32)
    cv = [E]+[(np.uint32((k+127) << 23) | mant) | np.uint32(sg << 31) for k in list(range(-26, -12))+[14, 15, 16] for sg in (0, 1)]
    out['f32_to_f16'] = ('fadd32 r14l, r6, r7', np.concatenate(cv)[:, None], {7: NEG0})
    out['f16_to_f32'] = ('fadd32 r14, r6l, r7', np.arange(1 << 16, dtype=np.uint32)[:, None], {7: NEG0})
    return out


def table_for(xs, s0, consts):
    part = xs[s0:s0+N]
    tab = np.zeros((N, 8), np.uint32)
    tab[:len(part), :part.shape[1]] = part
    for reg, val in consts.items():
        tab[:, reg-6] = val
    return tab.ravel().tolist()+[0]*(2*N), len(part)


def simulate_chunk(code, xs, s0, consts):
    """agx_run (simulator) result words (r14, r15 per input) of one 2^16-lane chunk"""
    table, n = table_for(xs, s0, consts)
    _, tab = COV.simulate(code, N, 256, 64, 0, table)
    return np.array(tab[8*N:10*N], np.uint32).reshape(N, 2)[:n]


def sim_first_chunk(code, xs, consts):
    simulate_chunk(code, xs, 0, consts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweeps', default='.')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    S = {k: v for k, v in sweeps().items() if re.search(a.sweeps, k)}
    if a.list:
        for k, (t, x, c) in S.items():
            print(k, t, len(x))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    R = B.Runner(Path('/scratch/apple-m2-playground/sim/float_edges'))
    for k, (text, xs, consts) in S.items():
        path = OUT/f'{k}.npz'
        if path.exists():
            continue
        body = RS.alu_body(RS.e(text), N)
        RS.gate(body)
        code = drm.build_main(body)          # the whole program: prologue + body + stop (a bare body hung the GPU)
        gpu_safety.require_budget(1)
        sim_first_chunk(code, xs, consts)                # pre-GPU gate: the exact kernel completes in agx_run
        hw = np.zeros((len(xs), 2), np.uint32)
        for s0 in range(0, len(xs), N):
            table, n = table_for(xs, s0, consts)
            res = [RS.run(R, code, N, 256, table)[1] for _ in range(2)]
            if res[0] != res[1]:
                raise RuntimeError(f'{k} chunk {s0}: runs disagree')
            hw[s0:s0+n] = np.array(res[0][8*N:10*N], np.uint32).reshape(N, 2)[:n]
        np.savez_compressed(path, x=xs, hw=hw)
        print(f'{k}: {len(xs)} inputs', flush=True)


if __name__ == '__main__':
    sys.exit(main())
