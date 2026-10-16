#!/usr/bin/env python3
"""PTX non-.ftz cvt.rpi / cvt.rmi (and ceil / floor .f32) fix-up on the M2 (safe-01 part 3).

AGX floor / ceil read fp32 denormals as signed zero, so ceil(+denormal) = +0 and floor(-denormal) = -0, where IEEE
(PTX without .ftz) gives +1.0 / -1.0. The fix-up uses only recorded-safe register forms (iadd r,r,r; icmpsel ult
r,r,r,r,r; mov_imm):
  ceil:  r14 = ceil r6;  r20 = r6 + 0xffffffff;  r14 = (r20 <u 0x007fffff) ? 1.0 : r14       (r6 in [1, 0x7fffff])
  floor: r14 = floor r6; r20 = r6 + 0x7fffffff;  r14 = (r20 <u 0x007fffff) ? -1.0 : r14      (r6 in [0x80000001, 0x807fffff])
Inputs: the float-edge set, every 2^-k boundary denormal, 2 x 16,384 random denormals, and edge-dense normal binades.
Each kernel runs in agx_run first (must equal the IEEE oracle); then on the GPU (twice, must agree) unless --sim-only.
Results: results/2026-10-16-float-edges/rounding_fixup.json.gz. Usage: verify_rounding_fixup.py [--sim-only | --check]
"""
import argparse, gzip, json, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import coverage as COV
COV.install()
import gpu_safety, run_bench as B, record_suite as RS, run_drm_compute as drm
import record_float_edges as FE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/2026-10-16-float-edges/rounding_fixup.json.gz'
N = FE.N


def body(op):
    b = RS.e(f'{op} r14, r6')
    b += B.mov(21, 0xffffffff if op == 'ceil' else 0x7fffffff)+RS.e('iadd r20, r6, r21')
    b += B.mov(21, 0x007fffff)+B.mov(22, 0x3f800000 if op == 'ceil' else 0xbf800000)
    b += RS.e('icmpsel ult, r14, r20, r21, r22, r14')
    return RS.alu_body(b, N)


def inputs():
    rng = np.random.default_rng(7)
    den = np.concatenate([1 << np.arange(23, dtype=np.uint32), (1 << np.arange(23, dtype=np.uint32))-1,
                          rng.integers(1, 1 << 23, 16384, dtype=np.uint32), np.array([1, 0x7fffff], np.uint32)])
    v = [FE.edge_set(), den, den | np.uint32(0x80000000)]
    import record_convert as RC
    for k in (-3, -1, 0, 1, 22, 23):
        ms = RC.mant_set(k, rng, [0, 1])[:2048]
        for sg in (0, 1):
            v.append((np.uint32((k+127) << 23) | ms) | np.uint32(sg << 31))
    x = np.unique(np.concatenate(v))
    return x[x != 0]  # keep a 2^16-lane chunk; +0 is in the edge set as its own case below
    

def oracle(op, x):
    f = x.view(np.float32).astype(np.float64)
    r = (np.ceil if op == 'ceil' else np.floor)(f).astype(np.float32).view(np.uint32)
    return np.where(np.isnan(f), np.uint32(0x7fc00000), r)


def table(x):
    tab = np.zeros((N, 8), np.uint32); tab[:len(x), 0] = x
    return tab.ravel().tolist()+[0]*(2*N)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sim-only', action='store_true')
    ap.add_argument('--check', action='store_true', help='offline: replay the recorded words through agx_run')
    a = ap.parse_args()
    x = np.concatenate([inputs(), np.array([0, 0x80000000], np.uint32)])
    assert len(x) <= N
    rec = json.load(gzip.open(OUT, 'rt')) if a.check else {'inputs': x.tolist(), 'ops': {}}
    if a.check:
        x = np.array(rec['inputs'], np.uint32)
    fails = 0
    R = None
    for op in ('ceil', 'floor'):
        b = body(op); RS.gate(b); code = drm.build_main(b)
        _, tab = COV.simulate(code, N, 256, 64, 0, table(x))
        sim = np.array(tab[8*N:8*N+2*len(x):2], np.uint32)
        want = oracle(op, x)
        ok_sim = np.array_equal(sim, want)
        line = f'{op}: simulator == IEEE (non-ftz) oracle on {len(x)} inputs: {ok_sim}'
        if a.check:
            hw = np.array(rec['ops'][op]['hw'], np.uint32)
            ok = ok_sim and np.array_equal(hw, want) and rec['ops'][op]['code'] == code.hex()
            line += f'; recorded hardware == oracle: {np.array_equal(hw, want)}'
        elif not a.sim_only:
            if not ok_sim:
                sys.exit(f'{op}: simulator differs from the oracle; refusing to submit')
            gpu_safety.require_budget(1)
            R = R or B.Runner(Path('/scratch/apple-m2-playground/sim/rounding_fixup'))
            res = [RS.run(R, code, N, 256, table(x))[1] for _ in range(2)]
            if res[0] != res[1]:
                sys.exit(f'{op}: runs disagree')
            hw = np.array(res[0][8*N:8*N+2*len(x):2], np.uint32)
            ok = np.array_equal(hw, want)
            rec['ops'][op] = {'code': code.hex(), 'hw': hw.tolist()}
            line += f'; hardware == oracle: {ok}'
        else:
            ok = ok_sim
        fails += not ok
        print(line, flush=True)
    if not a.check and not a.sim_only:
        json.dump(rec, gzip.open(OUT, 'wt'))
    print(f'{fails} failures')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
