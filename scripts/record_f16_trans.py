#!/usr/bin/env python3
"""Record every fp16 input of the fp16 transcendental forms the coverage suite recorded as safe (direct DRM).

Forms: `<op> r14l, r6h` (fp16 -> fp16) and `<op> r14, r6l` (fp16 source -> fp32 result), for rcp, rsqrt,
rsqrt_special, log2, exp2, sin_pt_1 and sin_pt_2. One straight-line kernel per form (load, mov r14 = 0, op, store);
the 65,536 inputs run in one 2^18-lane chunk, twice, and the runs must agree. Output:
/scratch/agxsim-data/transcendental/f16/<op>_<form>.npy (uint32 result words, index = fp16 input bits).
Usage: record_f16_trans.py [--ops rcp,...]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm
from record_suite import run, gate
import record_transcendentals as T

e, mov = agx_loop.encode, B.mov
OPS = ['rcp', 'rsqrt', 'rsqrt_special', 'log2', 'exp2', 'sin_pt_1', 'sin_pt_2']
FORMS = {'h2h': ('r14l', 'r6h', 16), 'h2f': ('r14', 'r6l', 0)}   # destination, source, input shift into r6


def body(op, form):
    d, s, _ = FORMS[form]
    b = e('device_load 0, i32, x, r6, u8_u9, r1, unsigned')+e('wait 0')+mov(14, 0)
    b += e(f'{op} {d}, {s}')
    b += mov(27, T.CHUNK)+e('iadd r26, r1, r27')+agx_loop.store(14, 8, 26)+e('wait 0')+e('iadd r2, r14, 0')
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ops', default=','.join(OPS))
    ap.add_argument('--out', type=Path, default=Path('/scratch/agxsim-data/transcendental/f16'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    gpu_safety.require_budget(1)
    R = B.Runner(Path('/scratch/apple-m2-playground/sim/f16trans'))
    C = T.CHUNK
    for op in a.ops.split(','):
        for form, (_, _, sh) in FORMS.items():
            path = a.out/f'{op}_{form}.npy'
            if path.exists():
                print(f'{path} exists, skipped'); continue
            gate(body(op, form))
            code = drm.build_main(body(op, form))
            xs = [x << sh for x in range(1 << 16)]
            table = xs+[0]*(C-len(xs))+[0]*C
            r = [run(R, code, C, T.WG, table)[1] for _ in range(2)]
            if r[0] != r[1]:
                raise RuntimeError(f'{op} {form}: runs disagree')
            np.save(path, np.array(r[0][C:C+(1 << 16)], np.uint32))
            print(f'saved {path}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
