#!/usr/bin/env python3
"""Record the exact fp32 results of the AGX transcendental instructions (hardware, direct DRM).

The functional simulator models rcp / rsqrt / rsqrt_special / log2 / exp2 / sin_pt_2 on fp32 operands as the
correctly rounded value, but the hardware uses a ~26-bit approximation: 10-15% of the test suite's lanes differ by
1-3 ulp, which is what still fails in `make suite-check` (8 alu cases and the random programs using them). The
suite's 1,024 samples per instruction cannot pin the approximation down, so this script sweeps every mantissa.

Tables (testsuite/transcendental/<op>.bin.gz): one int8 per input = hardware bits - reference bits, where the
reference is the exact value computed in float64 and rounded to float32 (the same formula the simulator uses:
`(1/x) as f32`, `(1/sqrt(x)) as f32`). Index order:
  rcp            x = 1.m             (2^23 entries, mantissa m)
  rsqrt(_special) x = 1.m, then 2.m  (2^24 entries: exponent 127 then 128)
The simulator applies a table when present (rust/src/transtab.rs) and scales by the exponent; this script also
checks that scaling on random exponents and stops if it does not hold.

Samples (testsuite/transcendental/samples/<op>_<lo>.bin.gz, raw u32 results for x = lo * 1.m over all 2^23
mantissas): log2 on [1,2), [2,4), [0.5,1); exp2 on [1,2), [0.5,1), [-1,-0.5); sin_pt_2 on [0.5,1), [0.25,0.5).
These functions are not separable by exponent; the samples are for fitting their algorithm offline.

Each dispatch runs CHUNK lanes: r6 = table[gid]; <op> r14, r6; table[CHUNK + gid] = r14. Every dispatch is run twice
and must agree. Usage: record_transcendentals.py [--only rcp,rsqrt,...] [--out testsuite/transcendental]
"""
import argparse, gzip, json, random, struct, sys
from pathlib import Path
import numpy as np
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm
from record_suite import run, gate

e, mov = agx_loop.encode, B.mov
ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1 << 18
WG = 256


def body(op):
    b = e('device_load 0, i32, x, r6, u8_u9, r1, unsigned')+e('wait 0')
    b += e(f'{op} r14, r6')
    b += mov(27, CHUNK)+e('iadd r26, r1, r27')+agx_loop.store(14, 8, 26)+e('wait 0')+e('iadd r2, r14, 0')
    return b


def results(R, op, inputs):
    """hardware result bits for a list/array of input bits (any length; padded to whole chunks)"""
    code = drm.build_main(body(op))
    gate(body(op))
    out = np.zeros(len(inputs), dtype=np.uint32)
    for s in range(0, len(inputs), CHUNK):
        part = [int(x) for x in inputs[s:s+CHUNK]]
        n = len(part)
        table = part+[0]*(CHUNK-n)+[0]*CHUNK
        r = [run(R, code, CHUNK, WG, table)[1] for _ in range(2)]
        if r[0] != r[1]:
            raise RuntimeError(f'{op}: two runs of chunk {s} disagree')
        out[s:s+n] = r[0][CHUNK:CHUNK+n]
        print(json.dumps({'op': op, 'done': s+n, 'of': len(inputs)}), flush=True)
    return out


def reference(op, xbits):
    x = xbits.view(np.float32).astype(np.float64)
    with np.errstate(all='ignore'):
        y = 1.0/x if op == 'rcp' else 1.0/np.sqrt(x)
    return y.astype(np.float32).view(np.uint32)


def mantissas(exp_bits):
    return (np.uint32(exp_bits << 23) | np.arange(1 << 23, dtype=np.uint32)).astype(np.uint32)


def save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as f:
        f.write(arr.tobytes())


def table_op(R, op, out):
    exps = [127] if op == 'rcp' else [127, 128]
    xs = np.concatenate([mantissas(x) for x in exps])
    hw = results(R, op, xs)
    delta = hw.astype(np.int64)-reference(op, xs).astype(np.int64)
    if np.abs(delta).max() > 127:
        raise RuntimeError(f'{op}: delta outside int8 ({np.abs(delta).max()})')
    save(out/f'{op}.bin.gz', delta.astype(np.int8))
    print(json.dumps({'op': op, 'entries': len(xs), 'delta_histogram': {int(k): int(v) for k, v in zip(*np.unique(delta, return_counts=True))}}))
    # exponent scaling check on random normal inputs whose results stay normal
    rng = random.Random(op)
    xs2 = np.array([(rng.randrange(1, 254) if op == 'rcp' else rng.randrange(2, 252)) << 23 | rng.getrandbits(23) for _ in range(1 << 16)], dtype=np.uint32)
    hw2 = results(R, op, xs2)
    bad = 0
    for x, h in zip(xs2, hw2):
        ex = int(x) >> 23 & 0xff; m = int(x) & 0x7fffff
        if op == 'rcp':
            t = int(hw[m]); re = (t >> 23 & 0xff)-(ex-127)
            want = None if not 0 < re < 255 else (re << 23) | (t & 0x7fffff)
        else:
            odd = (ex-127) & 1; t = int(hw[odd << 23 | m]); re = (t >> 23 & 0xff)-(ex-127-odd)//2
            want = None if not 0 < re < 255 else (re << 23) | (t & 0x7fffff)
        if want is not None and want != int(h): bad += 1
    print(json.dumps({'op': op, 'exponent_scaling_check': 'ok' if bad == 0 else f'{bad} mismatches'}))
    if bad:
        raise RuntimeError(f'{op}: results do not scale with the exponent; the table model does not apply')


SAMPLES = {'log2': [(127, '1'), (128, '2'), (126, '0.5')], 'exp2': [(127, '1'), (126, '0.5'), (126 | 0x100, '-0.5')],
           'sin_pt_2': [(126, '0.5'), (125, '0.25')]}


def sample_op(R, op, out):
    for exp_bits, label in SAMPLES[op]:
        xs = mantissas(exp_bits)
        save(out/'samples'/f'{op}_{label}.bin.gz', results(R, op, xs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', default='rcp,rsqrt,rsqrt_special,log2,exp2,sin_pt_2')
    ap.add_argument('--out', type=Path, default=ROOT/'testsuite/transcendental')
    a = ap.parse_args()
    gpu_safety.require_budget(1)
    R = B.Runner(Path('/scratch/apple-m2-playground/sim/transcendental'))
    for op in a.only.split(','):
        (table_op if op in ('rcp', 'rsqrt', 'rsqrt_special') else sample_op)(R, op, a.out)


if __name__ == '__main__':
    main()
