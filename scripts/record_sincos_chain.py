#!/usr/bin/env python3
"""Mesa's AGX sin / cos sequence on the GPU for given fp32 inputs (direct DRM), for the PTX sin/cos.approx error check.

agx_compile.c lowers fsin / fcos to: turns = x * (float)(0.5 / pi) [+ 0.25 for cos]; fract = turns - floor(turns)
(nir lower_ffract); quadrants = fract * 4; y = sin_pt_1(quadrants); result = sin_pt_2(y) * y. This kernel runs exactly
those instructions (fmul32 / fadd32 / floor / sin_pt_1 / sin_pt_2, all forms recorded safely by the coverage suite)
per lane, straight-line, and stores the result. Each 2^18-lane chunk runs twice and must agree.
Usage: record_sincos_chain.py sin|cos|sin2|cos2 INPUTS.npy OUT.npy  (sin2 / cos2: turns = fma(x, C_hi, x * C_lo))   (INPUTS: uint32 fp32 bit patterns)
"""
import struct, sys
from pathlib import Path
import numpy as np
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm
from record_suite import run, gate
import record_transcendentals as T

e, mov = agx_loop.encode, B.mov
f2b = lambda v: struct.unpack('<I', struct.pack('<f', v))[0]


def body(kind):
    b = e('device_load 0, i32, x, r6, u8_u9, r1, unsigned')+e('wait 0')
    b += mov(8, f2b(np.float32(0.5/np.pi)))+mov(9, f2b(0.25))+mov(12, f2b(4.0))
    if kind.endswith('2'):     # Cody-Waite turns: fma(x, C_hi, x * C_lo)
        lo = np.float32(0.5/np.pi - np.float64(np.float32(0.5/np.pi)))
        b += mov(15, f2b(lo))+e('fmul32 r16, r6, r15')+e('fmadd32 r7, r6, r8, r16')
    else:
        b += e('fmul32 r7, r6, r8')
    if kind.startswith('cos'):
        b += e('fadd32 r7, r7, r9')
    b += e('floor r10, r7')+e('fadd32 r10, r7, r10.neg')+e('fmul32 r10, r10, r12')
    b += e('sin_pt_1 r11, r10')+e('sin_pt_2 r13, r11')+e('fmul32 r14, r13, r11')
    b += mov(27, T.CHUNK)+e('iadd r26, r1, r27')+agx_loop.store(14, 8, 26)+e('wait 0')+e('iadd r2, r14, 0')
    return b


def main():
    kind, inp, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    xs = np.load(inp).astype(np.uint32)
    gate(body(kind))
    gpu_safety.require_budget(1)
    R = B.Runner(Path('/scratch/apple-m2-playground/sim/sincos'))
    code = drm.build_main(body(kind))
    out = np.zeros(len(xs), np.uint32)
    C = T.CHUNK
    for s in range(0, len(xs), C):
        part = [int(x) for x in xs[s:s+C]]
        n = len(part)
        table = part+[0]*(C-n)+[0]*C
        r = [run(R, code, C, T.WG, table)[1] for _ in range(2)]
        if r[0] != r[1]:
            raise RuntimeError(f'chunk {s}: two runs disagree')
        out[s:s+n] = r[0][C:C+n]
    np.save(outp, out)
    print(f'{kind}: {len(xs)} inputs recorded to {outp}')


if __name__ == '__main__':
    main()
