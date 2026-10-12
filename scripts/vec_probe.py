#!/usr/bin/env python3
"""Semantics of vector / narrow device loads (direct DRM): table word w = w*0x10001+0x01000000*(w&7) is not used;
table word w = w, read with `device_load 0, <fmt>, <mask>, <regs>, u8_u9, r5, unsigned` with r5 = lane, and each
destination register returned in r2 by a separate dispatch."""
import array, json, struct, subprocess
from pathlib import Path
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm
e, mov = agx_loop.encode, B.mov
FORMS = {
    'i32.xyzw': ('device_load 0, i32, xyzw, r13_r14_r15_r16, u8_u9, r5, unsigned', [13, 14, 15, 16]),
    'i32.xy': ('device_load 0, i32, xy, r13_r14, u8_u9, r5, unsigned', [13, 14]),
    'i16.x': ('device_load 0, i16, x, r13l, u8_u9, r5, unsigned', [13]),
    'i8.x': ('device_load 0, i8, x, r13l, u8_u9, r5, unsigned', [13]),
    'i16.xyzw': ('device_load 0, i16, xyzw, r13l_r13h_r14l_r14h, u8_u9, r5, unsigned', [13, 14]),
}


def run(R, body, n, words):
    (R.out/'t.bin').write_bytes(array.array('I', range(words)).tobytes())
    (R.out/'c.bin').write_bytes(drm.build_main(body)+bytes(16))
    r = subprocess.run([str(R.exe), '/dev/dri/renderD128', str(R.out/'c.bin'), str(n), str(R.out/'rb'), '1', '64', str(R.out/'t.bin')],
                       capture_output=True, timeout=30, env={'PATH': '/usr/bin', 'AGX_LOCAL_X': '32'})
    if r.returncode: gpu_safety.record('vec_probe fault'); raise RuntimeError(r.stdout.decode()[-300:])
    return list(struct.unpack(f'<{n+32}I', (R.out/'rb').read_bytes()[:4*(n+32)])[8:8+n])


def main():
    gpu_safety.require_budget(1)
    R = B.Runner(Path('/scratch/apple-m2-playground/sim/vec'))
    out = {}
    for name, (text, regs) in FORMS.items():
        for reg in regs:
            body = e('iadd r5, r1, 0')+b''.join(mov(r, 0xdeadbeef) for r in range(13, 17))+e(text)+e('wait 0')+e(f'iadd r2, r{reg}, 0')
            v = run(R, body, 64, 4096)
            out[f'{name}/r{reg}'] = v
            print(name, f'r{reg}', [hex(x) for x in v[:6]], flush=True)
    Path('/scratch/apple-m2-playground/sim/vec/semantics.json').write_text(json.dumps(out)+'\n')


if __name__ == '__main__':
    main()
