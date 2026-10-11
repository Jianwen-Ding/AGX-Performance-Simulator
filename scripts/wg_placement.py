#!/usr/bin/env python3
"""Where workgroups land: every lane stores (SR20 core << 16) | SR0 workgroup id; a short counted loop keeps
workgroups resident long enough to overlap. Prints per-core workgroup counts (direct DRM, known-safe forms)."""
import argparse, json, struct
from collections import Counter
from pathlib import Path
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm

e, mov = agx_loop.encode, B.mov


def body(trips):
    b = B.get_sr(20, 28)+B.get_sr(0, 27)+mov(26, 65536)+e('imadd r28, r28, r26, r27')
    b += mov(27, B.TS_BASE)+e('iadd r27, r1, r27')+agx_loop.store(28, 8, 27)+e('wait 0')
    return b+B.loop('fadd32_imm', 8, 64, trips)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wgs', default='38,76,114,152,190,228,304,608')
    ap.add_argument('--wg', default='256')
    ap.add_argument('--trips', type=int, default=200)
    ap.add_argument('--output', type=Path, default=Path('/scratch/apple-m2-playground/sim/placement'))
    a = ap.parse_args()
    gpu_safety.require_budget(1)
    R = B.Runner(a.output)
    rows = []
    for wg in map(int, a.wg.split(',')):
        for n in map(int, a.wgs.split(',')):
            count = n*wg
            words = B.TS_BASE+count
            (R.out/'t.bin').write_bytes(bytes(4*words))
            import subprocess, os
            (R.out/'c.bin').write_bytes(drm.build_main(body(a.trips))+bytes(16))
            env = {'PATH': '/usr/bin', 'AGX_LOCAL_X': str(wg), 'AGX_TABLE_OUT': str(R.out/'t.out')}
            r = subprocess.run([str(R.exe), '/dev/dri/renderD128', str(R.out/'c.bin'), str(count), str(R.out/'rb'), '2', '64',
                                str(R.out/'t.bin')], capture_output=True, timeout=60, env=env)
            if r.returncode: raise SystemExit(r.stdout.decode()[-300:])
            t = struct.unpack(f'<{words}I', (R.out/'t.out').read_bytes()[:4*words])
            per_wg = {}
            for i in range(0, count, 32):
                v = t[B.TS_BASE+i]; per_wg[v & 0xffff] = v >> 16
            per_core = Counter(per_wg.values())
            hist = Counter(per_core.values())
            row = {'wg': wg, 'workgroups': n, 'cores_used': len(per_core), 'max_per_core': max(per_core.values()),
                   'histogram': dict(sorted(hist.items())), 'order': [per_wg[k] for k in sorted(per_wg)][:80]}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'order'}), flush=True)
    (a.output/'placement.json').write_text(json.dumps(rows)+'\n')


if __name__ == '__main__':
    main()
