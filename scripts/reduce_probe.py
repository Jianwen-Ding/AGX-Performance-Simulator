#!/usr/bin/env python3
"""End-to-end validation kernel: per-workgroup sum reduction (direct DRM, validated forms only).

Thread with global index g and local index l (SR51) of a 256-thread workgroup:
  1. s = sum_{i < T} table[g + i*N]  (N = grid threads; 8 loads per wait, grid stride)
  2. tg[4l] = s; barrier
  3. for k in 128, 64, ..., 1: if (l < k) tg[4l] += tg[4(l+k)]; barrier   (barriers outside the ifs)
  4. r2 = tg[0]  -> the harness stores r2 to out[8 + g]
table[w] = w, so every thread of workgroup j must store sum over its threads t and i of (t + i*N) mod 2^32.
Every dispatch is checked against that oracle; timing is the minimum firmware duration after a warm-up.
"""
import argparse, json, statistics, struct, subprocess, time
from pathlib import Path
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm
from run_drm_tg_layout import tg, shared_word

e, mov = agx_loop.encode, B.mov
WG = 256
ROOT = Path(__file__).resolve().parents[1]


def jnone(n):
    return e(f'jmp_exec_none pc+{n}', f'jmp_exec_none pc+{n}')


def body(n, trips):
    """trips x 8 loads per thread."""
    b = B.get_sr(51, 23)+mov(2, 0)+e('iadd r5, r1, 0')+mov(14, n)
    # 1. grid-stride sum, 8 loads per wait
    b += mov(3, 0)+mov(4, trips)+e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    lp = b''
    for k in range(8):
        lp += e(f'device_load 0, i32, x, r{6+k}, u8_u9, r5, unsigned')+e('iadd r5, r5, r14')
    lp += e('wait 0')+b''.join(e(f'iadd r2, r2, r{6+k}') for k in range(8))
    lp += e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
    b += lp+e(f'jmp_exec_any pc-{len(lp)}', f'jmp_exec_any pc-{len(lp)}')+e('pop_exec r0l, 1')
    # 2. tg[4l] = s
    b += mov(24, 4)+e('imadd r24, r23, r24, 0')+tg(False, 'i32', 2, base=24, index=0)+e('wait 0')+bytes.fromhex('6800')
    # 3. tree
    k = WG//2
    while k >= 1:
        inner = mov(25, 4*k)+e('iadd r25, r24, r25')+tg(True, 'i32', 6, base=24, index=0)+tg(True, 'i32', 7, base=25, index=0)
        inner += e('wait 0')+e('iadd r6, r6, r7')+tg(False, 'i32', 6, base=24, index=0)+e('wait 0')
        b += mov(26, k)+e('if_icmp r0l, ult, r23, r26, 1')+jnone(6+len(inner))+inner+e('pop_exec r0l, 1')+bytes.fromhex('6800')
        k //= 2
    # 4. r2 = tg[0]
    b += mov(27, 0)+tg(True, 'i32', 2, base=27, index=0)+e('wait 0')
    return b


def oracle(n, trips):
    groups = n//WG
    steps = 8*trips
    out = []
    for j in range(groups):
        t0 = j*WG
        s = sum((t0+t+i*n) for t in range(WG) for i in range(steps)) & 0xffffffff
        out.extend([s]*WG)
    return out


def oracle_fast(n, trips):
    steps = 8*trips
    out = []
    for j in range(n//WG):
        t0 = j*WG
        s = (steps*(WG*t0+WG*(WG-1)//2)+WG*n*steps*(steps-1)//2) & 0xffffffff
        out.extend([s]*WG)
    return out


def run(R, code_body, n, trips, repeats):
    words = n*8*trips
    import array
    tbl = array.array('I', range(words))
    (R.out/'t.bin').write_bytes(tbl.tobytes())
    (R.out/'c.bin').write_bytes(drm.build_main(code_body)+bytes(16))
    env = {'PATH': '/usr/bin', 'AGX_LOCAL_X': str(WG), 'AGX_SHARED_WORD': str(shared_word(4*WG))}
    r = subprocess.run([str(R.exe), '/dev/dri/renderD128', str(R.out/'c.bin'), str(n), str(R.out/'rb'), str(repeats), '64',
                        str(R.out/'t.bin')], capture_output=True, timeout=60, env=env)
    smp = [json.loads(l) for l in r.stdout.decode().splitlines() if '"rep"' in l]
    if r.returncode or len(smp) != repeats or not all(x['fw_ts'][0] and x['fw_ts'][1] for x in smp):
        gpu_safety.record('reduce_probe fault'); raise RuntimeError(r.stdout.decode()[-300:])
    w = struct.unpack(f'<{n+32}I', (R.out/'rb').read_bytes()[:4*(n+32)])[8:8+n]
    return [x['fw_ts'][1]-x['fw_ts'][0] for x in smp], list(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', default='9728:1,9728:16,38912:16,155648:8,155648:32,1048576:4')
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--output', type=Path, default=Path('/scratch/apple-m2-playground/sim/reduce'))
    a = ap.parse_args()
    assert oracle(512, 2) == oracle_fast(512, 2)
    gpu_safety.require_budget(1)
    R = B.Runner(a.output)
    rows = []
    for c in a.cases.split(','):
        n, trips = map(int, c.split(':'))
        b = body(n, trips)
        R.run(B.loop('fmadd32_imm', 8, 64, 4000), 155648, 3, 256, gprs=64)   # warm-up
        fw, out = run(R, b, n, trips, a.repeats)
        ok = out == oracle_fast(n, trips)
        (a.output/f'reduce_{n}_{trips}.bin').write_bytes(drm.build_main(b))
        row = {'grid': n, 'trips': trips, 'loads_per_thread': 8*trips, 'exact': ok, 'fw_ns': fw, 'min_us': min(fw)/1e3,
               'code': f'reduce_{n}_{trips}.bin'}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'fw_ns'}), flush=True)
        if not ok: raise SystemExit('oracle mismatch: stopping')
    (a.output/'reduce.json').write_text(json.dumps({'wg': WG, 'rows': rows})+'\n')


if __name__ == '__main__':
    main()
