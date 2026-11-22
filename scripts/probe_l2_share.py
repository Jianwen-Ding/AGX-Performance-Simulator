#!/usr/bin/env python3
"""probe_l2_share.py: is the cache below L1 (the L2) private per core or shared, in capacity and in bandwidth?

Each workgroup (WG threads, default 256 = 8 SIMD groups; one per core while N <= 32) loops T trips over its own
region of REGKB KiB of device memory: per trip 4 coalesced loads (lanes on consecutive words, the 4 loads WG
words apart), wait, 4 fadds; the next trip starts 4 x WG words on (wrapping in the region). Regions of distinct
workgroups do not overlap. If the level that holds REGKB is private per core, time(N) == time(1); if it is shared
GPU-wide, per-workgroup time grows with N once N x REGKB exceeds it (capacity) or the traffic saturates it
(bandwidth). MMQ (352 KB of stack per workgroup) and other stack-heavy kernels undershoot only at full grids.

usage: probe_l2_share.py [hw|sim|both]; env N="1 8 16 32" REGKB="64 256 1024" T=4000 WG=256 B2B=500
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tg_alu as P

A = 0x1100000000


def build():
    e = P.e
    code = e('iadd r0l, 0, 0') + P.mov(3, 0) + e('iadd r4, u5, 0') + P.mov(20, 0) + P.mov(30, 0)
    code += e('get_sr r11, sr48', 'get_sr r11, sr48 (thread_position_in_threadgroup.x)')
    code += e('get_sr r15, sr0', 'get_sr r15, sr0 (threadgroup_position_in_grid.x)')
    code += e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    body = e('iadd r21, r20, r11') + e('and r21, r21, u16') + e('imadd r23, r15, u18, r21')
    body += e('iadd r24, r23, u17') + e('iadd r25, r24, u17') + e('iadd r26, r25, u17')
    for j, r in enumerate((23, 24, 25, 26)):
        body += e(f'device_load 0, i32, x, r{12 + j}, u8_u9, r{r}, unsigned')
    body += e('wait 0')
    for j in range(4):
        body += e(f'fadd32 r30, r30, r{12 + j}')
    body += e('iadd r20, r20, u19')
    body += e('iadd r3, 1, r3.sx') + e('while_icmp r0l, ult, r3, r4, 1')
    code += body + e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}') + e('pop_exec r0l, 1')
    return code + e('stop') + bytes(64)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    wg = int(os.environ.get('WG', '256'))
    t = int(os.environ.get('T', '4000'))
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'k.bin'); open(f, 'wb').write(build())
        for kb in [int(x) for x in os.environ.get('REGKB', '64 256 1024').split()]:
            regw = kb * 256
            for n in [int(x) for x in os.environ.get('N', '1 8 16 32').split()]:
                a = [f, '--grid', f'{n * wg},1,1', '--wg', f'{wg},1,1', '--regs', '64', '--uniform', f'5={t}',
                     '--uniform', f'8={A & 0xffffffff}', '--uniform', f'9={A >> 32}', '--uniform', f'16={regw - 1}',
                     '--uniform', f'17={wg}', '--uniform', f'18={regw}', '--uniform', f'19={4 * wg}',
                     '--alloc', f'{A:#x}={(n * regw + 4 * wg) * 4}']
                out = [f'REGKB {kb:5d} N {n:3d} T {t}']
                if mode in ('hw', 'both'):
                    chk = subprocess.run([P.RUN] + a, capture_output=True, text=True)
                    if '"instructions"' not in chk.stdout:
                        raise RuntimeError('emulator gate: ' + chk.stdout[-300:] + chk.stderr[-300:])
                    h = subprocess.run([P.HW] + a + ['--back-to-back', os.environ.get('B2B', '500'), '--timeout-ms', '20000'],
                                       capture_output=True, text=True)
                    m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
                    out.append(f'hw {int(m.group(1)) / 1000:9.1f}' if m else 'hw FAILED ' + (h.stdout + h.stderr)[-200:])
                if mode in ('sim', 'both'):
                    r = subprocess.run([P.DISP] + a + ['--launch-profile', 'runtime'], capture_output=True, text=True)
                    try:
                        out.append(f"sim {json.loads(r.stdout.strip().splitlines()[-1])['time_us']:9.1f}")
                    except Exception:
                        out.append('sim FAILED ' + (r.stdout + r.stderr)[-200:])
                print('  '.join(out), flush=True)


if __name__ == '__main__':
    main()
