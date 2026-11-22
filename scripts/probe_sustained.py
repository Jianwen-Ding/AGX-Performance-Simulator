#!/usr/bin/env python3
"""probe_sustained.py: does a core run slower when every core is busy, or when the load lasts milliseconds
(power / clock limits under all-core load)?

An ALU loop (probe_tg_alu.program PATTERN, default 8 independent fmadd32 per trip) in workgroups of WG threads
(default 256 = 8 SIMD groups), one per core at a time. For each grid N (workgroups) and trip count T: hw b2b median
and sim. If cores do not interfere, time(N=38) == time(N=1) and time(T) is linear in T.

usage: probe_sustained.py [hw|sim|both]; env N="1 8 38 76" T="2000 20000 200000" PAT=ffffffff WG=256 B2B=200
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tg_alu as P


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    wg = int(os.environ.get('WG', '256'))
    code = P.program(os.environ.get('PAT', 'ffffffff'))
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'k.bin'); open(f, 'wb').write(code)
        for t in [int(x) for x in os.environ.get('T', '2000 20000 200000').split()]:
            for n in [int(x) for x in os.environ.get('N', '1 8 38 76').split()]:
                a = [f, '--grid', f'{n * wg},1,1', '--wg', f'{wg},1,1', '--regs', '64', '--uniform', f'5={t}', '--shared', '4096']
                out = [f'T {t:7d} N {n:4d}']
                if mode in ('hw', 'both'):
                    chk = subprocess.run([P.RUN] + a, capture_output=True, text=True)
                    if '"instructions"' not in chk.stdout:
                        raise RuntimeError('emulator gate: ' + chk.stdout[-300:] + chk.stderr[-300:])
                    h = subprocess.run([P.HW] + a + ['--back-to-back', os.environ.get('B2B', '200'), '--timeout-ms', '20000'],
                                       capture_output=True, text=True)
                    m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
                    out.append(f'hw {int(m.group(1)) / 1000:10.1f}' if m else 'hw FAILED ' + (h.stdout + h.stderr)[-200:])
                if mode in ('sim', 'both'):
                    s = [x if x != '--shared' else '--tg-bytes' for x in a]
                    r = subprocess.run([P.DISP] + s + ['--launch-profile', 'runtime'], capture_output=True, text=True)
                    out.append(f"sim {json.loads(r.stdout.strip().splitlines()[-1])['time_us']:10.1f}")
                print('  '.join(out), flush=True)


if __name__ == '__main__':
    main()
