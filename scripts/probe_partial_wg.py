#!/usr/bin/env python3
"""probe_partial_wg.py: do the SIMD groups of a workgroup that exit early free their slots for the next workgroup?

Workgroups of WG (default 256) threads: lanes with thread_position_in_threadgroup.x >= u12 (ACTIVE groups x 32) jump
straight to the end; the others run T trips (u5) of 8 dependent iadds (a latency-bound chain). REGS halves set the
group cap per core (108 halves: 30 groups = 3 workgroups of 8 groups).

If a workgroup's slots stay held until its last group ends (waves of whole workgroups), N workgroups take about
N / (cores x resident) x the chain latency; if the exited groups' slots are reused at once, far less (down to the
~100 ns per workgroup launch).

usage: probe_partial_wg.py [hw|sim|both]; env N=7168 WG=256 ACTIVE="1 8" T="50 200 800" REGS=108 B2B=2000
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tg_alu as P


def program():
    code = P.e('iadd r0l, 0, 0')
    code += P.e('get_sr r6, sr48', 'get_sr r6, sr48 (thread_position_in_threadgroup.x)')
    code += P.mov(3, 0) + P.mov(21, 0)
    code += P.e('iadd r4, u5, 0')
    code += P.e('if_icmp r0l, ult, r6, u12, 1')
    body = b''.join(P.e('iadd r21, r21, 1') for _ in range(8))
    body += P.e('iadd r3, 1, r3.sx') + P.e('while_icmp r0l, ult, r3, r4, 1')
    loop = body + P.e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}')
    pop = P.e('pop_exec r0l, 1')
    skip = len(loop) + 6  # jmp_exec_none is 6 bytes: over the loop to the pop
    jn = P.e(f'jmp_exec_none pc+{skip}', f'jmp_exec_none pc+{skip}')
    assert len(jn) == 6, len(jn)
    code += jn + loop + pop
    return code + P.e('stop') + bytes(64)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    n = int(os.environ.get('N', '7168')); wg = int(os.environ.get('WG', '256')); regs = os.environ.get('REGS', '108')
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'k.bin'); open(f, 'wb').write(program())
        for act in os.environ.get('ACTIVE', '1 8').split():
            for t in os.environ.get('T', '50 200 800').split():
                a = [f, '--grid', f'{n * wg},1,1', '--wg', f'{wg},1,1', '--regs', regs, '--uniform', f'5={t}',
                     '--uniform', f'12={32 * int(act)}']
                chk = subprocess.run([P.RUN] + a, capture_output=True, text=True)
                if '"instructions"' not in chk.stdout:
                    raise RuntimeError('emulator gate: ' + chk.stdout[-300:] + chk.stderr[-300:])
                out = [f'N {n} WG {wg} REGS {regs} ACTIVE {act} T {t}']
                if mode in ('hw', 'both'):
                    h = subprocess.run([P.HW] + a + ['--back-to-back', os.environ.get('B2B', '2000'), '--timeout-ms', '10000'],
                                       capture_output=True, text=True)
                    m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
                    out.append(f'hw {int(m.group(1)) / 1000:9.1f}' if m else 'hw FAILED ' + (h.stdout + h.stderr)[-200:])
                if mode in ('sim', 'both'):
                    r = subprocess.run([P.DISP] + a + ['--launch-profile', 'runtime'], capture_output=True, text=True)
                    out.append(f"sim {json.loads(r.stdout.strip().splitlines()[-1])['time_us']:9.1f}")
                print('  '.join(out), flush=True)


if __name__ == '__main__':
    main()
