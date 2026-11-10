#!/usr/bin/env python3
"""probe_fetch_groups.py: do SIMD groups on one core share the instruction fetch of code beyond the i-cache?

Each SIMD group first spins in a small (cached) loop for (thread_position_in_threadgroup.x * K) trips (uniform 6 = K:
group g ends it ~32 g K trips after group 0), then runs T trips (uniform 5) of an outer loop whose body is BODY_KB of
straight-line independent iadds (beyond the 12 KiB instruction cache: every trip streams the body from memory).
K = 0: the groups start together (one fetch per line serves them all, if fetch is shared); a K that staggers them by
more than the i-cache's worth of code: each group must fetch every line itself. Comparing G = 1, 2, 4, 8 groups per
core (one workgroup of 32 G threads per core, 38 workgroups) separates a per-core serial fetch (time ~ G x) from fetches
that proceed in parallel (time ~ 1 x plus the stagger).

usage: probe_fetch_groups.py [hw|sim|both] ; env G="1 2 4 8" K="0 64" T=8 BODY_KB=48 B2B=1000
Prints one line per (G, K): hw / sim us.
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tg_alu as P


def program(body_kb):
    head = P.e('iadd r0l, 0, 0') + P.e('get_sr r6, sr48', 'get_sr r6, sr48 (thread_position_in_threadgroup.x)')
    head += P.e('imadd r6, r6, u6, 0') + P.mov(3, 0) + P.e('iadd r4, r6, 1')
    for r in range(21, 29):
        head += P.mov(r, r)
    # stagger: a small loop (cached), per-lane trip count
    spin = P.e('iadd r3, 1, r3.sx') + P.e('while_icmp r0l, ult, r3, r4, 1')
    code = head + P.e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1') + spin
    code += P.e(f'jmp_exec_any pc-{len(spin)}', f'jmp_exec_any pc-{len(spin)}') + P.e('pop_exec r0l, 1')
    # streaming loop: T trips of BODY_KB straight-line code
    code += P.mov(7, 0) + P.e('iadd r8, u5, 0')
    code += P.e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    one = [P.e(f'iadd r{21 + i % 8}, r{21 + i % 8}, 1') for i in range(8)]
    body = b''
    i = 0
    while len(body) < body_kb * 1024:
        body += one[i % 8]; i += 1
    body += P.e('iadd r7, 1, r7.sx') + P.e('while_icmp r0l, ult, r7, r8, 1')
    code += body + P.e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}') + P.e('pop_exec r0l, 1')
    return code + P.e('stop') + bytes(64)


def run(mode, code, g, k, t, d):
    f = os.path.join(d, 'k.bin'); open(f, 'wb').write(code)
    common = [f, '--grid', f'{38 * 32 * g},1,1', '--wg', f'{32 * g},1,1', '--regs', '64', '--uniform', f'5={t}', '--uniform', f'6={k}']
    if mode == 'hw':
        chk = subprocess.run([P.RUN] + common, capture_output=True, text=True)
        if '"instructions"' not in chk.stdout:
            raise RuntimeError('emulator gate: ' + chk.stdout[-300:] + chk.stderr[-300:])
        h = subprocess.run([P.HW] + common + ['--back-to-back', os.environ.get('B2B', '1000'), '--timeout-ms', '10000'], capture_output=True, text=True)
        m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
        if not m:
            raise RuntimeError('hw: ' + (h.stdout + h.stderr)[-300:])
        return int(m.group(1)) / 1000
    r = subprocess.run([P.DISP] + common + ['--launch-profile', 'runtime'], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])['time_us']


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    gs = [int(x) for x in os.environ.get('G', '1 2 4 8').split()]
    ks = [int(x) for x in os.environ.get('K', '0 64').split()]
    t = int(os.environ.get('T', '8'))
    code = program(int(os.environ.get('BODY_KB', '48')))
    with tempfile.TemporaryDirectory() as d:
        for k in ks:
            for g in gs:
                out = [f'G {g} K {k} T {t} code {len(code)}']
                for m in (['hw', 'sim'] if mode == 'both' else [mode]):
                    out.append(f'{m} {run(m, code, g, k, t, d):9.1f}')
                print('  '.join(out), flush=True)


if __name__ == '__main__':
    main()
