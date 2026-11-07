#!/usr/bin/env python3
"""probe_stack_fresh.py: short-lived threads that touch a fresh stack (bfs-like kernels built at -O0 / with spills).

Each thread stores K words to its stack (offsets 0, 4, ..., lane-interleaved stack layout), waits, loads them back
(LOADS=1) and stops: the whole cost is the stack traffic of many short workgroups. Hardware (agx_run_hw back to back,
--stack 4K) against agx_dispatch (zluda profile). env: KS="0 2 8 16", THREADS (default 1000448), WG (512), LOADS.

usage: probe_stack_fresh.py [hw|sim|both]
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_tg_alu import e, mov, HW, DISP, RUN


def program(k, loads):
    # the stack set-up store that the LLVM backend emits first in every kernel with a stack (copied from a working
    # clang-built kernel: `stack_store i8, 1, 2, 0, 20, 0`); without it stack accesses fault (Bad address)
    b = e('iadd r0l, 0, 0') + bytes.fromhex('b500410521800000') + mov(6, 7)
    for j in range(k):
        b += e(f'stack_store i32, 1, 0, x, 4, r6, {4 * j}, 0')
    if k:
        b += e('wait 0')
    if loads:
        for j in range(k):
            b += e(f'stack_load r{8 + j % 8}, i32, 1, 0, x, 4, {4 * j}, 0')
        if k:
            b += e('wait 0')
    return b + e('stop') + bytes(64)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    threads, wg = int(os.environ.get('THREADS', '1000448')), int(os.environ.get('WG', '512'))
    loads = os.environ.get('LOADS', '1') == '1'
    with tempfile.TemporaryDirectory() as d:
        for k in [int(x) for x in os.environ.get('KS', '0 2 8 16').split()]:
            f = os.path.join(d, 'k.bin'); open(f, 'wb').write(program(k, loads))
            stack = max(16, 4 * k)
            common = [f, '--grid', f'{threads},1,1', '--wg', f'{wg},1,1', '--regs', '40']
            rec = {'k': k, 'loads': loads, 'threads': threads, 'wg': wg}
            if mode in ('hw', 'both'):
                args = [HW] + common + ['--stack', str(stack), '--back-to-back', os.environ.get('B2B', '300'), '--timeout-ms', '10000']
                subprocess.run(args, capture_output=True, text=True)
                h = subprocess.run(args, capture_output=True, text=True)
                m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
                rec['hw_us'] = int(m.group(1)) / 1000 if m else (h.stdout + h.stderr)[-200:]
            if mode in ('sim', 'both'):
                r = subprocess.run([DISP] + common + ['--stack', str(stack), '--launch-profile', 'zluda', '--alloc', '0x100000=16'], capture_output=True, text=True,
                                   env=dict(os.environ, AGX_MEM_LEVELS='1'))
                lines = r.stdout.strip().splitlines()
                rec['sim_us'] = json.loads(lines[0])['time_us'] if lines else r.stderr[-200:]
                rec['levels'] = (r.stdout + r.stderr).strip().splitlines()[-1][:120]
            print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
