#!/usr/bin/env python3
"""probe_resident_order.py: with one resident workgroup per core, how are the workgroups beyond the first round placed?

N workgroups of 128 threads with SHARED (default 40000) bytes of threadgroup memory (one fits a core). Each runs an
in-cache loop (4 KB of straight-line iadds) for u5 trips, or u10 trips when its workgroup id is in [u7, u11) (the
"long" ones), then stores its core id (sr20) at word [workgroup id] of the output buffer.

Static pairing (workgroup i and i + 34 on one core) vs dynamic (a workgroup starts on whichever core frees first):
with N = 38 and workgroups 0-3 long (L trips) and the others short (S): static L + S, dynamic max(L, 2 S).

usage: probe_resident_order.py [hw|sim|both]; env N=38 S=100 L=300 LONG=0:4 SHARED=40000 B2B=5000 DUMP=1
Prints per case: hw / sim us, and (hw, DUMP=1) the core id of every workgroup.
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tg_alu as P

OUT = 0x1100000000


def program(body_kb=4):
    code = P.e('iadd r0l, 0, 0')
    code += P.e('get_sr r10, sr0', 'get_sr r10, sr0 (threadgroup_position_in_grid.x)')
    code += P.e('get_sr r11, sr20', 'get_sr r11, sr20 (core_index)')
    # trips: u10 when u7 <= wg < u11, else u5
    code += P.e('icmpsel ult, r8, r10, u11, u10, u5')
    code += P.e('icmpsel ult, r8, r10, u7, u5, r8')
    code += P.mov(7, 0)
    for r in range(21, 29):
        code += P.mov(r, r)
    code += P.e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    one = [P.e(f'iadd r{21 + i % 8}, r{21 + i % 8}, 1') for i in range(8)]
    body = b''
    i = 0
    while len(body) < body_kb * 1024:
        body += one[i % 8]; i += 1
    body += P.e('iadd r7, 1, r7.sx') + P.e('while_icmp r0l, ult, r7, r8, 1')
    code += body + P.e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}') + P.e('pop_exec r0l, 1')
    code += P.e('device_store 0, i32, x, r11, u8_u9, r10, unsigned, lsl 0, 0', 'device_store 0, i32, x, r11, u8_u9, r10, unsigned, 0')
    return code + P.e('stop') + bytes(64)


def args(f, n, s, l, lo, hi, shared):
    return [f, '--grid', f'{n * 128},1,1', '--wg', '128,1,1', '--regs', '64', '--shared', str(shared),
            '--uniform', f'5={s}', '--uniform', f'10={l}', '--uniform', f'7={lo}', '--uniform', f'11={hi}',
            '--uniform', '8=0', '--uniform', '9=17', '--alloc', f'{OUT:#x}=65536']


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    n = int(os.environ.get('N', '38')); s = int(os.environ.get('S', '100')); l = int(os.environ.get('L', '300'))
    lo, hi = (int(x) for x in os.environ.get('LONG', '0:4').split(':'))
    shared = int(os.environ.get('SHARED', '40000'))
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'k.bin'); open(f, 'wb').write(program())
        a = args(f, n, s, l, lo, hi, shared)
        out = [f'N {n} S {s} L {l} LONG {lo}:{hi} SHARED {shared}']
        chk = subprocess.run([P.RUN] + a, capture_output=True, text=True)
        if '"instructions"' not in chk.stdout:
            raise RuntimeError('emulator gate: ' + chk.stdout[-300:] + chk.stderr[-300:])
        if mode in ('hw', 'both'):
            if os.environ.get('DUMP', '1') == '1':
                dump = os.path.join(d, 'ids.out')
                h = subprocess.run([P.HW] + a + ['--dump', f'{OUT:#x}={4 * n}:{dump}'], capture_output=True, text=True)
                raw = open(dump, 'rb').read() if os.path.exists(dump) else b''
                ids = [int.from_bytes(raw[4 * i:4 * i + 4], 'little') for i in range(len(raw) // 4)]
                print('cores', ' '.join(map(str, ids)), flush=True)
                if not ids:
                    print(h.stdout[-300:], h.stderr[-300:])
            h = subprocess.run([P.HW] + a + ['--back-to-back', os.environ.get('B2B', '5000'), '--timeout-ms', '10000'],
                               capture_output=True, text=True)
            m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
            out.append(f'hw {int(m.group(1)) / 1000:9.1f}' if m else 'hw FAILED ' + (h.stdout + h.stderr)[-200:])
        if mode in ('sim', 'both'):
            r = subprocess.run([P.DISP] + [('--tg-bytes' if x == '--shared' else x) for x in a] + ['--launch-profile', 'runtime'], capture_output=True, text=True)
            out.append(f"sim {json.loads(r.stdout.strip().splitlines()[-1])['time_us']:9.1f}")
        print('  '.join(out), flush=True)


if __name__ == '__main__':
    main()
