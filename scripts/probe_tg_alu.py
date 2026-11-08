#!/usr/bin/env python3
"""probe_tg_alu.py: threadgroup loads interleaved with ALU work, hardware (agx_run_hw back to back) vs agx_dispatch.

Each SIMD group runs a counted loop (trip count in 32-bit uniform 5) whose body is a pattern of threadgroup loads `L`
and ALU instructions. Pattern letters (one instruction each):
  L  threadgroup_load i32, x, rD, rB l, 0     (D cycles over r13..r20, B over r5..r12: 8 independent loads)
  X  threadgroup_load i32, xy, rD_rD+1, ...    (two words per lane)
  B  threadgroup_load i32, x, rD, 0, 0         (immediate base 0: every lane reads word 0)
  i  iadd rT, rT, 1 on r21..r28 round robin   (independent of the loads)
  f  fmadd32 rT, rT, r29, r30 on r21..r28     (independent fp32)
  c  iadd rT, rT, rD of the latest load       (consumes a load result)
  p, q, r  iadd on 1, 2 or 3 dependency chains (r21, r21..r22, r21..r23)
  w  wait 0
The loop counter uses r3/r4 (iadd + while_icmp + jmp_exec_any: 3 instructions per trip).
Addresses: per-lane words (lane * 4 + 128 * j) or broadcast (one word, ADDR=bcast).

Times are the slope between two trip counts (T1, T2) in core cycles per trip per SIMD group at 1398 MHz, on the
busiest core (32 workgroups, one per core; NG SIMD groups per core: workgroups of 32*NG threads, or WG threads per
workgroup).

usage: probe_tg_alu.py [hw|sim|both] PATTERN... ; env NG="4 16 32", T1, T2, ADDR, WG
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rust', 'target', 'release')
HW = os.environ.get('AGX_RUN_HW', '/scratch/ptx-frontend-build/release/agx_run_hw')
DISP = os.environ.get('AGX_DISPATCH', os.path.join(REL, 'agx_dispatch'))
RUN = os.environ.get('AGX_RUN', os.path.join(REL, 'agx_run'))
ASM = os.environ.get('AGX_ASM', os.path.join(REL, 'agx_asm'))
MHZ = 1398.0
_c = {}


def e(t, want=None):
    k = (t, want)
    if k not in _c:
        line = t + ('\t' + want if want else '') + '\n'
        out = subprocess.run([ASM], input=line, capture_output=True, text=True, check=True).stdout.split()[0]
        if out == 'ERROR':
            raise ValueError(t)
        _c[k] = bytes.fromhex(out)
    return _c[k]


def mov(reg, v):
    return (0x62 | 2 << 7 | (2 * reg) << 9).to_bytes(2, 'little') + (v & 0xffffffff).to_bytes(4, 'little')


def tgload(reg, base, mask=1):
    # threadgroup_load i32, <mask>, r<reg>.., r<base>l, 0 (Mesa's 8-byte form; run_drm_tg_layout.tg); base None: immediate 0
    R, A, At = 2 * reg, 0 if base is None else 2 * base, 0b10 if base is None else 0
    raw = (0x69 | 1 << 8 | (R & 63) << 9 | 1 << 15 | (A & 63) << 16 | At << 22 | 2 << 24 | 1 << 34 | mask << 36
           | (A >> 6) << 58 | (R >> 6) << 60)
    b = raw.to_bytes(8, 'little')
    d = subprocess.run([ASM], input='?!' + b.hex() + '\n', capture_output=True, text=True).stdout
    assert d.startswith('threadgroup_load\t8'), d
    return b


def program(pattern, bcast=False):
    # r0l (the exec-stack depth) is not zero at entry under agx_run_hw: without this the push/pop fault (Bad address)
    head = e('iadd r0l, 0, 0') + mov(3, 0) + e('iadd r4, u5, 0') + e('and r31, r1, 31')
    for j in range(8):
        head += mov(5 + j, 0 if bcast else 128 * j) + (b'' if bcast else e(f'iadd r{5 + j}, r{5 + j}, r31, lsl 2'))
    for r in range(13, 31):
        head += mov(r, 0x3f800000 if r >= 29 else r)
    head += e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    body = b''
    nl = na = 0
    nc = {1: 0, 2: 0, 3: 0}
    last = 13
    for ch in pattern:
        if ch in 'LXB':
            d = 13 + (2 * nl if ch == 'X' else nl) % 8
            body += tgload(d, None if ch == 'B' else 5 + nl % 8, 0b11 if ch == 'X' else 1)
            last = d
            nl += 1
        elif ch == 'i':
            r = 21 + na % 8; body += e(f'iadd r{r}, r{r}, 1'); na += 1
        elif ch == 'f':
            r = 21 + na % 8; body += e(f'fmadd32 r{r}, r{r}, r29, r30'); na += 1
        elif ch == 'c':
            r = 21 + na % 8; body += e(f'iadd r{r}, r{r}, r{last}'); na += 1
        elif ch in 'pqr':
            n = 'pqr'.index(ch) + 1; r = 21 + nc[n] % n; nc[n] += 1
            body += e(f'iadd r{r}, r{r}, 1')
        elif ch == 'w':
            body += e('wait 0')
        else:
            raise ValueError(ch)
    body += e('iadd r3, 1, r3.sx') + e('while_icmp r0l, ult, r3, r4, 1')
    code = head + body + e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}') + e('pop_exec r0l, 1')
    return code + e('stop') + bytes(64)


def measure(key, code, groups, trips, d):
    wg = int(os.environ.get('WG', str(32 * groups)))
    f = os.path.join(d, 'k.bin'); open(f, 'wb').write(code)
    common = [f, '--grid', f'{32 * 32 * groups},1,1', '--wg', f'{wg},1,1', '--regs', '64', '--uniform', f'5={trips}']
    if key == 'hw':
        chk = subprocess.run([RUN] + common + ['--shared', '4096'], capture_output=True, text=True)
        if '"instructions"' not in chk.stdout:
            raise RuntimeError('emulator gate: ' + chk.stdout[-300:] + chk.stderr[-300:])
        args = [HW] + common + ['--shared', '4096', '--back-to-back', os.environ.get('B2B', '100'), '--timeout-ms', '10000']
        subprocess.run(args, capture_output=True, text=True)    # warm-up: the clock ramps during back-to-back runs
        h = subprocess.run(args, capture_output=True, text=True)
        m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
        if not m:
            raise RuntimeError('hw: ' + (h.stdout + h.stderr)[-300:])
        return int(m.group(1)) / 1000
    r = subprocess.run([DISP] + common + ['--tg-bytes', '4096', '--launch-profile', 'runtime'], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])['time_us']


def main():
    mode = sys.argv[1]
    pats = sys.argv[2:]
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    groups = [int(g) for g in os.environ.get('NG', '4 16 32').split()]
    T1, T2 = int(os.environ.get('T1', '1024')), int(os.environ.get('T2', '3072'))
    bcast = os.environ.get('ADDR', 'lane') == 'bcast'
    with tempfile.TemporaryDirectory() as d:
        for p in pats:
            code = program(p, bcast)
            for g in groups:
                rec = {'pattern': p, 'groups': g, 'addr': 'bcast' if bcast else 'lane', 'wg': int(os.environ.get('WG', str(32 * g)))}
                for key in ('hw', 'sim'):
                    if mode not in ('both', key):
                        continue
                    t = [measure(key, code, g, n, d) for n in (T1, T2)]
                    rec[key + '_us'] = t
                    rec[key + '_cyc'] = round((t[1] - t[0]) * MHZ / (T2 - T1), 2)
                if 'hw_cyc' in rec and 'sim_cyc' in rec:
                    rec['err'] = round(rec['sim_cyc'] / rec['hw_cyc'] - 1, 3)
                print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
