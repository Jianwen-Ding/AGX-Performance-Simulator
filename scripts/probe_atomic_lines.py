#!/usr/bin/env python3
"""probe_atomic_lines.py: how do device atomics (add, no return) whose lanes partly collide cost, GPU-wide?

Blender Cycles' path-state kernels update small per-kernel / per-shader counter arrays (one or two 64-byte lines) with
a per-lane index: lanes collide on a few words of the same line. The 2026-10-16 contention probe measured one hot
word (0.86 ns per lane-operation GPU-wide) and distinct words; a hashed 32-counter workload ran almost as slow as one
counter. This probe separates the cases.

Each lane's word = (lane & MASK) * SPACING (+ the workgroup's id * 64 words with PERWG=1: every workgroup its own
lines); T trips of `atomic add` (uniform base u4_u5, register index) per thread. MASK=0: one word; MASK=31 SPACING=1:
32 distinct words in 2 lines; MASK=3 SPACING=1: 4 words of one line, 8 lanes each; MASK=3 SPACING=16: 4 lines, 8
lanes each.

usage: probe_atomic_lines.py [hw|sim|both]; env PAT="0:1 31:1 3:1 3:16" (MASK:SPACING) T=500 NWG=38 G=1 PERWG=0
B2B=200 REGS=32 FADD=0 (1: float add). Prints one line per pattern: hw / sim us and ns per lane-operation.
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tg_alu as P
import run_drm_atomic as DA

BUF = 0x1100000000


def build(mask, spacing, perwg):
    code = P.e('iadd r0l, 0, 0')
    code += P.e('get_sr r11, sr48', 'get_sr r11, sr48 (thread_position_in_threadgroup.x)')
    code += P.e('and r11, r11, 31')
    code += P.e(f'and r21, r11, {mask}')
    if spacing > 1:
        code += P.e(f'imadd r21, r21, {spacing}, 0')
    if perwg:
        code += P.e('get_sr r15, sr0', 'get_sr r15, sr0 (threadgroup_position_in_grid.x)')
        code += P.e('imadd r21, r15, 64, r21')
    fadd = os.environ.get('FADD') == '1'
    code += P.mov(12, 0x3f800000 if fadd else 1) + P.mov(7, 0) + P.e('iadd r8, u6, 0')
    code += P.e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    # FADD=1: float32 add (op 1 `sub` with u5 = 3, bit 45: playground 2026-10-17-atomic-variants; Cycles' film
    # accumulation, ZLUDA's atom.add.f32)
    raw = DA.mesa_atomic(1, 0, 8, 2 * 21, 2 * 12, 0) & ~(3 << 45) | 3 << 45 if fadd else DA.mesa_atomic(0, 0, 8, 2 * 21, 2 * 12, 0)
    body = raw.to_bytes(8, 'little')
    body += P.e('iadd r7, 1, r7.sx') + P.e('while_icmp r0l, ult, r7, r8, 1')
    code += body + P.e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}') + P.e('pop_exec r0l, 1')
    return code + P.e('stop') + bytes(64)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    T = int(os.environ.get('T', '500'))
    nwg, g = int(os.environ.get('NWG', '38')), int(os.environ.get('G', '1'))
    perwg = os.environ.get('PERWG') == '1'
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    with tempfile.TemporaryDirectory() as d:
        for pat in os.environ.get('PAT', '0:1 31:1 3:1 3:16').split():
            mask, spacing = (int(x) for x in pat.split(':'))
            f = os.path.join(d, f'k{mask}_{spacing}.bin'); open(f, 'wb').write(build(mask, spacing, perwg))
            a = [f, '--grid', f'{nwg * 32 * g},1,1', '--wg', f'{32 * g},1,1', '--regs', os.environ.get('REGS', '32'),
                 '--uniform', f'4={BUF & 0xffffffff}', '--uniform', f'5={BUF >> 32}', '--uniform', f'6={T}',
                 '--alloc', f'{BUF:#x}={max(65536, (nwg + 1) * 256 + 4096)}']
            chk = subprocess.run([P.RUN] + a + ['--dump', f'{BUF:#x}=256:{d}/out.bin'], capture_output=True, text=True)
            if '"instructions"' not in chk.stdout:
                raise RuntimeError('emulator gate: ' + chk.stdout[-300:] + chk.stderr[-300:])
            ops = nwg * 32 * g * T
            out = [f'{"FADD " if os.environ.get("FADD") == "1" else ""}PAT {mask}:{spacing} PERWG {int(perwg)} NWG {nwg} G {g} T {T}']
            if mode in ('hw', 'both'):
                h = subprocess.run([P.HW] + a + ['--back-to-back', os.environ.get('B2B', '200'), '--timeout-ms', '10000'],
                                   capture_output=True, text=True)
                m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
                out.append(f'hw {int(m.group(1)) / 1000:9.1f} ({int(m.group(1)) / ops:.3f} ns/op)' if m else 'hw FAILED ' + (h.stdout + h.stderr)[-200:])
            if mode in ('sim', 'both'):
                r = subprocess.run([P.DISP] + a + ['--launch-profile', 'runtime'], capture_output=True, text=True)
                us = json.loads(r.stdout.strip().splitlines()[-1])['time_us']
                out.append(f'sim {us:9.1f} ({us * 1000 / ops:.3f} ns/op)')
            print('  '.join(out), flush=True)


if __name__ == '__main__':
    main()
