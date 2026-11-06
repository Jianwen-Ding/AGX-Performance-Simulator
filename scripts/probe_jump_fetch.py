#!/usr/bin/env python3
"""probe_jump_fetch.py: instruction fetch of branchy code visited once per command. One SIMD group runs N one-line
blocks (7 x iadd + jmp_exec_any + wait filler = 64 bytes), each visited once:
  straight  8 x iadd per line, no jumps (the measured streaming case)
  seq       each block jumps to the next line (taken jump, sequential lines)
  rand      each block jumps to a random other line of the N (taken jump, non-sequential lines)
  rand4     as rand, but each block is 4 lines (3 lines of iadd, then the jump): a jump every 256 bytes
Two sizes give ns per block (slope), hardware (agx_run_hw back to back) vs agx_dispatch (zluda profile).

Safety: every jump target is a block start inside the program (checked here, and agx_run must execute exactly the
expected instruction count before the GPU sees it), and 16 KB of `stop` follow the last block, so neither a jump nor
sequential prefetch past the end can leave the code object (agx_run_hw maps code + 64 bytes, rounded to 16 KB pages).
Two GPU faults (21:28 / 21:29, 2026-11-06) fell inside the first version's runs, which had no padding, while Blender ran on
the GPU: the cause is not established. Run it only on an idle GPU, after `gpu_safety.py status`.
2026-11-06 hardware (M2 Max, lower of two clock states): straight 70, seq 324, rand 235, rand4 451 ns per block."""
import json, os, random, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HW = os.environ.get('AGX_RUN_HW', '/scratch/zluda-air-target/release/agx_run_hw')
DISP = os.environ.get('AGX_DISPATCH', '/scratch/agxsim-target/release/agx_dispatch')
RUN = os.environ.get('AGX_RUN', '/scratch/agxsim-target/release/agx_run')
ASM = os.environ.get('AGX_ASM', '/scratch/agxsim-target/release/agx_asm')
_c = {}
def e(t):
    if t not in _c: _c[t] = bytes.fromhex(subprocess.run([ASM], input=t + '\n', capture_output=True, text=True, check=True).stdout.split()[0])
    return _c[t]
def iadds(k, i0=0): return b''.join(e(f'iadd r{5 + (i0 + j) % 4}, r{5 + (i0 + j) % 4}, 1') for j in range(k))
def program(kind, n, seed=1):
    pad = e('stop') * (16384 // len(e('stop')))
    if kind == 'straight':
        return iadds(8) * n + e('stop') + pad, n
    L = 4 if kind == 'rand4' else 1
    size = 64 * L
    order = list(range(n))
    if kind.startswith('rand'):
        rest = order[1:]; random.Random(seed).shuffle(rest); order = [0] + rest
    nxt = {order[i]: order[i + 1] for i in range(n - 1)}
    code = bytearray()
    for b in range(n):
        start = b * size
        blk = iadds(7 + 8 * (L - 1))
        jpc = start + len(blk)
        if b in nxt:
            assert 0 <= nxt[b] < n and nxt[b] * size != jpc
            blk += e(f'jmp_exec_any pc+{nxt[b] * size - jpc}' if nxt[b] * size > jpc else f'jmp_exec_any pc-{jpc - nxt[b] * size}')
        else:
            blk += e('stop')
        blk += e('wait 0') * ((size - len(blk)) // 2)
        assert len(blk) == size, (len(blk), size)
        code += blk
    return bytes(code) + pad, n
mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
if mode in ('both', 'hw'):
    import gpu_safety; gpu_safety.require_budget(1)
kinds = os.environ.get('KINDS', 'straight seq rand rand4').split()
N1, N2 = int(os.environ.get('N1', '1024')), int(os.environ.get('N2', '8192'))
with tempfile.TemporaryDirectory() as d:
    for kind in kinds:
        rec = {'kind': kind}
        for key in ('hw', 'sim'):
            if mode not in ('both', key): continue
            t = []
            for n in (N1, N2):
                code, _ = program(kind, n)
                f = os.path.join(d, f'{kind}{n}.bin'); open(f, 'wb').write(code)
                common = [f, '--grid', '32,1,1', '--wg', '32,1,1', '--regs', '16']
                chk = subprocess.run([RUN] + common, capture_output=True, text=True)
                if '"instructions"' not in chk.stdout: rec['error'] = chk.stdout[-200:] + chk.stderr[-200:]; break
                rec['instructions_' + str(n)] = json.loads(chk.stdout)['instructions']
                want = 8 * n + 1 if kind == 'straight' else (8 + 8 * 3 * (kind == 'rand4')) * n
                if rec['instructions_' + str(n)] != want: rec['error'] = f'emulator ran {rec["instructions_" + str(n)]} instructions, expected {want}'; break
                if key == 'hw':
                    # a warm-up run first (the clock ramps up during back-to-back runs), then the measured one
                    subprocess.run([HW] + common + ['--back-to-back', '200', '--timeout-ms', '10000'], capture_output=True, text=True)
                    h = subprocess.run([HW] + common + ['--back-to-back', '200', '--timeout-ms', '10000'], capture_output=True, text=True)
                    m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout); t.append(int(m.group(1)) / 1000 if m else None)
                    if not m: rec['hw_err'] = (h.stdout + h.stderr)[-200:]; break
                else:
                    r = subprocess.run([DISP] + common + ['--launch-profile', 'zluda'], capture_output=True, text=True)
                    t.append(json.loads(r.stdout)['time_us'])
            if len(t) == 2 and None not in t:
                rec[key + '_us'] = t; rec[key + '_ns_per_block'] = round((t[1] - t[0]) * 1000 / (N2 - N1), 2)
        print(json.dumps(rec), flush=True)
