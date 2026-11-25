#!/usr/bin/env python3
"""probe_jump_distance.py: cost of a taken jump to an uncached line by jump distance (follow-up of probe_jump_fetch.py,
whose seq case (jump to the next line, 324 ns) was slower than a random target (235 ns), unexplained). Same blocks and
safety as probe_jump_fetch.py (7 x iadd + jmp_exec_any + wait filler = 64 bytes; every target a block start, the
emulator must run exactly 8N instructions; 16 KB of stop after the code). Kinds: straight, fwdK (K interleaved chains
of stride K lines), skipD (single pass, each block D lines after the last), twoG_o (pass 1 lines iG, pass 2 lines iG+o), rgapA_B / bgapA_B (single pass, random gaps in [A, B] lines, forward / descending), bskipD, winW_G_o (pass pairs over windows of W), rsubG (random order over every G-th line), back (each block jumps to the line before), randS (random order, seed S)."""
import json, os, random, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HW = os.environ.get('AGX_RUN_HW', '/scratch/ptx-frontend-air-target/release/agx_run_hw')
DISP = os.environ.get('AGX_DISPATCH', '/scratch/agxsim-target/release/agx_dispatch')
RUN = os.environ.get('AGX_RUN', '/scratch/agxsim-target/release/agx_run')
ASM = os.environ.get('AGX_ASM', '/scratch/agxsim-target/release/agx_asm')
_c = {}
def e(t):
    if t not in _c: _c[t] = bytes.fromhex(subprocess.run([ASM], input=t + '\n', capture_output=True, text=True, check=True).stdout.split()[0])
    return _c[t]
def iadds(k, i0=0): return b''.join(e(f'iadd r{5 + (i0 + j) % 4}, r{5 + (i0 + j) % 4}, 1') for j in range(k))
def filler(nbytes):
    """the never-executed lines between blocks: FILL=stop (default), wait (wait 0), iadd (iadd r9, r9, 1 lines)"""
    f = os.environ.get('FILL', 'stop')
    unit = {'stop': e('stop'), 'wait': e('wait 0'), 'iadd': e('iadd r9, r9, 1')}[f]
    assert nbytes % len(unit) == 0 or f == 'iadd'
    out = unit * (nbytes // len(unit)); return out + e('wait 0') * ((nbytes - len(out)) // 2)
def gen_visits(visits, lines, size, pad):
    """code of `lines` lines; the listed lines are blocks visited in that order, the others filler"""
    assert len(set(visits)) == len(visits) and visits[0] == 0
    nxt = {visits[i]: visits[i + 1] for i in range(len(visits) - 1)}; code = bytearray()
    for ln in range(lines):
        if ln in nxt or ln == visits[-1]:
            blk = iadds(7); jpc = ln * size + len(blk)
            if ln in nxt:
                t = nxt[ln] * size; assert t != jpc and 0 <= nxt[ln] < lines
                blk += e(f'jmp_exec_any pc+{t - jpc}' if t > jpc else f'jmp_exec_any pc-{jpc - t}')
            else:
                blk += e('stop')
            blk += e('wait 0') * ((size - len(blk)) // 2)
        else:
            blk = filler(size)
        assert len(blk) == size
        code += blk
    return bytes(code) + pad
def order_for(kind, n):
    """block visiting order: fwdK = K interleaved chains of stride K lines (fwd1 = the old seq), back = each block jumps
    to the line before it, randS = random order with seed S"""
    if kind.startswith('fwd'):
        k = int(kind[3:]); return [b for c in range(k) for b in range(c, n, k)]
    if kind == 'back': return [0] + list(range(n - 1, 0, -1))
    if kind.startswith('rand'):
        rest = list(range(1, n)); random.Random(int(kind[4:] or 1)).shuffle(rest); return [0] + rest
    raise SystemExit('unknown kind ' + kind)
def program(kind, n):
    pad = e('stop') * (16384 // len(e('stop')))
    if kind == 'straight':
        return iadds(8) * n + e('stop') + pad, n
    size = 64
    if kind.startswith('skip'):
        # single pass: blocks at lines 0, D, 2D, ...; each jumps D lines forward; the D-1 lines between hold stop and
        # are never executed (a pure jump distance, no line revisited or adjacent to an earlier one for D >= 2)
        d = int(kind[4:]); code = bytearray()
        for b in range(n):
            blk = iadds(7); jpc = len(blk)
            blk += e(f'jmp_exec_any pc+{d * size - jpc}') if b < n - 1 else e('stop')
            blk += e('wait 0') * ((size - len(blk)) // 2)
            assert len(blk) == size
            code += blk + filler((d - 1) * size)
        return bytes(code) + pad, n
    if kind.startswith('two'):
        # two passes: lines 0, G, 2G, ... (n/2 of them), then o, G+o, 2G+o, ...: the second pass's cost against a
        # single pass (skipG) shows whether line o came in with line 0 (the fill granularity) and how long it stays
        g, o = map(int, kind[3:].split('_')); assert 0 < o < g and n % 2 == 0
        h = n // 2; visits = [i * g for i in range(h)] + [i * g + o for i in range(h)]
        lines = h * g; nxt = {visits[i]: visits[i + 1] for i in range(n - 1)}; code = bytearray()
        for ln in range(lines):
            if ln in nxt or ln == visits[-1]:
                blk = iadds(7); jpc = ln * size + len(blk)
                if ln in nxt:
                    t = nxt[ln] * size; assert t != jpc and 0 <= nxt[ln] < lines
                    blk += e(f'jmp_exec_any pc+{t - jpc}' if t > jpc else f'jmp_exec_any pc-{jpc - t}')
                else:
                    blk += e('stop')
                blk += e('wait 0') * ((size - len(blk)) // 2)
            else:
                blk = filler(size)
            assert len(blk) == size
            code += blk
        return bytes(code) + pad, n
    if kind.startswith('win'):
        # winW_G_o: windows of W lines iG; each window visits lines iG, then iG+o (reuse distance W)
        w, g, o = map(int, kind[3:].split('_')); assert n % (2 * w) == 0 and 0 < o < g
        visits = [i * g + k for b in range(0, n // 2, w) for k in (0, o) for i in range(b, b + w)]
        return gen_visits(visits, (n // 2) * g, size, pad), n
    if kind.startswith('rsub'):
        # rsubG: random order over lines 0, G, 2G, ... only (the others filler): which neighbourhood the random case reuses
        g = int(kind[4:]); rest = list(range(1, n)); random.Random(1).shuffle(rest)
        return gen_visits([0] + [b * g for b in rest], n * g, size, pad), n
    if kind[:4] in ('rgap', 'bgap', 'bski'):
        # single pass with gaps: rgapA_B forward gaps uniform in [A, B] lines, bgapA_B the same descending (block 0
        # jumps to the top block first), bskipD descending with constant gap D
        if kind.startswith('bskip'): a = b_ = int(kind[5:])
        else: a, b_ = map(int, kind[4:].split('_'))
        rng = random.Random(7); pos = [0]
        for _ in range(n - 1): pos.append(pos[-1] + rng.randint(a, b_))
        lines = pos[-1] + 1
        visits = pos if kind.startswith('rgap') else [0] + pos[:0:-1]
        return gen_visits(visits, lines, size, pad), n
    order = order_for(kind, n)
    assert sorted(order) == list(range(n)) and order[0] == 0
    nxt = {order[i]: order[i + 1] for i in range(n - 1)}
    code = bytearray()
    for b in range(n):
        start = b * size
        blk = iadds(7)
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
kinds = os.environ.get('KINDS', 'straight fwd1 fwd2 fwd4 fwd16 back rand1 rand2').split()
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
                want = 8 * n + 1 if kind == 'straight' else 8 * n
                if rec['instructions_' + str(n)] != want: rec['error'] = f'emulator ran {rec["instructions_" + str(n)]} instructions, expected {want}'; break
                if key == 'hw':
                    # a warm-up run first (the clock ramps up during back-to-back runs), then the measured one
                    subprocess.run([HW] + common + ['--back-to-back', '200', '--timeout-ms', '10000'], capture_output=True, text=True)
                    h = subprocess.run([HW] + common + ['--back-to-back', '200', '--timeout-ms', '10000'], capture_output=True, text=True)
                    m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout); t.append(int(m.group(1)) / 1000 if m else None)
                    if not m: rec['hw_err'] = (h.stdout + h.stderr)[-200:]; break
                else:
                    r = subprocess.run([DISP] + common + ['--launch-profile', 'runtime'], capture_output=True, text=True)
                    t.append(json.loads(r.stdout)['time_us'])
            if len(t) == 2 and None not in t:
                rec[key + '_us'] = t; rec[key + '_ns_per_block'] = round((t[1] - t[0]) * 1000 / (N2 - N1), 2)
        print(json.dumps(rec), flush=True)
