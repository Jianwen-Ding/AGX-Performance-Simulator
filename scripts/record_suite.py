#!/usr/bin/env python3
"""Record the hardware test suite: exact AGX behaviour captured on the M2 Max for off-device development.

Every case is one dispatch through tests/drm_compute.c (direct DRM). It stores, in testsuite/<category>/<name>.json.gz:
  code      the main shader exactly as submitted (hex)
  launch    grid, workgroup size, register halves, threadgroup bytes
  layout    out buffer (8 guards 0xcdcdcdcd | rounded count | 8 guards | 16 words 0xa5a50000+i), the table, their VAs,
            and the uniforms the harness sets: u4 = 0, u5 = count, u6_u7 = out VA, u8_u9 = table VA
  input     the table before the dispatch (little-endian u32 words, base64)
  expect    the out buffer and the table after one dispatch
  runs      2: both runs must agree exactly (otherwise the case is marked nondeterministic)
The simulator reproduces a case by mapping the two buffers at those VAs, running every SIMD group of the grid and
comparing out and table word for word.

Categories: alu (one instruction per case, 1,024 lanes of random / edge inputs), alu_masked (cross-lane
instructions with a quarter of the lanes inactive). More categories are added by later generators in this file.

Usage: record_suite.py [--only CATEGORY] [--limit N] [--out testsuite]
"""
import argparse, base64, gzip, hashlib, json, random, struct, subprocess, sys, time
from pathlib import Path
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm
import suite_templates as ST

e, mov = agx_loop.encode, B.mov
ROOT = Path(__file__).resolve().parents[1]
HARNESS_DATA_VA = None   # read from the harness output


def b64(words):
    return base64.b64encode(struct.pack(f'<{len(words)}I', *words)).decode()


def run(R, code, count, local_x, table, regs=64, shared=0, dims=None):
    """One dispatch; returns (out words, table words after, fw_ns, table_va). dims = (gx, gy, gz, lx, ly, lz) for a
    multi-dimensional launch (count still sets the out buffer and u5). `code` must be a whole program
    (run_drm_compute.build_main(body)): a bare body has no prologue and no final `stop`, and executing past its end hung
    the GPU on 2026-10-16."""
    if not bytes(code).endswith(b'\x88\x00'):           # callers pass the unpadded program; run() pads it below
        raise ValueError('refusing to submit: the program does not end with stop (0x8800); wrap the body with '
                         'run_drm_compute.build_main()')
    (R.out/'c.bin').write_bytes(code+bytes(16))
    (R.out/'t.bin').write_bytes(struct.pack(f'<{len(table)}I', *table))
    env = {'PATH': '/usr/bin', 'AGX_LOCAL_X': str(local_x), 'AGX_TABLE_OUT': str(R.out/'t.out')}
    if dims:
        gx, gy, gz, lx, ly, lz = dims
        env.update({'AGX_GLOBAL_X': str(gx), 'AGX_GLOBAL_Y': str(gy), 'AGX_GLOBAL_Z': str(gz), 'AGX_LOCAL_X': str(lx),
                    'AGX_LOCAL_Y': str(ly), 'AGX_LOCAL_Z': str(lz)})
    if shared:
        from run_drm_tg_layout import shared_word
        env['AGX_SHARED_WORD'] = str(shared_word(shared))
    r = subprocess.run([str(R.exe), '/dev/dri/renderD128', str(R.out/'c.bin'), str(count), str(R.out/'rb'), '1', str(regs),
                        str(R.out/'t.bin')], capture_output=True, timeout=60, env=env)
    lines = [json.loads(l) for l in r.stdout.decode().splitlines() if l.startswith('{')]
    smp = [x for x in lines if 'rep' in x]
    if r.returncode or len(smp) != 1 or not all(smp[0]['fw_ts']):
        if smp:   # only a submitted dispatch can time out or fault; a host-side rejection does not touch the GPU
            gpu_safety.record('record_suite fault')
        raise RuntimeError('dispatch failed: '+r.stdout.decode()[-300:])
    va = next(x['table']['va'] for x in lines if 'table' in x)
    rounded = (count+31)//32*32
    words = 8+rounded+8+16
    out = list(struct.unpack(f'<{words}I', (R.out/'rb').read_bytes()[:4*words]))
    tab = list(struct.unpack(f'<{len(table)}I', (R.out/'t.out').read_bytes()[:4*len(table)]))
    return out, tab, smp[0]['fw_ts'][1]-smp[0]['fw_ts'][0], va


F32_SPECIAL = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00000,
               0x00000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0x4b000000, 0x4effffff, 0xcf000000, 0x40490fdb, 0x3fffffff,
               0x4f800000, 0x3eaaaaab, 0xc0000000, 0x41200000]
F16_SPECIAL = [0x0000, 0x8000, 0x3c00, 0xbc00, 0x3800, 0x7c00, 0xfc00, 0x7e00, 0x7d00, 0x0001, 0x03ff, 0x0400, 0x7bff,
               0x4248, 0x5640, 0xc000]
INT_SPECIAL = [0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 0xff, 0x100, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0x80000001,
               0xffffffff, 0xfffffffe, 0x55555555, 0xaaaaaaaa, 0x12345678, 0x0000ffff, 0xffff0000]


def operand(rng):
    x = rng.random()
    if x < 0.2: return rng.choice(F32_SPECIAL)
    if x < 0.35: return rng.choice(F16_SPECIAL) | rng.choice(F16_SPECIAL) << 16
    if x < 0.55: return rng.choice(INT_SPECIAL)
    if x < 0.7: return struct.unpack('<I', struct.pack('<f', rng.uniform(-100, 100)))[0]
    if x < 0.8: return rng.randrange(0, 64) if rng.random() < 0.5 else (-rng.randrange(1, 64)) & 0xffffffff
    return rng.getrandbits(32)


def alu_body(instr, n, masked=False):
    """r5 = 8*gid; r6..r13 = table[8*gid .. +8]; r14 = r15 = 0xdeadbeef; instr; table[8n + 2*gid + k] = r14, r15;
    r2 = r14. masked: the instruction runs only in lanes with (gid & 3) != 3."""
    b = mov(27, 8)+e('imadd r5, r1, r27, 0')+e('iadd r26, r5, 4')
    b += e('device_load 0, i32, xyzw, r6_r7_r8_r9, u8_u9, r5, unsigned')+e('device_load 0, i32, xyzw, r10_r11_r12_r13, u8_u9, r26, unsigned')
    b += e('wait 0')+mov(14, 0xdeadbeef)+mov(15, 0xdeadbeef)
    if masked:
        b += e('and r24, r1, 3')+mov(25, 3)+e('if_icmp r0l, ult, r24, r25, 1')+instr+e('pop_exec r0l, 1')
    else:
        b += instr
    b += mov(27, 8*n)+mov(25, 2)+e('imadd r26, r1, r25, r27')+agx_loop.store(14, 8, 26)+e('iadd r26, r26, 1')
    b += agx_loop.store(15, 8, 26)+e('wait 0')+e('iadd r2, r14, 0')
    return b


CROSS = ('simd_', 'quad_', 'icmp_ballot', 'fcmp_ballot', 'icmp_quad', 'fcmp_quad')


def gen_alu(masked):
    n, wg = 1024, 256
    for t in ST.templates():
        if masked and not t.startswith(CROSS): continue
        rng = random.Random(hashlib.sha256(t.encode()).digest())
        table = [operand(rng) for _ in range(8*n)]+[0]*(2*n)
        name = t.replace(' ', '_').replace(',', '').replace('.', '-')
        yield ('alu_masked' if masked else 'alu', name, t, alu_body(ST.encode(t), n, masked), n, wg, 64, 0, table)


def store_like(load_text):
    """device_store with the format / mask / shift / signedness / base / index of a load form (the control store's
    encoding with those fields copied; exact decode round trip)."""
    _, g = agx_loop._tools()
    lb = e(load_text); ln = g.opcode_to_number(lb); ld = g.get_instruction_descriptor(ln)
    lf = dict(ld.decode_fields(ln & ((1 << 8*ld.decode_size(ln))-1)))
    ref = bytes.fromhex('45112c0e00c01200'); n = g.opcode_to_number(ref); d = g.get_instruction_descriptor(n)
    f = dict(d.decode_fields(n))
    for k in ('F', 'Fx', 'mask', 's', 'Ou', 'At', 'A', 'O', 'Ot', 'R', 'Rt', 'g'):
        if k in lf and k in f: f[k] = lf[k]
    x = d.encode_fields(f); m = g.get_instruction_descriptor(x)
    if m is not d or dict(m.decode_fields(x)) != f: raise ValueError('store encoding mismatch')
    return x.to_bytes(8, 'little')


def regs_for(fmt, mask):
    n = len(mask)
    if fmt == 'i32':
        return '_'.join(f'r{14+k}' for k in range(n))
    return '_'.join(f'r{14+k//2}{"lh"[k % 2]}' for k in range(n))


LOAD_FORMS = []
for fmt in ('i32', 'i16', 'i8', 'u8norm'):
    for mask in ('x', 'xy', 'xyz', 'xyzw', 'y', 'yw', 'xz', 'zw'):
        LOAD_FORMS.append((fmt, mask, 'unsigned', '', 'u8_u9'))
for fmt in ('i32', 'i16', 'i8'):
    for mode, sh in (('signed', ''), ('unsigned', ', lsl 1'), ('unsigned', ', lsl 2'), ('signed', ', lsl 2')):
        LOAD_FORMS.append((fmt, 'xy', mode, sh, 'u8_u9'))
    LOAD_FORMS.append((fmt, 'x', 'unsigned', '', 'r24_r25'))
    LOAD_FORMS.append((fmt, 'xyzw', 'signed', '', 'r24_r25'))


def mem_prologue(n):
    """r24_r25 = table + 8 KiB (u8_u9 copied, validated int64 form); r14..r17 = 0xdeadbeef."""
    b = e('iadd r24, u8, 0')+e('iadd r25, u9, 0')+mov(26, 2048)+e('iadd r24_r25, r24_r25, r26, lsl 2')
    return b+b''.join(mov(r, 0xdeadbeef) for r in range(14, 18))


ITEM = {'i32': 4, 'i16': 2, 'i8': 1, 'u8norm': 1}


def load_index(mode, base):
    """r5: unsigned (gid*37 & 255) + 3; signed (gid & 255) - 128 (around table + 8 KiB; + 256 with the u8_u9 base)."""
    if mode == 'unsigned':
        return mov(27, 37)+e('imadd r5, r1, r27, 0')+mov(27, 255)+e('and r5, r5, r27')+e('iadd r5, r5, 3'), (3, 258)
    lo = -128 if base == 'r24_r25' else 128
    return mov(27, 255)+e('and r5, r1, r27')+mov(27, lo & 0xffffffff)+e('iadd r5, r5, r27'), (lo, lo+255)


def check_bounds(fmt, mask, sh, base, rng_idx, table_words):
    shift = int(sh.split()[-1]) if sh else 0
    off0 = 8192 if base == 'r24_r25' else 0
    comps = max('xyzw'.index(c) for c in mask)+1
    lo = off0+(rng_idx[0] << shift)*ITEM[fmt]
    hi = off0+((rng_idx[1] << shift)+comps)*ITEM[fmt]
    if lo < 0 or hi > 4*table_words: raise ValueError(f'out of table: {fmt} {mask} {sh} {base}: bytes {lo}..{hi}')


def gen_mem_load():
    n, wg = 1024, 256
    for fmt, mask, mode, sh, base in LOAD_FORMS:
        t = f'device_load 0, {fmt}, {mask}, {regs_for(fmt, mask)}, {base}, r5, {mode}{sh}'
        rng = random.Random(t)
        table = [rng.getrandbits(32) for _ in range(8192)]+[0]*(4*n)
        idx, rng_idx = load_index(mode, base)
        check_bounds(fmt, mask, sh, base, rng_idx, 8192)
        b = mem_prologue(n)+idx+e(t)+e('wait 0')
        b += mov(27, 8192)+mov(26, 4)+e('imadd r26, r1, r26, r27')
        for k in range(4):
            b += agx_loop.store(14+k, 8, 26)+e('iadd r26, r26, 1')
        b += e('wait 0')+e('iadd r2, r14, 0')
        yield ('mem_load', t.replace(' ', '_').replace(',', ''), t, b, n, wg, 64, 0, table)


def gen_mem_store():
    n, wg = 1024, 256
    for fmt, mask, mode, sh, base in LOAD_FORMS:
        if fmt == 'u8norm': continue
        t = f'device_load 0, {fmt}, {mask}, {regs_for(fmt, mask)}, {base}, r5, {mode}{sh}'
        rng = random.Random('store'+t)
        table = [rng.getrandbits(32) for _ in range(12288)]
        b = mem_prologue(n)
        # distinct per-lane targets: element index 2*gid >> shift (+ signed: - 512 around the pair base = table + 8 KiB)
        shift = int(sh.split()[-1]) if sh else 0
        lo = -512 if mode == 'signed' and base == 'r24_r25' else 0
        # element index m*(gid + (gid >> 5)): lanes of one SIMD group overlap (component order), groups do not
        m = 2 if shift == 0 else 1
        b += mov(27, 5)+e('asr r28, r1, r27')+mov(27, m)+e('imadd r5, r1, r27, 0')+e('imadd r5, r28, r27, r5')
        if lo: b += mov(27, lo & 0xffffffff)+e('iadd r5, r5, r27')
        check_bounds(fmt, mask, sh, base, (lo, lo+m*(n-1+(n-1)//32)), 12288)
        for k in range(4):   # values: (gid << 8) | 0x11*(k+1) in each register
            b += mov(26, 256)+e(f'imadd r{14+k}, r1, r26, 0')+mov(26, 0x11*(k+1))+e(f'or r{14+k}, r{14+k}, r26')
        b += store_like(t)+e('wait 0')+e('iadd r2, r1, 0')
        yield ('mem_store', 'store_'+t.split(' ', 1)[1].replace(' ', '_').replace(',', ''), 'device_store with the fields of: '+t, b, n, wg, 64, 0, table)


def gen_tg():
    """Each lane writes its value to its slot, barrier, reads the slot of lane (local+1) % wg in every format."""
    from run_drm_tg_layout import tg
    n, wg = 1024, 256
    for fmt in ('i32', 'i16', 'i8'):
        size = {'i32': 4, 'i16': 2, 'i8': 1}[fmt]
        for mask in (1, 3, 15):
            if fmt != 'i32' and mask != 1: continue
            for index_mode in ('imm0', 'reg'):
                name = f'tg_{fmt}_mask{mask}_{index_mode}'
                rng = random.Random(name)
                table = [rng.getrandbits(32) for _ in range(4*n)]+[0]*(4*n)   # inputs [0, 4n), outputs [4n, 8n)
                b = B.get_sr(51, 23)+mov(27, 4*size)+e('imadd r24, r23, r27, 0')         # byte base 16*size*local... (vec4 slots)
                b += mov(27, 4)+e('imadd r5, r1, r27, 0')+e('device_load 0, i32, xyzw, r14_r15_r16_r17, u8_u9, r5, unsigned')+e('wait 0')
                idx = ('r', 22) if index_mode == 'reg' else 0
                if index_mode == 'reg': b += mov(22, 0)
                b += tg(False, fmt, 14, base=24, index=idx, mask=mask)+e('wait 0')+bytes.fromhex('6800')
                b += mov(27, 1)+e('iadd r25, r23, r27')+mov(27, wg-1)+e('and r25, r25, r27')+mov(27, 4*size)+e('imadd r25, r25, r27, 0')
                b += b''.join(mov(r, 0xdeadbeef) for r in range(14, 18))
                b += tg(True, fmt, 14, base=25, index=idx, mask=mask)+e('wait 0')
                b += mov(27, 4*n)+mov(26, 4)+e('imadd r26, r1, r26, r27')
                for k in range(4):
                    b += agx_loop.store(14+k, 8, 26)+e('iadd r26, r26, 1')
                b += e('wait 0')+e('iadd r2, r14, 0')+bytes.fromhex('6800')
                yield ('threadgroup', name, f'threadgroup store/load {fmt} mask {mask} index {index_mode}, neighbour read', b, n, wg, 64, 16*wg, table)


ATOMIC_OPS = ['add', 'sub', 'xchg', 'cmpxchg', 'umin', 'imin', 'umax', 'imax', 'and', 'or', 'xor']


def gen_atomic():
    import run_drm_atomic as DA
    n, wg = 1024, 256
    for op, name in enumerate(ATOMIC_OPS):
        for contended in (False, True):
            for ret in (False, True):
                case = f'atomic_{name}_{"shared4" if contended else "distinct"}_{"ret" if ret else "noret"}'
                rng = random.Random(case)
                b = mov(27, 3 if contended else 0xffffffff)+e('and r5, r1, r27')+mov(27, 2*n)+e('iadd r5, r5, r27')
                b += mov(27, 2)+e('imadd r6, r1, r27, 0')+e('device_load 0, i32, xy, r6_r7, u8_u9, r6, unsigned')+e('wait 0')
                b += mov(14, 0xdeadbeef)+DA.atomic(op, 14, 5, 6, ret)+e('wait 0')
                b += mov(27, 3*n)+e('iadd r26, r1, r27')+agx_loop.store(14, 8, 26)+e('wait 0')+e('iadd r2, r14, 0')
                # operands table[2*gid] (value / new), table[2*gid+1] (compare, cmpxchg); targets table[2n + (gid or gid&3)];
                # returned values table[3n + gid]
                pick = lambda: rng.choice([0, 1, 0x7fffffff, 0x80000000, 0xffffffff, rng.getrandbits(32)])
                tbl = [pick() for _ in range(2*n)]+[pick() for _ in range(n)]+[0]*n
                yield ('atomic', case, f'device atomic {name} (op {op}), {"4 shared words" if contended else "distinct words"}, '
                       f'{"returning the old value" if ret else "no return"}; operands table[2*gid], table[2*gid+1]', b, n, wg, 64, 0, tbl)


def gen_tgatomic():
    import run_drm_tg_atomic as TA
    from run_drm_tg_layout import tg
    n, wg = 1024, 256
    for op, name in enumerate(ATOMIC_OPS):
        for contended in (False, True):
            for ret in (False, True):
                case = f'tgatomic_{name}_{"shared4" if contended else "distinct"}_{"ret" if ret else "noret"}'
                rng = random.Random(case)
                tbl = [rng.getrandbits(32) for _ in range(3*n)]+[0]*(2*n)
                b = B.get_sr(51, 23)+mov(27, 3 if contended else 0xffffffff)+e('and r24, r23, r27')+mov(27, 16)+e('imadd r24, r24, r27, 0')
                # initial threadgroup words = table[gid] for the owning lanes (slot = local*16 bytes)
                b += mov(27, 16)+e('imadd r25, r23, r27, 0')+e('device_load 0, i32, x, r8, u8_u9, r1, unsigned')
                b += mov(27, 2)+e('imadd r6, r1, r27, 0')+mov(27, n)+e('iadd r6, r6, r27')+e('device_load 0, i32, xy, r6_r7, u8_u9, r6, unsigned')
                b += e('wait 0')+tg(False, 'i32', 8, base=25, index=0)+e('wait 0')+bytes.fromhex('6800')
                b += mov(14, 0xdeadbeef)+TA.local_atomic(op, 14, 24, 6, ret)+e('wait 0')+bytes.fromhex('6800')
                b += tg(True, 'i32', 15, base=25, index=0)+e('wait 0')
                b += mov(27, 3*n)+mov(26, 2)+e('imadd r26, r1, r26, r27')+agx_loop.store(14, 8, 26)+e('iadd r26, r26, 1')
                b += agx_loop.store(15, 8, 26)+e('wait 0')+e('iadd r2, r14, 0')
                yield ('tg_atomic', case, f'threadgroup atomic {name} (op {op}), {"4 shared words" if contended else "distinct words"}, '
                       f'{"ret" if ret else "no return"}; then every lane reads its own slot', b, n, wg, 64, 16*wg, tbl)


def jnone(n):
    return e(f'jmp_exec_none pc+{n}', f'jmp_exec_none pc+{n}')


def if_else(cond_if, cond_else, then, els, skip=True):
    """Mesa structured if/else: if_*; [jmp_exec_none over then]; then; else_*; [jmp_exec_none over else]; else; pop."""
    b = e(cond_if)+(jnone(6+len(then)) if skip else b'')+then
    if cond_else:
        b += e(cond_else)+(jnone(6+len(els)) if skip else b'')+els
    return b+e('pop_exec r0l, 1')


def counted_reg(body, trips_reg):
    """do-while loop with the trip count in a register (per lane); r3 counter."""
    head = mov(3, 0)+e(f'iadd r4, r{trips_reg}, 0')+e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    lp = body+e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
    return head+lp+e(f'jmp_exec_any pc-{len(lp)}', f'jmp_exec_any pc-{len(lp)}')+e('pop_exec r0l, 1')


def gen_control(gate_only=False):
    """Lane-dependent control flow; r6..r13 = table[8*gid ..] (ints 0..63 and floats), r2 = result."""
    n, wg = 1024, 256
    load = mov(27, 8)+e('imadd r5, r1, r27, 0')+e('iadd r26, r5, 4')+e('device_load 0, i32, xyzw, r6_r7_r8_r9, u8_u9, r5, unsigned')
    load += e('device_load 0, i32, xyzw, r10_r11_r12_r13, u8_u9, r26, unsigned')+e('wait 0')+mov(2, 1000)
    cases = []
    A = lambda k: e(f'iadd r2, r2, {k}')
    for cc in ('ult', 'ugt', 'seq', 'slt', 'sgt', 'ueq'):
        ce = {'ult': 'ugt', 'ugt': 'ult', 'seq': 'slt', 'slt': 'sgt', 'sgt': 'slt', 'ueq': 'ult'}[cc]
        cases.append((f'if_else_{cc}', f'if r6 {cc} r7 (+1) else-if r6 {ce} r7 (+2)',
                      if_else(f'if_icmp r0l, {cc}, r6, r7, 1', f'else_icmp r0l, {ce}, r6, r7, 1', A(1), A(2))))
    for cc in ('lt', 'gt', 'eq', 'ltn', 'gtn', 'gte', 'lte'):
        cases.append((f'if_else_f{cc}', f'if f(r10) {cc} f(r11) (+1) else-if f(r10) lt f(r11) (+2)',
                      if_else(f'if_fcmp r0l, {cc}, r10, r11, 1', 'else_fcmp r0l, lt, r10, r11, 1', A(1), A(2))))
    cases.append(('if_no_skip', 'if r6 ult r7 without jmp_exec_none', if_else('if_icmp r0l, ult, r6, r7, 1', 'else_icmp r0l, ugt, r6, r7, 1', A(1), A(2), False)))
    cases.append(('if_only', 'if r6 ult 32 (+5)', if_else('if_icmp r0l, ult, r6, r25, 1', None, A(5), b'')))
    inner = if_else('if_icmp r0l, ult, r7, r8, 1', 'else_icmp r0l, ugt, r7, r8, 1', A(10), A(20))
    cases.append(('nested2', 'if r6<r9 { if r7<r8 +10 else +20 } else +3', if_else('if_icmp r0l, ult, r6, r9, 1', 'else_icmp r0l, ugt, r6, r9, 1', inner+A(1), A(3))))
    inner3 = if_else('if_icmp r0l, ult, r8, r9, 1', None, A(100), b'')
    inner2 = if_else('if_icmp r0l, ult, r7, r9, 1', 'else_icmp r0l, ugt, r7, r9, 1', inner3+A(10), A(20))
    cases.append(('nested3', 'three nested ifs', if_else('if_icmp r0l, ult, r6, r9, 1', 'else_icmp r0l, ugt, r6, r9, 1', inner2, A(3))))
    cases.append(('uniform_true', 'if 1 < 2 (all lanes) ', if_else('if_icmp r0l, ult, r24, r25, 1', 'else_icmp r0l, ugt, r24, r25, 1', A(1), A(2))))
    cases.append(('uniform_false', 'if 2 < 1 (no lane): then-side skipped', if_else('if_icmp r0l, ult, r25, r24, 1', 'else_icmp r0l, ugt, r25, r24, 1', A(1), A(2))))
    trips = e('and r20, r6, r27')   # r20 = r6 & 31 (+1 below: do-while)
    cases.append(('loop_lane_trips', 'per-lane trip count (r6 & 31) + 1, r2 += r3 each trip', trips+counted_reg(e('iadd r2, r2, r3'), 20)))
    cases.append(('loop_if_inside', 'per-lane loop with if (r3 & 1) inside', trips+counted_reg(e('and r21, r3, r24')+if_else('if_icmp r0l, seq, r21, r24, 1', None, e('iadd r2, r2, r3'), b''), 20)))
    inner_loop = mov(16, 0)+e('and r21, r7, r28')+counted_reg_nested(e('iadd r2, r2, 1'))
    cases.append(('loop_nested', 'outer (r6&31)+1 trips, inner (r7&7)+1 trips via r16/r21', trips+counted_reg(inner_loop, 20)))
    cases.append(('loop_float_cond', 'while_fcmp: x += 0.5 while x < f(r10)', mov(22, 0)+loop_fcmp()))
    for name, desc, body in cases:
        rng = random.Random(name)
        table = []
        for _ in range(n):
            table += [rng.randrange(64) for _ in range(4)]+[struct.unpack('<I', struct.pack('<f', rng.uniform(-4, 4)))[0] for _ in range(2)]+[rng.randrange(64) for _ in range(2)]
        pre = mov(24, 1)+mov(25, 2)+mov(27, 31)+mov(28, 7)
        full = load+pre+body+mov(27, 8*n)+e('iadd r26, r1, r27')+agx_loop.store(2, 8, 26)+e('wait 0')
        if gate_only:
            yield (name, mov(2, 1000)+pre+body, table)
            continue
        yield ('control', name, desc, full, n, wg, 64, 0, table+[0]*n)


def counted_reg_nested(body):
    """inner do-while with its own counter r16 and bound r21 (+1)."""
    head = e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    lp = body+e('iadd r16, 1, r16.sx')+e('while_icmp r0l, ult, r16, r21, 1')
    return head+lp+e(f'jmp_exec_any pc-{len(lp)}', f'jmp_exec_any pc-{len(lp)}')+e('pop_exec r0l, 1')


def loop_fcmp():
    """x (r22, float) starts 0.0; do { x += 0.5; r2 += 1 } while (x < f(r10)), at most ~9 trips for r10 in [-4, 4]."""
    head = e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    lp = e('fadd32 r22, r22, 0.5')+e('iadd r2, r2, 1')+e('while_fcmp r0l, lt, r22, r10, 1')
    return head+lp+e(f'jmp_exec_any pc-{len(lp)}', f'jmp_exec_any pc-{len(lp)}')+e('pop_exec r0l, 1')


def control_gate():
    """Emulate every control case for lanes 0..31 (r6..r13 preset from the table, no memory instructions)."""
    for name, body, table in gen_control(gate_only=True):
        def init(st):
            for t in range(32):
                for k in range(8):
                    st.set_reg32(6+k, t, table[8*t+k])
        try:
            r = agx_loop.emulate(body, 32, lane_ids=True, num_registers=32, init=init)
        except NotImplementedError:
            # applegpu has no emulation for ltn/gtn/gte/lte; these cases share the structure of if_else_flt
            if not name.startswith('if_else_f'): raise
            continue
        if not all(r['exec']): raise RuntimeError(f'{name}: lanes left inactive')
        if r['counts'].get('jmp_exec_any', 0) > 200000: raise RuntimeError(f'{name}: runaway loop')


def f32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def gen_kernels():
    """Whole kernels with host oracles: yield (cat, name, desc, body, n, wg, regs, shared, table, oracle(out, table) -> bool)."""
    import math, reduce_probe as RP
    from run_drm_tg_layout import tg
    BAR = bytes.fromhex('6800')
    # 1. tree reductions (reduce_probe): out[8 + g] = workgroup sum
    for n, trips in ((2048, 2), (9728, 4), (38912, 1)):
        table = list(range(n*8*trips))
        want = RP.oracle_fast(n, trips)
        yield ('kernel', f'reduce_{n}_{trips}', f'per-workgroup tree reduction over {8*trips} loads per thread (reduce_probe)',
               RP.body(n, trips), n, 256, 64, 1024, table, lambda out, tab, w=want, n=n: out[8:8+n] == w)
    # 2. saxpy: y[i] = 2.5 * x[i] + y[i] (fused); x = table[i], y = table[n + i]
    n = 4096
    rng = random.Random('saxpy')
    xs = [f32(rng.uniform(-100, 100)) for _ in range(n)]; ys = [f32(rng.uniform(-100, 100)) for _ in range(n)]
    b = e('device_load 0, i32, x, r6, u8_u9, r1, unsigned')+mov(27, n)+e('iadd r5, r1, r27')+e('device_load 0, i32, x, r7, u8_u9, r5, unsigned')
    b += e('wait 0')+mov(9, f32(2.5))+e('fmadd32 r8, r6, r9, r7')+agx_loop.store(8, 8, 5)+e('wait 0')+e('iadd r2, r8, 0')
    fl = lambda w: struct.unpack('<f', struct.pack('<I', w))[0]
    want = [f32(math.fma(2.5, fl(x), fl(y))) for x, y in zip(xs, ys)]
    yield ('kernel', 'saxpy', 'y = 2.5 x + y (fmadd32) over 4096 floats', b, n, 256, 64, 0, xs+ys,
           lambda out, tab, w=want: tab[n:2*n] == w)
    # 3. naive integer matmul C (64x64) = A (64x64) B (64x64): thread g -> C[g / 64][g % 64]
    N = 64; n = N*N
    rng = random.Random('matmul')
    A = [rng.randrange(16) for _ in range(n)]; Bm = [rng.randrange(16) for _ in range(n)]
    b = mov(27, N-1)+e('and r21, r1, r27')+mov(27, 6)+e('asr r20, r1, r27')                  # r20 = row, r21 = col
    b += mov(27, N)+e('imadd r5, r20, r27, 0')+mov(27, n)+e('iadd r6, r21, r27')+mov(10, 0)    # r5 = row*64 (A), r6 = 4096 + col (B)
    body = e('device_load 0, i32, x, r7, u8_u9, r5, unsigned')+e('device_load 0, i32, x, r8, u8_u9, r6, unsigned')+e('wait 0')
    body += e('imadd r10, r7, r8, r10')+e('iadd r5, r5, 1')+mov(27, N)+e('iadd r6, r6, r27')
    b += mov(22, N)+counted_reg(body, 22)+mov(27, 2*n)+e('iadd r26, r1, r27')+agx_loop.store(10, 8, 26)+e('wait 0')+e('iadd r2, r10, 0')
    C = [sum(A[r*N+k]*Bm[k*N+c] for k in range(N)) for r in range(N) for c in range(N)]
    yield ('kernel', 'matmul_int_64', 'C = A B, 64x64 int32, one thread per element, 64-trip loop', b, n, 256, 64, 0, A+Bm+[0]*n,
           lambda out, tab, w=C: tab[2*n:3*n] == w)
    # 4. per-workgroup exclusive prefix sum (256 lanes): simd_prefix_iadd + per-group totals in threadgroup memory
    n = 4096
    rng = random.Random('scan')
    vals = [rng.randrange(1000) for _ in range(n)]
    b = B.get_sr(53, 20)+e('device_load 0, i32, x, r6, u8_u9, r1, unsigned')+e('wait 0')+e('simd_prefix_iadd r7, r6')
    b += e('iadd r8, r7, r6')+B.get_sr(52, 21)+mov(25, 31)+mov(27, 16)+e('imadd r22, r20, r27, 0')
    b += if_else('if_icmp r0l, seq, r21, r25, 1', None, tg(False, 'i32', 8, base=22, index=0)+e('wait 0'), b'')+BAR
    # r9 = sum of totals of SIMD groups 0 .. warp-1 (loop over w with an if guard for warp 0)
    inner = e('imadd r23, r16, r27, 0')+tg(True, 'i32', 11, base=23, index=0)+e('wait 0')+e('iadd r9, r9, r11')
    loop = mov(16, 0)+e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    lp = inner+e('iadd r16, 1, r16.sx')+e('while_icmp r0l, ult, r16, r20, 1')
    loop += lp+e(f'jmp_exec_any pc-{len(lp)}', f'jmp_exec_any pc-{len(lp)}')+e('pop_exec r0l, 1')
    b += mov(9, 0)+mov(24, 0)+if_else('if_icmp r0l, ugt, r20, r24, 1', None, loop, b'')
    b += e('iadd r10, r7, r9')+mov(27, n)+e('iadd r26, r1, r27')+agx_loop.store(10, 8, 26)+e('wait 0')+e('iadd r2, r10, 0')+BAR
    want = []
    for wgi in range(n//256):
        acc = 0
        for v in vals[256*wgi:256*wgi+256]:
            want.append(acc); acc += v
    yield ('kernel', 'scan_wg256', 'per-workgroup exclusive prefix sum (simd_prefix_iadd, threadgroup totals, barrier)', b, n, 256, 64, 1024,
           vals+[0]*n, lambda out, tab, w=want: tab[n:2*n] == w)
    # 5. histogram of (v & 63): threadgroup atomic add per lane, then 64 lanes add their bin to the global histogram
    import run_drm_tg_atomic as TA, run_drm_atomic as DA
    n = 8192
    rng = random.Random('hist')
    vals = [rng.getrandbits(32) for _ in range(n)]
    b = B.get_sr(51, 20)+mov(25, 64)+mov(27, 16)+e('imadd r22, r20, r27, 0')+mov(8, 0)
    b += if_else('if_icmp r0l, ult, r20, r25, 1', None, tg(False, 'i32', 8, base=22, index=0)+e('wait 0'), b'')+BAR
    b += e('device_load 0, i32, x, r6, u8_u9, r1, unsigned')+e('wait 0')+mov(27, 63)+e('and r7, r6, r27')+mov(27, 16)
    b += e('imadd r23, r7, r27, 0')+mov(9, 1)+TA.local_atomic(0, 0, 23, 9, False)+e('wait 0')+BAR
    add = tg(True, 'i32', 10, base=22, index=0)+e('wait 0')+mov(27, n)+e('iadd r5, r20, r27')+DA.atomic(0, 0, 5, 10, False)+e('wait 0')
    b += if_else('if_icmp r0l, ult, r20, r25, 1', None, add, b'')+e('iadd r2, r7, 0')
    hist = [0]*64
    for v in vals: hist[v & 63] += 1
    yield ('kernel', 'histogram64', 'histogram of 8192 values (v & 63): threadgroup atomics then device atomics', b, n, 256, 64, 1024,
           vals+[0]*64, lambda out, tab, w=hist: tab[n:n+64] == w)
    # 6. 16x16 transpose through threadgroup memory: workgroup j transposes tile j of a 16 x (16*wgs) matrix
    n = 4096
    rng = random.Random('transpose')
    vals = [rng.getrandbits(32) for _ in range(n)]
    b = B.get_sr(51, 20)+mov(27, 15)+e('and r21, r20, r27')+mov(27, 4)+e('asr r22, r20, r27')   # r21 = x, r22 = y
    b += e('device_load 0, i32, x, r6, u8_u9, r1, unsigned')+e('wait 0')+mov(27, 4)+e('imadd r23, r20, r27, 0')
    b += tg(False, 'i32', 6, base=23, index=0)+e('wait 0')+BAR
    b += mov(27, 16)+e('imadd r24, r21, r27, r22')+mov(27, 4)+e('imadd r24, r24, r27, 0')+tg(True, 'i32', 7, base=24, index=0)+e('wait 0')
    b += mov(27, n)+e('iadd r26, r1, r27')+agx_loop.store(7, 8, 26)+e('wait 0')+e('iadd r2, r7, 0')
    want = []
    for j in range(n//256):
        tile = vals[256*j:256*j+256]
        want += [tile[16*x+y] for y in range(16) for x in range(16)]
    yield ('kernel', 'transpose16', '16x16 tile transpose through threadgroup memory', b, n, 256, 64, 1024, vals+[0]*n,
           lambda out, tab, w=want: tab[n:2*n] == w)


SR_GEOM = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 48, 49, 50, 51, 52, 53, 56, 57, 58, 59, 80, 81, 82]


def gen_geometry():
    """Every thread of a 1-D/2-D/3-D launch stores the geometry special registers at table[linear * 24 + k]
    (linear = x + gx*(y + gy*z) from SR80..82). count = gx (the harness's u5 / out size)."""
    shapes = [(64, 1, 1, 32, 1, 1), (96, 4, 1, 32, 2, 1), (48, 6, 4, 16, 3, 2), (42, 3, 2, 7, 3, 2), (256, 2, 2, 64, 2, 2),
              (24, 24, 1, 8, 8, 1), (33, 1, 1, 33, 1, 1), (100, 3, 1, 10, 3, 1)]
    for gx, gy, gz, lx, ly, lz in shapes:
        total = gx*gy*gz
        b = B.get_sr(80, 20)+B.get_sr(81, 21)+B.get_sr(82, 22)+mov(27, gy)+e('imadd r23, r22, r27, r21')
        b += mov(27, gx)+e('imadd r23, r23, r27, r20')+mov(27, len(SR_GEOM))+e('imadd r26, r23, r27, 0')
        for k, sr in enumerate(SR_GEOM):
            b += B.get_sr(sr, 14)+agx_loop.store(14, 8, 26)+e('iadd r26, r26, 1')
        b += e('wait 0')+e('iadd r2, r23, 0')
        name = f'geom_{gx}x{gy}x{gz}_wg{lx}x{ly}x{lz}'
        yield ('geometry', name, f'special registers {SR_GEOM} per thread, grid {gx}x{gy}x{gz}, workgroup {lx}x{ly}x{lz}',
               b, gx, lx, 64, 0, [0]*(total*len(SR_GEOM)), None, (gx, gy, gz, lx, ly, lz))


import re as _re


def retarget(t, a, b, d):
    """Map the template's registers: r6/r7 -> a/a+1, r8/r9 -> b/b+1, r14/r15 -> d/d+1 (pairs stay aligned)."""
    m = {6: a, 7: a+1, 8: b, 9: b+1, 14: d, 15: d+1}
    return _re.sub(r'(?<![A-Za-z0-9])r(\d+)', lambda x: f'r{m.get(int(x.group(1)), int(x.group(1)))}', t)


def gen_random():
    """Random programs over the template pool: 24-40 instructions reading inputs r6..r13 and results r16..r23, writing
    r16..r23; a third wrap chunks in lane-dependent if/else and per-lane loops. r16..r23 -> table[8n + 8*gid + k]."""
    n, wg = 1024, 256
    pool = [t for t in ST.templates() if not t.startswith('GETSR')]
    load = mov(27, 8)+e('imadd r5, r1, r27, 0')+e('iadd r26, r5, 4')+e('device_load 0, i32, xyzw, r6_r7_r8_r9, u8_u9, r5, unsigned')
    load += e('device_load 0, i32, xyzw, r10_r11_r12_r13, u8_u9, r26, unsigned')+e('wait 0')
    for k in range(150):
        rng = random.Random(f'random-program-{k}')
        init = b''.join(mov(16+i, rng.getrandbits(32)) for i in range(8))
        def chunk(m):
            out = b''; text = []
            for _ in range(m):
                t = rng.choice(pool)
                a = rng.choice([6, 8, 10, 12, 16, 18, 20, 22]); bb = rng.choice([6, 8, 10, 12, 16, 18, 20, 22]); d = rng.choice([16, 18, 20, 22])
                tt = retarget(t, a, bb, d)
                out += e(tt); text.append(tt)
            return out, text
        body = load+init
        lines = []
        structured = k >= 100
        nins = rng.randrange(24, 41)
        if not structured:
            c, t = chunk(nins); body += c; lines += t
        else:
            c1, t1 = chunk(nins//3); c2, t2 = chunk(nins//3); c3, t3 = chunk(nins//3)
            body += c1+mov(24, rng.randrange(1, 60))+if_else('if_icmp r0l, ult, r6, r24, 1', 'else_icmp r0l, ugt, r6, r24, 1', c2, c3)
            c4, t4 = chunk(4)
            body += mov(27, 7)+e('and r25, r7, r27')+counted_reg(c4, 25)
            lines += t1+['if r6 < k {']+t2+['} else {']+t3+['}', 'loop (r7 & 7) + 1 times {']+t4+['}']
        body += mov(27, 8*n)+mov(25, 8)+e('imadd r26, r1, r25, r27')
        for i in range(8):
            body += agx_loop.store(16+i, 8, 26)+e('iadd r26, r26, 1')
        body += e('wait 0')+e('iadd r2, r16, 0')
        table = [operand(rng) for _ in range(8*n)]+[0]*(8*n)
        yield ('random', f'prog_{k:03d}', '; '.join(lines), body, n, wg, 64, 0, table)


GENERATORS = {'alu': lambda: gen_alu(False), 'alu_masked': lambda: gen_alu(True), 'mem_load': gen_mem_load,
              'mem_store': gen_mem_store, 'threadgroup': gen_tg, 'atomic': gen_atomic, 'tg_atomic': gen_tgatomic,
              'control': gen_control, 'kernel': gen_kernels, 'geometry': gen_geometry, 'random': gen_random}


def record(R, out_dir, cat, name, desc, body, n, wg, regs, shared, table, oracle=None, dims=None):
    code = drm.build_main(body)
    res = [run(R, code, n, wg, table, regs, shared, dims) for _ in range(2)]
    if oracle and not oracle(res[0][0], res[0][1]):
        raise RuntimeError(f'{name}: hardware result differs from the host oracle (kernel bug?); not recorded')
    det_out, det_table = res[0][0] == res[1][0], res[0][1] == res[1][1]
    det = det_out and det_table
    out, tab, fw, va = res[0]
    rounded = (n+31)//32*32
    words = 8+rounded+8+16
    table_off = max(1 << 20, (0x8000+words*4+16383)//16384*16384)
    case = {'name': name, 'category': cat, 'description': desc, 'device': 'M2 Max G14C B1 (direct DRM)',
            'code': code.hex(), 'launch': {'grid': list(dims[:3]) if dims else [n, 1, 1], 'wg': list(dims[3:]) if dims else [wg, 1, 1],
                                          'count': n, 'regs': regs, 'shared_bytes': shared},
            'layout': {'table_va': va, 'out_va': va-table_off+0x8000, 'out_words': words,
                       'uniforms': {'u4': 0, 'u5': n, 'u6_u7': 'out_va', 'u8_u9': 'table_va'},
                       'out_init': 'words 0..words-16 = 0xcdcdcdcd, last 16 = 0xa5a50000 + i'},
            'input': {'table': b64(table)}, 'expect': {'out': b64(out), 'table': b64(tab)},
            'deterministic': det, 'deterministic_out': det_out, 'deterministic_table': det_table, 'fw_ns': [res[0][2], res[1][2]], 'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    p = out_dir/cat/f'{name}.json.gz'
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, 'wt') as f:
        json.dump(case, f)
    return det


def gate(body):
    """Every instruction decodes (the raw get_sr / 0x78 forms excepted) and nothing undecodable is submitted."""
    _, g = agx_loop._tools()
    p = 0
    while p < len(body):
        n = g.opcode_to_number(body[p:p+16]); d = g.get_instruction_descriptor(n)
        if d is None: raise RuntimeError(f'undecodable at {p}: {body[p:p+8].hex()}')
        if d.name.startswith('TODO') or d.name in ('trap', 'call', 'ret', 'doorbell'): raise RuntimeError(f'refusing {d.name}')
        p += d.decode_size(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', default=','.join(GENERATORS))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', type=Path, default=ROOT/'testsuite')
    a = ap.parse_args()
    if 'control' in a.only.split(','): control_gate()
    gpu_safety.require_budget(1)
    R = B.Runner(Path('/scratch/apple-m2-playground/sim/suite'))
    total = nondet = 0
    for cat in a.only.split(','):
        for k, item in enumerate(GENERATORS[cat]()):
            c, name, desc, body, n, wg, regs, shared, table = item[:9]
            oracle = item[9] if len(item) > 9 else None
            dims = item[10] if len(item) > 10 else None
            if a.limit and k >= a.limit: break
            gate(body)
            det = record(R, a.out, c, name, desc, body, n, wg, regs, shared, table, oracle, dims)
            total += 1; nondet += not det
            print(json.dumps({'category': c, 'case': name, 'deterministic': det}), flush=True)
    print(json.dumps({'cases': total, 'nondeterministic': nondet}))


if __name__ == '__main__':
    main()
