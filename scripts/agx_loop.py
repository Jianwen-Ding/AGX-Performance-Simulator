#!/usr/bin/env python3
"""Counted AGX loop bodies plus an offline emulator check (no GPU access).

Loop shape (Mesa's nested-loop pattern; the control prefix leaves in-bounds
lanes at r0l=0 and out-of-bounds lanes at r0l=1):

    mov_imm r3, 0 ; mov_imm r4, K ; mov_imm r2, P ; mov_imm r5..r7, 0
    push_exec r0l, 1                 ; out-of-bounds lanes -> 2
  top:
    <M chain steps round-robin over r2 (and r5..r7 for 2/4 chains)>
    iadd r3, 1, r3.sx
    while_icmp r0l, ult, r3, r4, 1   ; lanes with r3 >= K -> r0l = 1
    jmp_exec_any top
    pop_exec r0l, 1                  ; finished 1 -> 0, out-of-bounds 2 -> 1
    iadd/fadd32 r2, r2, rN           ; fold extra chains into the stored r2

The counter counts up and the loop continues only while r3 < K, so an
inverted condition reading exits after one iteration instead of hanging.
emulate() runs the bytes with applegpu's per-instruction exec semantics
(jmp_exec_any resolved here) under a step limit and returns lane state.
"""
from pathlib import Path
import sys

import patch_agx_sr as patcher
import run_agx_timestamps as chains

APPLEGPU = chains.APPLEGPU


def _tools():
    sys.path.insert(0, str(APPLEGPU))
    import assemble
    import applegpu
    return assemble, applegpu


def encode(text, want=None):
    """Assemble one line and require decode(bytes) to round-trip (or match want)."""
    assemble, g = _tools()
    b = bytes(assemble.assemble_line(text))
    n = g.opcode_to_number(b); d = g.get_instruction_descriptor(n)
    if d is None or d.decode_size(n) != len(b):
        raise ValueError('inconsistent assembly: '+text)
    shown = ' '.join(str(d.disassemble(n)).split())
    if shown != ' '.join((want or text).split()):
        raise ValueError(f'round-trip mismatch: {text!r} -> {shown!r}')
    return b


CHAIN_REGS = (2, 5, 6, 7)  # r3/r4 hold the loop counter and trip count


def loop_body(trips, steps, kind='iadd8', preload=0x10000000, chains_n=1):
    """Return (bytes, oracle) for `trips` iterations of `steps` chain steps over `chains_n` registers."""
    if not (1 <= trips <= 0x7fffffff and 0 <= steps <= 4096):
        raise ValueError('bounded trips/steps required')
    _, g = _tools()
    gpu, _ = patcher.load_model(APPLEGPU)
    start = preload if kind == 'iadd8' else 0
    if chains_n not in (1, 2, 4) or steps % chains_n:
        raise ValueError('chains must be 1/2/4 and divide steps')
    regs = CHAIN_REGS[:chains_n]
    head = chains.mov_imm(3, 0)+chains.mov_imm(4, trips)+chains.mov_imm(2, start)
    head += b''.join(chains.mov_imm(r, 0) for r in regs[1:])
    head += encode('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    step = [chains.encode_step(kind, r, gpu) for r in regs]
    loop = b''.join(step[k % chains_n] for k in range(steps))
    loop += encode('iadd r3, 1, r3.sx')
    loop += encode('while_icmp r0l, ult, r3, r4, 1')
    jump = encode(f'jmp_exec_any pc-{len(loop)}', f'jmp_exec_any pc-{len(loop)}')
    tail = encode('pop_exec r0l, 1')
    op = 'iadd' if kind == 'iadd8' else 'fadd32'
    tail += b''.join(encode(f'{op} r2, r2, r{r}') for r in regs[1:])
    body = head+loop+jump+tail
    return body, chains.oracle(trips*steps, kind, preload)


def store_regbase(reg, base_reg, index_reg):
    """device_store 0, i32, x, r<reg>, r<base>_r<base+1>, r<index>, unsigned: 64-bit register base."""
    _, g = _tools()
    ref = bytes.fromhex('45112c0e00c01200')
    n = g.opcode_to_number(ref); d = g.get_instruction_descriptor(n)
    f = dict(d.decode_fields(n)); f.update(R=2*reg, At=0, A=2*base_reg, O=2*index_reg)
    x = d.encode_fields(f); m = g.get_instruction_descriptor(x)
    if m is not d or dict(m.decode_fields(x)) != f or m.decode_size(x) != 8:
        raise ValueError('register-base store encoding mismatch')
    return x.to_bytes(8, 'little')


def store(reg, base_uniform, index_reg):
    """device_store 0, i32, x, r<reg>, u<base>_u<base+1>, r<index>, unsigned (the control's store form).

    applegpu's assembler cannot parse this form, so only the R/A/O fields of the
    control's own store are changed, and the result must decode back to them."""
    _, g = _tools()
    ref = bytes.fromhex('45112c0e00c01200')  # control: device_store 0, i32, x, r2, u6_u7, r1, unsigned, 0
    n = g.opcode_to_number(ref); d = g.get_instruction_descriptor(n)
    f = dict(d.decode_fields(n)); f.update(R=2*reg, A=2*base_uniform, O=2*index_reg)
    x = d.encode_fields(f); m = g.get_instruction_descriptor(x)
    if m is not d or dict(m.decode_fields(x)) != f or m.decode_size(x) != 8:
        raise ValueError('store encoding mismatch')
    return x.to_bytes(8, 'little')


def uniform_store(reg, half_offset):
    """uniform_store 2, i16, xy, 0, r<reg> (as r<reg>l_r<reg>h), <half_offset>: 32 bits into two uniform halfs.

    Field-edited from the Vulkan control preamble's own uniform_store (c500803d00803000)
    with an exact decode round trip, like store()."""
    _, g = _tools()
    ref = bytes.fromhex('c500803d00803000')
    n = g.opcode_to_number(ref); d = g.get_instruction_descriptor(n)
    f = dict(d.decode_fields(n)); f.update(R=2*reg, O=half_offset)
    x = d.encode_fields(f); m = g.get_instruction_descriptor(x)
    size = m.decode_size(x) if m is d else None
    if m is not d or dict(m.decode_fields(x)) != f:
        raise ValueError(f'uniform_store encoding mismatch for offset {half_offset}')
    return x.to_bytes(size, 'little')


def atomic_add(base_uniform, index_reg, src_reg):
    """atomic add, no return: [u<base>_u<base+1> + r<index>*4] += r<src> (32-bit).

    Built from applegpu's atomic field list with its observed constants
    (u1=1, u2=0, u3=1, u4=4, u5=2), which also satisfy Mesa AGX2.xml's fixed bits
    0:5=0x15 and 26/31/46=1; decode must round-trip exactly."""
    _, g = _tools()
    d = next(x for x in g.instruction_descriptors if x.name == 'atomic')
    f = {'op': 0, 'g': 0, 'Rt': 0, 'R': 0, 'At': 1, 'A': 2*base_uniform, 'Ot': 0, 'O': 2*index_reg, 'Ou': 1,
         'S': 2*src_reg, 'u1': 1, 'u2': 0, 'u3': 1, 'u4': 4, 'u5': 2}
    x = d.encode_fields(f); b = x.to_bytes(8, 'little')
    n = g.opcode_to_number(b); m = g.get_instruction_descriptor(n)
    if m is not d or dict(m.decode_fields(n)) != f or x & 63 != 0x15 or not ((x >> 26) & (x >> 31) & (x >> 46) & 1):
        raise ValueError('atomic encoding mismatch')
    return b


CHASE_LOAD = 'device_load 0, i32, x, r2, u8_u9, r2, unsigned'  # r2 = table[r2] (element-scaled index)
TABLE_VA = 0x1100100000  # data BO + 1 MiB in tests/drm_compute.c


CHASE_REGS = (2,)+tuple(range(5, 20))  # up to 16 chase chains; r3/r4 hold the loop counter


def chase_starts(table, chains_n, start=0):
    """Starting indices spread evenly along the table's single cycle."""
    if chains_n == 1: return [start]
    cycle = [start]; x = table[start]
    while x != start:
        cycle.append(x); x = table[x]
    return [cycle[k*len(cycle)//chains_n] for k in range(chains_n)]


def chase_body(trips, steps, start=0, starts=None):
    """`trips` iterations of `steps` rounds; each round issues one dependent load per chain
    (r2 = table[r2], r5 = table[r5], ...) and then a single wait 0. Chains fold into r2 by iadd."""
    starts = list(starts) if starts is not None else [start]
    if not (1 <= trips <= 0x7fffffff and 1 <= steps <= 256 and len(starts) in (1, 2, 4, 8, 16)):
        raise ValueError('bounded trips/steps and 1/2/4/8/16 chains required')
    regs = CHASE_REGS[:len(starts)]
    head = chains.mov_imm(3, 0)+chains.mov_imm(4, trips)
    head += b''.join(chains.mov_imm(r, v) for r, v in zip(regs, starts))
    head += encode('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    rnd = b''.join(encode(f'device_load 0, i32, x, r{r}, u8_u9, r{r}, unsigned') for r in regs)+encode('wait 0')
    loop = rnd*steps
    loop += encode('iadd r3, 1, r3.sx')
    loop += encode('while_icmp r0l, ult, r3, r4, 1')
    jump = encode(f'jmp_exec_any pc-{len(loop)}', f'jmp_exec_any pc-{len(loop)}')
    tail = encode('pop_exec r0l, 1')+b''.join(encode(f'iadd r2, r2, r{r}') for r in regs[1:])
    return head+loop+jump+tail


def stream_body(trips, steps, stride_words):
    """Grid-stride streaming: lane i (global index in r1) loads T[i + j*S], j = 0..trips*steps-1.

    Each round issues `steps` independent loads (r6..r13) with r5 advancing by S (r14),
    one wait 0, then adds them into r2. r1 is only read (the control suffix uses it).
    """
    if not (1 <= trips <= 0x7fffffff and steps in (1, 2, 4, 8) and 1 <= stride_words < 1 << 31):
        raise ValueError('bounded trips, 1/2/4/8 steps and stride required')
    regs = range(6, 6+steps)
    head = chains.mov_imm(3, 0)+chains.mov_imm(4, trips)+chains.mov_imm(14, stride_words)+chains.mov_imm(2, 0)
    head += encode('iadd r5, r1, 0')
    head += encode('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    loop = b''
    for r in regs:
        loop += encode(f'device_load 0, i32, x, r{r}, u8_u9, r5, unsigned')+encode('iadd r5, r5, r14')
    loop += encode('wait 0')
    loop += b''.join(encode(f'iadd r2, r2, r{r}') for r in regs)
    loop += encode('iadd r3, 1, r3.sx')
    loop += encode('while_icmp r0l, ult, r3, r4, 1')
    jump = encode(f'jmp_exec_any pc-{len(loop)}', f'jmp_exec_any pc-{len(loop)}')
    return head+loop+jump+encode('pop_exec r0l, 1')


def stream_oracle(lane, trips, steps, stride_words):
    """Sum of T[lane + j*S] for T[w] = w."""
    n = trips*steps
    return (n*lane+stride_words*n*(n-1)//2) & 0xffffffff


def check_stream(trips, steps, stride_words, active=17):
    import array
    words = stride_words*trips*steps
    table = array.array('I', range(words))
    body = stream_body(trips, steps, stride_words)
    r = emulate(body, active, table=table, lane_ids=True)
    want = [stream_oracle(t, trips, steps, stride_words) for t in range(32)]
    ok = (all(r['r2'][t] == want[t] for t in range(active)) and r['r0l'] == [0]*active+[1]*(32-active)
          and r['counts'].get('jmp_exec_any') == trips and r['counts'].get('device_load') == trips*steps)
    return ok, body, want, r


def chase_table(entries, stride_words, seed=1):
    """Single random cycle (Sattolo) over entries spaced stride_words apart; other words 0xffffffff."""
    import array
    import random
    rng = random.Random(seed)
    order = array.array('I', range(entries))
    for i in range(entries-1, 0, -1):
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    table = array.array('I', [0xffffffff])*(entries*stride_words)
    table[0::stride_words] = array.array('I', (v*stride_words for v in order))  # next element index (in words)
    return table


def chase_oracle(table, hops, start=0):
    """Index after `hops` dependent loads; raises if the chain leaves the cycle."""
    period = None
    x = start
    seen = 0
    for _ in range(hops):
        x = table[x]
        if x == 0xffffffff: raise ValueError('chain left the cycle')
        seen += 1
        if x == start:
            period = seen
            break
    if period is None: return x
    x = start
    for _ in range(hops % period): x = table[x]
    return x


class _Memory:
    def __init__(self, base, words):
        self.base, self.words = base, words
    def get_u32(self, address):
        i, r = divmod(address-self.base, 4)
        if r or not 0 <= i < len(self.words): raise RuntimeError(f'emulated load outside table: {address:#x}')
        return self.words[i]


def emulate(body, active, max_steps=2_000_000, table=None, lane_ids=False, num_registers=24):
    """Run body on one 32-lane SIMD group; lanes < active start at r0l=0, others 1."""
    _, g = _tools()
    class Uniforms(g.Uniforms):
        def get_reg64(self, regid, thread=None):  # applegpu's uniform-base path passes a thread
            return super().get_reg64(regid)
    unif = Uniforms(); unif.set_reg32(8, TABLE_VA & 0xffffffff); unif.set_reg32(9, TABLE_VA >> 32)
    st = g.CoreState(num_registers=num_registers, uniforms=unif, device_memory=_Memory(TABLE_VA, table or []))
    for t in range(g.SIMD_WIDTH):
        st.set_reg16(0, t, 0 if t < active else 1)
        st.set_reg32(2, t, 0xdeadbeef)  # stale; the body must overwrite r2
        if lane_ids: st.set_reg32(1, t, t)  # control prefix leaves the global thread index in r1
    st.exec = [t < active for t in range(g.SIMD_WIDTH)]
    pc = 0; executed = 0; trace = {}
    while pc < len(body):
        n = g.opcode_to_number(body[pc:]); d = g.get_instruction_descriptor(n)
        size = d.decode_size(n); n &= (1 << (8*size))-1
        trace[d.name] = trace.get(d.name, 0)+1
        if d.name == 'jmp_exec_any':
            off = dict(d.decode_fields(n))['off']
            off = off-(1 << 32) if off & (1 << 31) else off
            pc = pc+off if any(st.exec) else pc+size
        else:
            d.exec(n, st); pc += size
        executed += 1
        if executed > max_steps:
            raise RuntimeError('emulator step limit: loop does not terminate')
    if pc != len(body):
        raise RuntimeError(f'control left the body at pc={pc}')
    return {'r2': [st.get_reg32(2, t) for t in range(g.SIMD_WIDTH)],
            'r0l': [st.get_reg16(0, t) for t in range(g.SIMD_WIDTH)],
            'exec': list(st.exec), 'executed': executed, 'counts': trace}


def chase_want(table, hops, starts):
    return sum(chase_oracle(table, hops, s) for s in starts) & 0xffffffff


def check_chase(trips, steps, table, active=17, start=0, chains_n=1):
    """Emulate a chase loop against `table`; active lanes must hold the host oracle."""
    starts = chase_starts(table, chains_n, start)
    body = chase_body(trips, steps, starts=starts)
    want = chase_want(table, trips*steps, starts)
    r = emulate(body, active, table=table)
    ok = (all(r['r2'][t] == want for t in range(active)) and r['r0l'] == [0]*active+[1]*(32-active)
          and r['counts'].get('jmp_exec_any') == trips and r['counts'].get('device_load') == trips*steps*chains_n
          and r['counts'].get('wait') == trips*steps)
    return ok, body, want, r


def check(trips, steps, kind='iadd8', active=17, preload=0x10000000, chains_n=1):
    """Emulate and require: active lanes hold the oracle and are re-enabled; others stay masked."""
    body, want = loop_body(trips, steps, kind, preload, chains_n)
    r = emulate(body, active)
    ok = (all(r['r2'][t] == want for t in range(active)) and
          r['r0l'] == [0]*active+[1]*(32-active) and r['exec'] == [True]*active+[False]*(32-active) and
          r['counts'].get('jmp_exec_any') == trips)
    return ok, body, want, r


if __name__ == '__main__':
    for trips, steps, kind, active, *rest in [(1, 0, 'iadd8', 32), (1, 1, 'iadd8', 17), (3, 2, 'iadd8', 17),
                                       (10, 4, 'fadd4', 5), (100, 1, 'iadd8', 32), (257, 3, 'fadd4', 1),
                                       (7, 8, 'iadd8', 32, 4), (9, 8, 'fadd4', 20, 2), (5, 16, 'fadd6', 3, 4)]:
        chains_n = rest[0] if rest else 1
        ok, body, want, r = check(trips, steps, kind, active, chains_n=chains_n)
        print({'trips': trips, 'steps': steps, 'chains': chains_n, 'kind': kind, 'active': active, 'ok': ok, 'oracle': hex(want),
               'bytes': len(body), 'executed': r['executed'], 'jumps': r['counts'].get('jmp_exec_any')})
