#!/usr/bin/env python3
"""How threadgroup_load/store addresses threadgroup memory (direct DRM).

Mesa (agx_compile.c:1028-1054) addresses shared memory with the BYTE offset in the
16-bit base register and an immediate index of 0. The earlier probes used base 0 and
an element index register, and found only declared/4 bytes usable. This probe:

  stride   Fill memory with a sentinel through base addressing; lane i stores V(i)
           through the INDEX form (format i8/i16/i32); then lane i reads, through base
           addressing, byte offsets size*i and 4*size*i. Which one returns V(i) gives
           the index stride.
  capacity Lane i stores at base byte B+4i and reads B+4*(31-i) (base addressing only),
           for B up to the declared size.

Only decoded, Mesa-emitted instruction forms are used (no candidate opcodes).
"""
import json, os, struct, subprocess, sys, tempfile
from pathlib import Path

import agx_loop
import run_drm_compute as drm
from run_drm_opcode_describe import get_sr, WAIT
from run_drm_sync_probe import finish, H

REPO = Path(__file__).resolve().parents[1]
mov, enc = agx_loop.chains.mov_imm, agx_loop.encode
FMT = {'i8': (0, 1), 'i16': (1, 2), 'i32': (2, 4)}
BARRIER = H('6800')


def tg(load, fmt, reg, base=None, index=0, mask=1):
    """threadgroup_{load,store} <fmt>, x, value, base, index (Mesa agx_pack.c:769-792).

    value: i32 -> r<reg> (32-bit); i8/i16 -> r<reg>l. base: None = immediate 0, else 16-bit
    register r<base>l. index: int = immediate, ('r', n) = r<n>l."""
    _, g = agx_loop._tools()
    f, _ = FMT[fmt]
    wide = fmt == 'i32'
    R = 2*reg
    A, At = (0, 0b10) if base is None else (2*base, 0)
    O, Ot = (index & 0xffff, 1) if isinstance(index, int) else (2*index[1], 0)
    raw = ((0x69 if load else 0x29) | (1 << 8 if wide else 0) | (R & 63) << 9 | 1 << 15 | (A & 63) << 16 | At << 22
           | f << 24 | (O & 63) << 28 | Ot << 34 | mask << 36 | (O >> 6) << 48 | (A >> 6) << 58 | (R >> 6) << 60)
    b = raw.to_bytes(8, 'little'); n = g.opcode_to_number(b); d = g.get_instruction_descriptor(n)
    text = ' '.join(str(d.disassemble(n)).split()) if d else None
    comps = ''.join(c for k, c in enumerate('xyzw') if mask >> k & 1)
    nreg = bin(mask).count('1')
    regs = '_'.join(f'r{reg+k}' for k in range(nreg)) if wide else '_'.join(f'r{reg}{"lh"[k % 2]}' if False else f'r{reg}l' for k in range(1))
    want = (f"threadgroup_{'load' if load else 'store'} {fmt}, {comps}, {regs if wide else f'r{reg}l'}, "
            f"{'0' if base is None else f'r{base}l'}, {index if isinstance(index, int) else f'r{index[1]}l'}")
    if d is None or d.decode_size(n) != 8 or text != want:
        raise ValueError(f'encoding mismatch: {text!r} != {want!r}')
    return b


def shared_word(nbytes):
    return 0x4d | 1 << 8 | 0x24 << 10 | ((nbytes // 256) & 255) << 24


def fill(nbytes, lanes):
    """Every lane writes the sentinel to bytes 4*(lane + lanes*k) for the whole range (base addressing)."""
    b = mov(9, 0xcdcdcdcd)+mov(10, 4*lanes)+enc('imadd r22, r24, r19, r27')   # r22 = 4*lane (r19 = 4, r27 = 0)
    for _ in range(nbytes // (4*lanes)):
        b += tg(False, 'i32', 9, base=22)+enc('iadd r22, r22, r10')
    return b


def stride_body(fmt, nbytes, lanes=32):
    _, size = FMT[fmt]
    b = get_sr(80, 24)+mov(19, 4)+mov(27, 0)+fill(nbytes, lanes)+BARRIER
    b += mov(6, 0xa0000000 if fmt == 'i32' else 0xa000)+enc('iadd r6, r6, r24')        # V(i) (i16/i8 use r6l)
    if fmt == 'i8': b += mov(6, 0)+enc('iadd r6, r24, 64')                             # V(i) = 0x40+i fits a byte
    b += tg(False, fmt, 6, index=('r', 24))+BARRIER
    out = b''
    for slot, mult in enumerate((size, 4*size)):
        out += mov(11, mult)+mov(5, 0)+enc('imadd r23, r24, r11, r27')+tg(True, fmt, 5, base=23)+WAIT
        out += mov(12, 32*slot)+enc('iadd r13, r24, r12')+agx_loop.store(5, 8, 13)+WAIT
    return b+out+finish()


def mask_body(mask_bit):
    """i32 store through index i with only component <mask_bit> enabled; read bytes 16i+4k, k=0..3."""
    b = get_sr(80, 24)+mov(19, 4)+mov(27, 0)+fill(2048, 32)+BARRIER
    reg = 6
    for k in range(4):   # r6..r9 = 0xb0000000 | k<<8 | lane: the readback names the source register
        b += mov(6+k, 0xb0000000 | k << 8)+enc(f'iadd r{6+k}, r{6+k}, r24')
    b += tg_masked(6, mask_bit)
    b += BARRIER
    for k in range(4):
        b += mov(11, 16)+mov(12, 4*k)+enc('imadd r23, r24, r11, r12')+tg(True, 'i32', 5, base=23)+WAIT
        b += mov(12, 32*k)+enc('iadd r13, r24, r12')+agx_loop.store(5, 8, 13)+WAIT
    return b+finish()


def tg_masked(reg, bit):
    """Store with mask = 1<<bit: the decoder names consecutive registers from r<reg> for the enabled components."""
    _, g = agx_loop._tools()
    f, _ = FMT['i32']
    R = 2*reg; O = 2*24
    raw = (0x29 | 1 << 8 | (R & 63) << 9 | 1 << 15 | 0b10 << 22 | f << 24 | (O & 63) << 28 | (1 << bit) << 36 | (O >> 6) << 48)
    b = raw.to_bytes(8, 'little'); n = g.opcode_to_number(b); d = g.get_instruction_descriptor(n)
    text = ' '.join(str(d.disassemble(n)).split())
    if d.decode_size(n) != 8 or not text.startswith(f"threadgroup_store i32, {'xyzw'[bit]}, r{reg}, 0, r24l"):
        raise ValueError(text)
    return b


def combined_body(base_bytes):
    """Store through base register (= base_bytes) AND index register i; read bytes base+16i and base+4i."""
    b = get_sr(80, 24)+mov(19, 4)+mov(27, 0)+fill(2048, 32)+BARRIER
    b += mov(6, 0xc0000000)+enc('iadd r6, r6, r24')+mov(21, base_bytes)+tg(False, 'i32', 6, base=21, index=('r', 24))+BARRIER
    for k, mult in enumerate((16, 4)):
        b += mov(11, mult)+mov(12, base_bytes)+enc('imadd r23, r24, r11, r12')+tg(True, 'i32', 5, base=23)+WAIT
        b += mov(12, 32*k)+enc('iadd r13, r24, r12')+agx_loop.store(5, 8, 13)+WAIT
    return b+finish()


def capacity_body(B, nbytes, lanes=32):
    b = get_sr(80, 24)+mov(19, 4)+mov(27, 0)
    b += mov(9, 0x7e000000)+enc('iadd r6, r9, r24')+mov(10, B)+enc('imadd r22, r24, r19, r10')+tg(False, 'i32', 6, base=22)+BARRIER
    b += mov(11, 0xfffffffc)+mov(12, B+4*(lanes-1))+enc('imadd r23, r24, r11, r12')+tg(True, 'i32', 5, base=23)+WAIT
    return b+agx_loop.store(5, 8, 24)+WAIT+finish()


def main():
    out = Path(tempfile.mkdtemp(prefix='tglayout-', dir='/scratch/apple-m2-playground/drm-compute'))
    exe = out/'drm_compute'
    subprocess.run(['cc', '-std=c11', '-O2', str(REPO/'tests/drm_compute.c'), '-o', str(exe)], check=True)

    def run(body, nbytes, words=64):
        (out/'c.bin').write_bytes(drm.build_main(body)+bytes(16)); (out/'t.bin').write_bytes(struct.pack('<I', 0xcdcdcdcd)*words)
        env = dict(os.environ, AGX_LOCAL_X='32', AGX_SHARED_WORD=str(shared_word(nbytes)), AGX_TABLE_OUT=str(out/'t.out'))
        r = subprocess.run([str(exe), '/dev/dri/renderD128', str(out/'c.bin'), '32', str(out/'rb'), '1', '64', str(out/'t.bin')],
                           capture_output=True, env=env, timeout=20)
        if r.returncode: sys.exit('dispatch failed: '+r.stdout.decode()[-300:])
        return list(struct.unpack(f'<{words}I', (out/'t.out').read_bytes()))

    res = {'stride': {}, 'capacity': []}
    for fmt in ('i32', 'i16', 'i8'):
        t = run(stride_body(fmt, 4096), 4096)
        res['stride'][fmt] = {'natural_offset_reads': [hex(v) for v in t[:32]], 'x4_offset_reads': [hex(v) for v in t[32:64]]}
    res['mask'] = {}
    for bit in range(4):
        t = run(mask_body(bit), 2048, 128)
        res['mask']['xyzw'[bit]] = {f'byte 16i+{4*k}': [hex(v) for v in t[32*k:32*k+4]] for k in range(4)}
    res['combined'] = {}
    for base_bytes in (64, 100):
        t = run(combined_body(base_bytes), 2048, 64)
        res['combined'][str(base_bytes)] = {'read base+16i': [hex(v) for v in t[:4]], 'read base+4i': [hex(v) for v in t[32:36]],
                                           'all_lanes_at_base+16i': t[:32] == [0xc0000000+i for i in range(32)]}
    for nbytes in (2048, 4096, 8192):
        for B in sorted({0, nbytes//4-128, nbytes//4, nbytes//2, nbytes-128, nbytes}):
            t = run(capacity_body(B, nbytes), nbytes, 32)
            res['capacity'].append({'declared_bytes': nbytes, 'base_byte': B, 'highest_byte_touched': B+127,
                                    'exact': t == [0x7e000000+31-i for i in range(32)], 'sample': hex(t[0])})
    text = json.dumps(res, indent=1)
    if len(sys.argv) > 2 and sys.argv[1] == '--output': Path(sys.argv[2]).write_text(text+'\n')
    print(text if len(text) < 4000 else text[:4000])


if __name__ == '__main__':
    main()
