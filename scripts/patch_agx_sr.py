#!/usr/bin/env python3
"""Pinned data-only special-register read probe. Unknown selectors are experimental."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

from patch_agx_imadd import load_model, integrity, CODE_START, CODE_SIZE
from patch_agx_shader import CONTROL_SHA256

SITE = CODE_START + 0x96


def patch(blob, selector, root=Path('/save/workspace/AGX_Probe/applegpu'), alt=False):
    """alt=True sets bit 15: applegpu's undocumented TODO.sr_thing form."""
    if type(selector) is not int or not 0 <= selector <= 255:
        raise ValueError('selector must be an integer 0..255')
    if hashlib.sha256(blob).hexdigest() != CONTROL_SHA256:
        raise ValueError('unrecognized control binary')
    if (len(blob) != 2572 or blob[48:68] != integrity(blob) or
            struct.unpack_from('<II', blob, CODE_START - 8) != (CODE_SIZE, 0) or
            blob[SITE:SITE+6] != bytes.fromhex('620978563412')):
        raise ValueError('pinned layout mismatch')
    gpu, _ = load_model(root)
    # Dt=2: 32-bit ordinary destination. D=4: physical half-register r2.
    raw = 0x72 | (2 << 7) | (4 << 9) | ((selector & 63) << 16) | ((selector >> 6) << 26)
    raw |= (1 << 15) if alt else 0
    read = raw.to_bytes(4, 'little')
    desc = gpu.get_instruction_descriptor(raw)
    expected = 'TodoSrThingInstructionDesc' if alt else 'MovFromSrInstructionDesc'
    if (type(desc).__name__ != expected or desc.decode_size(raw) != 4 or
            dict(desc.decode_fields(raw)) != {'Dt': 2, 'D': 4, 'SR': selector}):
        raise ValueError('special register decode mismatch')
    wait = bytes.fromhex('3800')
    wd = gpu.get_instruction_descriptor(int.from_bytes(wait, 'little'))
    if wd.name != 'wait' or wd.decode_size(int.from_bytes(wait, 'little')) != 2:
        raise ValueError('padding wait decode mismatch')
    changed = bytearray(blob)
    changed[SITE:SITE+6] = read + wait
    changed[48:68] = integrity(changed)
    changed = bytes(changed)
    offsets = [i for i,(a,b) in enumerate(zip(blob, changed)) if a != b]
    if any(not (48 <= i < 68 or SITE <= i < SITE + 6) for i in offsets):
        raise ValueError('unexpected mutation')
    return changed, {'selector': selector, 'alt_bit15': alt, 'site': SITE, 'old': blob[SITE:SITE+6].hex(),
                     'new': (read+wait).hex(), 'decoded': str(desc.disassemble(raw)),
                     'fields': dict(desc.decode_fields(raw)), 'changed_offsets': offsets,
                     'input_sha256': hashlib.sha256(blob).hexdigest(),
                     'patched_sha256': hashlib.sha256(changed).hexdigest(),
                     'wrapper_sha1': changed[48:68].hex()}


def _relocate(blob, main, gpu, strict=True):
    """Install a grown main, relocate the unchanged preamble, refresh sizes/SHA1."""
    # Original main_size includes 16 zero bytes following stop; retain this
    # instruction-fetch padding before aligning the relocated preamble.
    preamble_offset=(0x80+len(main)+16+127)//128*128
    preamble=blob[CODE_START+0x100:CODE_START+CODE_SIZE]
    code=blob[CODE_START:CODE_START+0x80]+main
    code+=bytes(preamble_offset-len(code))+preamble
    changed=bytearray(blob[:CODE_START]+code)
    struct.pack_into('<I',changed,0x41d,len(code))       # agx shader code size
    struct.pack_into('<I',changed,0x83d,len(main)+16)    # main size including padding
    struct.pack_into('<I',changed,0x845,preamble_offset)
    struct.pack_into('<I',changed,CODE_START-8,len(code))
    struct.pack_into('<Q',changed,40,len(changed))      # wrapper total size
    changed[48:68]=integrity(changed)
    # Retain a bounded instruction walk: prefix untouched, known store/base
    # operands untouched, only data reads and explicit address stride doubled.
    decoded=[];cursor=0
    while cursor<len(main):
        n=gpu.opcode_to_number(main[cursor:]);desc=gpu.get_instruction_descriptor(n)
        if desc is None:
            if strict: raise ValueError('unknown main instruction')
            # Opcode scans: record the undecodable remainder instead of guessing its length.
            decoded.append({'offset':0x80+cursor,'bytes':main[cursor:].hex(),'assembly':'<undecoded>'})
            break
        size=desc.decode_size(n)
        decoded.append({'offset':0x80+cursor,'bytes':main[cursor:cursor+size].hex(),'assembly':str(desc.disassemble(n))})
        cursor+=size
    if strict and (cursor!=len(main) or not decoded[-1]['assembly'].strip().startswith('stop')):
        raise ValueError('main termination mismatch')
    return changed,code,preamble_offset,decoded


def patch_pair(blob, selector, gap, constant=False, root=Path('/save/workspace/AGX_Probe/applegpu'), alt=False):
    """Two data reads/stores per lane, separated by exactly gap integer adds.

    Rebuild the pinned main; preserve register allocation, rodata and preamble.
    Adjust only measured size/offset fields when the main grows. Caller must
    dispatch <=16 lanes with the unchanged child's 32-word payload allocation.
    """
    patch(blob, selector, root)  # Enforce the original binary and decoder pins.
    if type(gap) is not int or not 0 <= gap <= 256:
        raise ValueError('gap must be an integer 0..256')
    if (struct.unpack_from('<II',blob,0x419)!=(5,336) or
            struct.unpack_from('<I',blob,0x83d)[0]!=70 or
            struct.unpack_from('<II',blob,0x845)!=(256,128)):
        raise ValueError('measured metadata mismatch')
    gpu,_ = load_model(root)
    sys.path.insert(0,str(root.resolve()))
    import assemble
    if Path(assemble.__file__).resolve()!=root.resolve()/'assemble.py':
        raise ValueError('assembler origin mismatch')
    if hashlib.sha256((root/'assemble.py').read_bytes()).hexdigest()!='6954f39ea05462ba67b1bba93b101470210268e96b3c6de748909db38d940425':
        raise ValueError('assembler hash mismatch')
    def encode(text):
        data=bytes(assemble.assemble_line(text))
        n=gpu.opcode_to_number(data);desc=gpu.get_instruction_descriptor(n)
        if desc is None or desc.decode_size(n)!=len(data) or desc.decode_remainder(n)!=0:
            raise ValueError('inconsistent assembly: '+text)
        return data
    sr,_=patch(blob,selector,root,alt)
    read=bytes.fromhex('620978563412') if constant else sr[SITE:SITE+4]
    prefix=blob[CODE_START+0x80:SITE]  # Original index and active-count check.
    double=encode('iadd r1, r1.sx, r1.sx')
    address=blob[CODE_START+0x9c:CODE_START+0xa4]
    store=blob[CODE_START+0xa4:CODE_START+0xac]
    wait=blob[CODE_START+0xac:CODE_START+0xae]
    end=blob[CODE_START+0xae:CODE_START+0xb6]  # pop_exec and stop.
    increment=encode('iadd r1, 1, r1.sx')
    delay=encode('iadd r2, 1, r2.sx')
    if (double.hex()!='0e05422664000000' or increment.hex()!='0e05012064000000' or
            delay.hex()!='0e09014064000000'):
        raise ValueError('address/delay encoding mismatch')
    main=prefix+read+double+address+store+wait+delay*gap+read+increment+store+wait+end
    changed,code,preamble_offset,decoded=_relocate(blob,main,gpu)
    return bytes(changed), {'selector':selector,'alt_bit15':alt,'pair_gap':gap,'constant_control':constant,
                           'main_size':len(main)+16,'code_size':len(code),'preamble_offset':preamble_offset,
                           'main_hex':main.hex(),'main_disassembly':decoded,
                           'input_sha256':hashlib.sha256(blob).hexdigest(),
                           'patched_sha256':hashlib.sha256(changed).hexdigest(),
                           'assembler_sha256':hashlib.sha256((root/'assemble.py').read_bytes()).hexdigest(),
                           'metadata_mutations':[40,48,0x41d,0x83d,0x845,CODE_START-8]}


def patch_body(blob, body, root=Path('/save/workspace/AGX_Probe/applegpu'), gprs=None, strict=True, trailer=b''):
    """Replace only the original mov_imm r2 with an arbitrary body leaving data in r2.

    The body must not touch r0l/r1 or uniforms; the original index/bounds
    prefix and address/store/pop_exec/stop suffix are byte-identical.
    gprs overrides the 16-bit register allocation (metadata 0x861, nr_gprs=6
    in the control); the driver packs it as USC Registers in groups of 8.
    """
    patch(blob, 80, root)  # Enforce original binary and decoder pins.
    if (struct.unpack_from('<II',blob,0x419)!=(5,336) or
            struct.unpack_from('<I',blob,0x83d)[0]!=70 or
            struct.unpack_from('<II',blob,0x845)!=(256,128)):
        raise ValueError('measured metadata mismatch')
    gpu,_=load_model(root)
    # trailer: bytes placed after stop, i.e. never executed (fetch/decode-ahead probes).
    main=blob[CODE_START+0x80:SITE]+bytes(body)+blob[CODE_START+0x9c:CODE_START+0xb6]+bytes(trailer)
    changed,code,preamble_offset,decoded=_relocate(blob,main,gpu,strict and not trailer)
    if gprs is not None:
        if struct.unpack_from('<HH',changed,0x861)!=(6,8) or not 8<=gprs<=256:
            raise ValueError('nr_gprs layout mismatch or out of range')
        struct.pack_into('<H',changed,0x861,gprs)
        changed[48:68]=integrity(changed)
    return bytes(changed), {'gprs':gprs,'trailer_hex':bytes(trailer).hex(),'body_hex':bytes(body).hex(),'main_hex':main.hex(),
                           'main_disassembly':decoded,'main_size':len(main)+16,
                           'code_size':len(code),'preamble_offset':preamble_offset,
                           'input_sha256':hashlib.sha256(blob).hexdigest(),
                           'patched_sha256':hashlib.sha256(changed).hexdigest()}


def patch_even(blob, selector, constant=False, root=Path('/save/workspace/AGX_Probe/applegpu'), alt=False):
    """Read/store only even global indices, under the original outer bounds mask."""
    changed,evidence=patch(blob,selector,root,alt)
    gpu,_=load_model(root)
    read=bytes.fromhex('620978563412') if constant else changed[SITE:SITE+6]
    # and r2,r1,1; nested if r0l, unsigned-equal,r2,0,1; paired pop_exec.
    parity=bytes.fromhex('7e0942128000')
    condition=bytes.fromhex('520844020000')
    n=gpu.opcode_to_number(condition)
    if dict(gpu.get_instruction_descriptor(n).decode_fields(n))!={'Dt':0,'cc':0,'ccn':0,'At':9,'A':4,'Bt':0,'B':0,'n':1}:
        raise ValueError('parity condition decode mismatch')
    main=blob[CODE_START+0x80:SITE]+parity+condition+read
    main+=blob[CODE_START+0x9c:CODE_START+0xae]+blob[CODE_START+0xae:CODE_START+0xb4]+blob[CODE_START+0xae:CODE_START+0xb6]
    if len(main)+16>128:raise ValueError('main exceeds existing aligned slot')
    result=bytearray(blob)
    result[CODE_START+0x80:CODE_START+0x100]=main+bytes(128-len(main))
    struct.pack_into('<I',result,0x83d,len(main)+16)
    result[48:68]=integrity(result)
    return bytes(result), {'selector':selector,'alt_bit15':alt,'sparse_even':True,'constant_control':constant,
                           'main_hex':main.hex(),'main_size':len(main)+16,
                           'input_sha256':hashlib.sha256(blob).hexdigest(),
                           'patched_sha256':hashlib.sha256(result).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--selector', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    blob, evidence = patch(a.input.read_bytes(), a.selector)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'shader.bin').write_bytes(blob)
    (a.output / 'patch.json').write_text(json.dumps(evidence, indent=2)+'\n')


if __name__ == '__main__':
    main()
