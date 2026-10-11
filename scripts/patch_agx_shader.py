#!/usr/bin/env python3
"""One pinned G14C B1 mov-immediate field patch; no general binary fuzzing.

Mesa 25.3.6 vk_shader.c: 72-byte wrapper; SHA1 at [48:68], computed
across the entire blob with those 20 bytes zero. hk_shader.c serializes
stage, two metadata structs, aligned uint32 code/data lengths, then bytes.
This deliberately accepts only the measured constant control blob.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

CONTROL_SHA256 = 'b79bd25941bdea1a0095b219fe0333ac5c18822225de811c9308a38702e9e098'
MODEL_SHA256 = '986f6c6ec5547c5f61e357c7c0df614cb05b3382eed705d4ef7a79c4d3a2c46f'
DISASM_SHA256 = 'e7ef4908a0e35fb62c4c81b8f063f5235cec0221cd377fd3fdfe23981a0b86bd'
CODE_START = 0x8bc
CODE_SIZE = 336
MAIN_OFFSET = 0x80
PREAMBLE_OFFSET = 0x100
TARGET_RELATIVE = 0x96


def digest(b): return hashlib.sha256(b).hexdigest()


def integrity(b):
    zeroed = bytearray(b); zeroed[48:68] = bytes(20)
    return hashlib.sha1(zeroed).digest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True, help='new scratch artifact directory')
    p.add_argument('--applegpu', type=Path, default=Path('/save/workspace/AGX_Probe/applegpu'))
    args = p.parse_args()
    b = args.input.read_bytes()
    if digest(b) != CONTROL_SHA256: raise ValueError('unrecognized control hash; refuse patch')
    if digest((args.applegpu/'applegpu.py').read_bytes()) != MODEL_SHA256: raise ValueError('model hash mismatch')
    if digest((args.applegpu/'disassemble.py').read_bytes()) != DISASM_SHA256: raise ValueError('disassembler hash mismatch')
    if (len(b) != 2572 or b[:16] != b'MesaVkShaderBin\0' or
        struct.unpack_from('<I',b,16)[0] != 26 or
        b[20:36].hex() != '2313bd7c3909b8ad0f31883eab5c412e' or
        struct.unpack_from('<I',b,36)[0] != 0 or
        struct.unpack_from('<Q',b,40)[0] != len(b) or b[48:68] != integrity(b) or
        b[72] != 5 or struct.unpack_from('<II',b,CODE_START-8) != (CODE_SIZE,0)):
        raise ValueError('wrapper/stage/code-size/integrity mismatch')
    sys.dont_write_bytecode = True  # Reference imports must not write into vendor trees.
    sys.path.insert(0,str(args.applegpu.resolve()))
    import applegpu
    import disassemble
    offset = CODE_START + TARGET_RELATIVE
    old = b[offset:offset+6]
    if old.hex() != '620978563412': raise ValueError('target instruction mismatch')
    instr = applegpu.opcode_to_number(old)
    desc = applegpu.get_instruction_descriptor(instr)
    fields = dict(desc.decode_fields(instr))
    if type(desc).__name__ != 'MovImm32InstructionDesc' or fields != {'Dt':2,'D':4,'imm32':0x12345678,'kill':0}:
        raise ValueError('target decode mismatch')
    # Explicit single field at bits [16:48]; preserve length, destination, kill.
    patched_instr = desc.patch_fields(instr, {'imm32':0x12345679})
    new = desc.to_bytes(patched_instr)
    expected_fields = fields | {'imm32':0x12345679}
    if len(new)!=6 or dict(desc.decode_fields(patched_instr))!=expected_fields or instr^patched_instr != 1<<16:
        raise ValueError('patch changed more than justified immediate bit')
    code = b[CODE_START:CODE_START+CODE_SIZE]
    # Verify the known instruction boundary by walking from the observed main.
    cursor = MAIN_OFFSET
    while cursor<TARGET_RELATIVE:
        n=applegpu.opcode_to_number(code[cursor:]); d=applegpu.get_instruction_descriptor(n)
        if d is None: raise ValueError('unknown instruction before target')
        cursor += d.decode_size(n)
    if cursor!=TARGET_RELATIVE: raise ValueError('target not on decoded boundary')
    changed = bytearray(b); changed[offset:offset+6]=new
    stale = bytes(changed)
    changed[48:68]=integrity(changed)
    patched = bytes(changed)
    changes = [i for i,(x,y) in enumerate(zip(b,patched)) if x!=y]
    if [i for i in changes if not 48<=i<68] != [offset+2]: raise ValueError('unexpected byte changes')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'patched.bin').write_bytes(patched)
    (args.output/'stale-integrity.bin').write_bytes(stale)
    evidence = {'input_sha256':digest(b),'patched_sha256':digest(patched),'stale_sha256':digest(stale),
        'input_hex':b.hex(),'patched_hex':patched.hex(),'stale_hex':stale.hex(),
        'wrapper_sha1_before':b[48:68].hex(),'wrapper_sha1_after':patched[48:68].hex(),
        'code_start':CODE_START,'code_size':CODE_SIZE,'main_offset':MAIN_OFFSET,'preamble_offset':PREAMBLE_OFFSET,
        'target_blob_offset':offset,'target_code_offset':TARGET_RELATIVE,
        'old_instruction':old.hex(),'new_instruction':new.hex(),'before_fields':fields,'after_fields':expected_fields,
        'before_asm':str(desc.disassemble(instr)),'after_asm':str(desc.disassemble(patched_instr)),
        'changed_byte_offsets':changes,'model_sha256':MODEL_SHA256,'disassembler_sha256':DISASM_SHA256}
    for label,blob in [('before',b),('after',patched)]:
        for region, start in [('main',MAIN_OFFSET),('preamble',PREAMBLE_OFFSET)]:
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture): disassemble.disassemble(blob[CODE_START+start:CODE_START+CODE_SIZE],start)
            text=capture.getvalue(); (args.output/(label+'-'+region+'.txt')).write_text(text)
            evidence[label+'_'+region+'_disassembly']=text
    (args.output/'patch.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(args.output),'patched_sha256':digest(patched)}))
    return 0

if __name__=='__main__': sys.exit(main())
