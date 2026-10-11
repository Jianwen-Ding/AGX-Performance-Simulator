#!/usr/bin/env python3
"""One pinned G14C B1 imadd addend-C field patch; no general binary fuzzing.

Mesa 25.3.6 vk_shader.c: 72-byte wrapper; SHA1 at [48:68], computed
across the entire blob with those 20 bytes zero. hk_shader.c serializes
stage, two metadata structs, aligned uint32 code/data lengths, then bytes.
This deliberately accepts only the measured index control blob
(SHA256 d0b99dace1de6eb2467c64ce872e8f5c02458d1edae0a3f74b9ed2c17cff0ba3)
and changes only the imadd immediate addend field C from 0 to 1:
instruction-relative bit 40, blob byte 0x957, 0x00 -> 0x01.

The index shader consumes the imadd destination r2 only as store data
(xor with u9, then device_store data operand); addresses, bounds and the
execution mask do not depend on r2. The patcher verifies that data-only
path by bounded decode of the post-target main region before and after.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

CONTROL_SHA256 = 'd0b99dace1de6eb2467c64ce872e8f5c02458d1edae0a3f74b9ed2c17cff0ba3'
MODEL_SHA256 = '986f6c6ec5547c5f61e357c7c0df614cb05b3382eed705d4ef7a79c4d3a2c46f'
DISASM_SHA256 = 'e7ef4908a0e35fb62c4c81b8f063f5235cec0221cd377fd3fdfe23981a0b86bd'
CODE_START = 0x8bc
CODE_SIZE = 336
MAIN_OFFSET = 0x80
PREAMBLE_OFFSET = 0x100
TARGET_RELATIVE = 0x96
TARGET_SIZE = 8
OLD_INSTRUCTION = bytes.fromhex('9e09820659000000')
NEW_INSTRUCTION = bytes.fromhex('9e09820659010000')
BEFORE_FIELDS = {'S': 0, 'N': 0, 'Dt': 3, 'D': 4, 'At': 10, 'As': 1, 'A': 2,
                 'Bt': 6, 'Bs': 1, 'B': 16, 'Ct': 0, 'Cs': 0, 'C': 0, 's': 0}
# Exact pinned post-target main disassembly (pinned input + pinned model).
# It shows r2 flowing only into the xor data operand and the device_store
# data operand; address operands stay u6_u7/r1 and control stays r0l-based.
POST_TARGET_DISASSEMBLY = (
    '  9e: 7e09442a5900         xor              r2, r2, u9\n'
    '  a4: 0e05082064000000     iadd             r1, 8, r1.sx\n'
    '  ac: 45112c0e00c01200     device_store     0, i32, x, r2, u6_u7, r1, unsigned, 0\n'
    '  b4: 3800                 wait             0\n'
    '  b6: 520e00000000         pop_exec         r0l, 1\n'
    '  bc: 8800                 stop             \n'
)
BEFORE_TARGET_LINE = '  96: 9e09820659000000     imadd            r2.cache, r1.cache.sx, u8.sx, 0\n'
AFTER_TARGET_LINE = '  96: 9e09820659010000     imadd            r2.cache, r1.cache.sx, u8.sx, 1\n'


def digest(b):
    return hashlib.sha256(b).hexdigest()


def integrity(b):
    zeroed = bytearray(b)
    zeroed[48:68] = bytes(20)
    return hashlib.sha1(zeroed).digest()


def load_model(applegpu_root):
    """Import the pinned applegpu model without writing into vendor trees."""
    root = Path(applegpu_root).resolve()
    if digest((root / 'applegpu.py').read_bytes()) != MODEL_SHA256:
        raise ValueError('model hash mismatch')
    if digest((root / 'disassemble.py').read_bytes()) != DISASM_SHA256:
        raise ValueError('disassembler hash mismatch')
    sys.dont_write_bytecode = True  # Reference imports must not write into vendor trees.
    sys.path.insert(0, str(root))
    import applegpu
    import disassemble
    if (Path(applegpu.__file__).resolve() != root / 'applegpu.py' or
            Path(disassemble.__file__).resolve() != root / 'disassemble.py'):
        raise ValueError('imported model origin mismatch')
    return applegpu, disassemble


def disassembly(disassemble, blob, start):
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        disassemble.disassemble(blob[CODE_START + start:CODE_START + CODE_SIZE], start)
    return capture.getvalue()


def build_patch(b, applegpu_root):
    """Validate the pinned index blob and return the one-field patch evidence.

    Raises ValueError on any malformed input, layout mismatch, decode
    mismatch or data-path violation. Performs no driver calls.
    """
    if not isinstance(b, (bytes, bytearray)):
        raise TypeError('input must be bytes')
    b = bytes(b)
    if digest(b) != CONTROL_SHA256:
        raise ValueError('unrecognized control hash; refuse patch')
    applegpu, disassemble = load_model(applegpu_root)
    if (len(b) != 2572 or b[:16] != b'MesaVkShaderBin\0' or
        struct.unpack_from('<I', b, 16)[0] != 26 or
        b[20:36].hex() != '2313bd7c3909b8ad0f31883eab5c412e' or
        struct.unpack_from('<I', b, 36)[0] != 0 or
        struct.unpack_from('<Q', b, 40)[0] != len(b) or b[48:68] != integrity(b) or
        b[72] != 5 or struct.unpack_from('<II', b, CODE_START - 8) != (CODE_SIZE, 0)):
        raise ValueError('wrapper/stage/code-size/integrity mismatch')
    offset = CODE_START + TARGET_RELATIVE
    old = b[offset:offset + TARGET_SIZE]
    if old != OLD_INSTRUCTION:
        raise ValueError('target instruction mismatch')
    instr = applegpu.opcode_to_number(b[offset:])
    desc = applegpu.get_instruction_descriptor(instr)
    fields = dict(desc.decode_fields(instr))
    if type(desc).__name__ != 'IMAddInstructionDesc' or fields != BEFORE_FIELDS:
        raise ValueError('target decode mismatch')
    if desc.decode_size(instr) != TARGET_SIZE:
        raise ValueError('target length mismatch')
    # Immediate C uses low bits [40:46) and high bits [54:56); preserve length,
    # destination, both multiplicand operands, sign extension, shift,
    # saturation and negation.
    patched_instr = desc.patch_fields(instr, {'C': 1})
    new = desc.to_bytes(patched_instr)
    expected_fields = fields | {'C': 1}
    if (new != NEW_INSTRUCTION or len(new) != TARGET_SIZE or
            dict(desc.decode_fields(patched_instr)) != expected_fields or
            int.from_bytes(old, 'little') ^ int.from_bytes(new, 'little') != 1 << 40):
        raise ValueError('patch changed more than the justified addend bit')
    code = b[CODE_START:CODE_START + CODE_SIZE]
    # Verify the known instruction boundary by walking from the observed main.
    cursor = MAIN_OFFSET
    while cursor < TARGET_RELATIVE:
        n = applegpu.opcode_to_number(code[cursor:])
        d = applegpu.get_instruction_descriptor(n)
        if d is None:
            raise ValueError('unknown instruction before target')
        cursor += d.decode_size(n)
    if cursor != TARGET_RELATIVE:
        raise ValueError('target not on decoded boundary')
    changed = bytearray(b)
    changed[offset:offset + TARGET_SIZE] = new
    stale = bytes(changed)
    changed[48:68] = integrity(changed)
    patched = bytes(changed)
    changes = [i for i, (x, y) in enumerate(zip(b, patched)) if x != y]
    if [i for i in changes if not 48 <= i < 68] != [offset + 5]:
        raise ValueError('unexpected byte changes')
    # Data-only path: bounded decode of both mains and preambles. Only the
    # target line may change; the post-target region must match the pinned
    # text showing r2 consumed solely as xor/store data, never as an
    # address, bound or execution mask.
    before_main = disassembly(disassemble, b, MAIN_OFFSET)
    after_main = disassembly(disassemble, patched, MAIN_OFFSET)
    before_preamble = disassembly(disassemble, b, PREAMBLE_OFFSET)
    after_preamble = disassembly(disassemble, patched, PREAMBLE_OFFSET)
    if before_preamble != after_preamble:
        raise ValueError('preamble changed')
    if BEFORE_TARGET_LINE not in before_main or AFTER_TARGET_LINE not in after_main:
        raise ValueError('target disassembly line mismatch')
    if before_main.replace(BEFORE_TARGET_LINE, '') != after_main.replace(AFTER_TARGET_LINE, ''):
        raise ValueError('non-target main disassembly changed')
    if POST_TARGET_DISASSEMBLY not in before_main or POST_TARGET_DISASSEMBLY not in after_main:
        raise ValueError('data-only store path not confirmed by decode')
    return {'patched': patched, 'stale': stale, 'evidence': {
        'input_sha256': digest(b), 'patched_sha256': digest(patched), 'stale_sha256': digest(stale),
        'input_hex': b.hex(), 'patched_hex': patched.hex(), 'stale_hex': stale.hex(),
        'wrapper_sha1_before': b[48:68].hex(), 'wrapper_sha1_after': patched[48:68].hex(),
        'code_start': CODE_START, 'code_size': CODE_SIZE, 'main_offset': MAIN_OFFSET,
        'preamble_offset': PREAMBLE_OFFSET,
        'target_blob_offset': offset, 'target_code_offset': TARGET_RELATIVE,
        'old_instruction': old.hex(), 'new_instruction': new.hex(),
        'before_fields': fields, 'after_fields': expected_fields,
        'before_asm': str(desc.disassemble(instr)), 'after_asm': str(desc.disassemble(patched_instr)),
        'changed_byte_offsets': changes,
        'model_sha256': MODEL_SHA256, 'disassembler_sha256': DISASM_SHA256,
        'before_main_disassembly': before_main, 'after_main_disassembly': after_main,
        'before_preamble_disassembly': before_preamble, 'after_preamble_disassembly': after_preamble}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True, help='new scratch artifact directory')
    p.add_argument('--applegpu', type=Path, default=Path('/save/workspace/AGX_Probe/applegpu'))
    args = p.parse_args()
    try:
        b = args.input.read_bytes()
    except OSError as e:
        raise ValueError(f'input unreadable: {e}')
    result = build_patch(b, args.applegpu)
    evidence = result['evidence']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'patched.bin').write_bytes(result['patched'])
    (args.output / 'stale-integrity.bin').write_bytes(result['stale'])
    for label in ('before', 'after'):
        for region in ('main', 'preamble'):
            (args.output / (label + '-' + region + '.txt')).write_text(evidence[label + '_' + region + '_disassembly'])
    (args.output / 'patch.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'passed': True, 'output': str(args.output), 'patched_sha256': evidence['patched_sha256']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
