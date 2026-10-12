"""ALU instruction templates for the hardware-recorded test suite (record_suite.py).

Each template is one applegpu-assemblable instruction with sources among r6..r13 (filled per lane from the
input table; 16-bit halves r6l/r6h ...) and destination r14 (r14_r15 for 64-bit, r14l for 16-bit). Excluded on
purpose (they hung the GPU in the sub-op finding): funop functions 2, 3, 5, 7, 11, 15; convert with the round
field's bit 27 set; icond 3/7 and fcond 4 ("never" codes, kept out to stay with named encodings).
"""
ICOND = ['ueq', 'ult', 'ugt', 'seq', 'slt', 'sgt']
FCOND = ['eq', 'lt', 'gt', 'ltn', 'gte', 'lte', 'gtn']
ICOND_N = ['nueq', 'ugte', 'ulte', 'nseq', 'sgte', 'slte']   # inverted forms exist for ballots only
FCOND_N = ['neq', 'nlt', 'ngt', 'ngte', 'nlte']


def templates():
    T = []
    add = T.extend
    # integer add / sub / 64-bit / 16-bit
    add(['iadd r14, r6, r7', 'iadd r14, r6, 5', 'isub r14, r6, r7', 'isub r14, r6, 9', 'iadd r14, r6, r7, lsl 1',
         'iadd r14, r6, r7, lsl 4', 'iadd.sat r14, r6, r7', 'isub.sat r14, r6, r7', 'iadd r14_r15, r6_r7, r8_r9',
         'isub r14_r15, r6_r7, r8_r9', 'iadd r14_r15, r6, r7', 'iadd r14_r15, r6.sx, r7.sx', 'iadd r14_r15, r6_r7, r8.sx, lsl 2',
         'iadd r14l, r6l, r7l', 'iadd r14l, r6h, r7l', 'iadd r14, r6l, r7h', 'iadd r14, r6.sx, r7l.sx', 'iadd.sat r14l, r6l, r7l'])
    # multiply-add
    add(['imadd r14, r6, r7, r8', 'imadd r14, r6, r7, 0', 'imadd r14, r6, 3, r8', 'imadd r14, r6, r7, r8, lsl 2',
         'imadd r14_r15, r6, r7, r8_r9', 'imadd r14_r15, r6.sx, r7.sx, r8_r9', 'imadd r14_r15, r6, r7, 0',
         'imadd r14l, r6l, r7l, r8l', 'imadd.sat r14, r6, r7, r8'])
    # bit ops
    for op in ('and', 'or', 'xor', 'nand', 'nor', 'xnor'):
        add([f'{op} r14, r6, r7', f'{op} r14, r6, 90', f'{op} r14l, r6l, r7h'])
    add(['not r14, r6', 'mov r14, r6'])
    # bit field / shifts / interleave
    add(['bfi r14, r6, r7, r8', 'bfi r14, r6, r7, 8', 'bfeil r14, r6, r7, r8', 'extr r14, r6, r7, r8', 'shlhi r14, r6, r7, r8',
         'shrhi r14, r6, r7, r8', 'asr r14, r6, r8', 'asr r14, r6, 3', 'asrh r14, r6, r8', 'bitrev r14, r6', 'popcount r14, r6',
         'ffs r14, r6', 'intl r14, r6, r7', 'intl r14, r6, 0', 'bfi r14l, r6l, r7l, r8l', 'popcount r14l, r6l'])
    # compare-select
    for cc in ICOND:
        add([f'icmpsel {cc}, r14, r6, r7, r8, r9'])
    for cc in FCOND:
        add([f'fcmpsel {cc}, r14, r6, r7, r8, r9'])
    add(['icmpsel ult, r14l, r6l, r7l, r8l, r9l', 'fcmpsel lt, r14l, r6l, r7l, r8l, r9l', 'icmpsel seq, r14, r6, 0, r8, 7'])
    # float 32
    for f in ('fadd32', 'fmul32'):
        add([f'{f} r14, r6, r7', f'{f} r14, r6.neg, r7.abs', f'{f} r14, r6, 0.5', f'{f}.sat r14, r6, r7', f'{f} r14, r6l, r7'])
    add(['fmadd32 r14, r6, r7, r8', 'fmadd32 r14, r6.neg, r7, r8.abs', 'fmadd32 r14, r6, 0.25, 1.5', 'fmadd32.sat r14, r6, r7, r8',
         'fmadd32 r14, r6l, r7, r8'])
    # float 16
    for f in ('fadd16', 'fmul16'):
        add([f'{f} r14l, r6l, r7l', f'{f} r14l, r6h.neg, r7l.abs', f'{f} r14l, r6l, 0.5', f'{f}.sat r14l, r6l, r7l'])
    add(['fmadd16 r14l, r6l, r7l, r8l', 'fmadd16 r14l, r6l.neg, r7h, r8l.abs', 'fmadd16.sat r14l, r6l, r7l, r8l'])
    # rounding / transcendental (named funop functions only)
    for f in ('floor', 'ceil', 'trunc', 'rint', 'rcp', 'rsqrt', 'rsqrt_special', 'log2', 'exp2', 'sin_pt_1', 'sin_pt_2'):
        add([f'{f} r14, r6', f'{f} r14l, r6l'])
    add(['rcp r14, r6.neg', 'log2 r14, r6.abs'])
    # convert (round field rtz / rte only)
    for m in ('f_to_u32', 'f_to_s32', 'u32_to_f', 's32_to_f', 'f_to_u16', 'f_to_s16', 'u16_to_f', 's16_to_f', 'u8_to_f', 's8_to_f'):
        for rnd in ('rtz', 'rte'):
            add([f'convert {m}, r14, r6, {rnd}'])
    add(['convert f_to_u32, r14l, r6, rtz', 'convert u16_to_f, r14l, r6l, rte'])
    # cross-lane
    for s in ('simd_shuffle', 'simd_shuffle_down', 'simd_shuffle_up', 'simd_shuffle_xor', 'quad_shuffle', 'quad_shuffle_down',
              'quad_shuffle_up', 'quad_shuffle_xor', 'simd_shuffle_rotate_up', 'quad_shuffle_rotate_up'):
        add([f'{s} r14, r6, r7l', f'{s} r14, r6, 3'])
    for op in ('and', 'or', 'xor', 'iadd', 'fadd', 'fmul', 'min.u', 'max.u', 'min.s', 'max.s', 'min.f', 'max.f'):
        add([f'simd_{op} r14, r6', f'quad_{op} r14, r6'])
    for op in ('iadd', 'fadd', 'fmul', 'and', 'or', 'xor', 'min.s', 'min.u', 'max.s', 'max.u'):
        add([f'simd_prefix_{op} r14, r6'])
    for op in ('iadd', 'fadd', 'fmul'):
        add([f'quad_prefix_{op} r14, r6'])
    for cc in ICOND+ICOND_N:
        add([f'icmp_ballot r14, {cc}, r6, r7'])
    for cc in ('seq', 'ult', 'slt', 'nseq'):
        add([f'icmp_quad_ballot r14, {cc}, r6, r7'])
    for cc in FCOND+FCOND_N:
        add([f'fcmp_ballot r14, {cc}, r6, r7'])
    for cc in ('eq', 'lt', 'gt', 'gte'):
        add([f'fcmp_quad_ballot r14, {cc}, r6, r7'])
    # moves / special registers
    add(['mov_imm r14, 305419896, 0b0'])
    add([f'GETSR {n}' for n in (0, 4, 7, 48, 51, 52, 53, 56, 57, 58, 59, 80)])   # get_sr r14, sr<n> (raw encoding)
    return T


def encode(t):
    """bytes of one template (GETSR n: get_sr r14, sr<n> via the validated raw encoding)."""
    import agx_loop
    if t.startswith('GETSR'):
        from run_drm_opcode_describe import get_sr
        return get_sr(int(t.split()[1]), 14)
    return agx_loop.encode(t)
