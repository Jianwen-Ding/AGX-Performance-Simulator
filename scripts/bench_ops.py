"""Instruction templates for the benchmark suite (scripts/run_bench.py).

Each entry: name -> (template, class). `{d}` is the chain register (dest and first
source, so consecutive steps are dependent); r20..r23 hold constants. 16-bit forms
use the low half. Only instructions whose base form already ran on hardware
without a fault (ALU bit-flip baselines, sub-op sweeps, this session's findings)
are listed; no new encodings.
"""
OPS = {
    # integer ALU
    'iadd':        ('iadd r{d}, r{d}, r20', 'int'),
    'isub':        ('isub r{d}, r{d}, r20', 'int'),
    'iadd_lsl2':   ('iadd r{d}, r{d}, r20, lsl 2', 'int'),
    'imadd':       ('imadd r{d}, r{d}, r21, r20', 'int'),
    'imul':        ('imadd r{d}, r{d}, r21, 0', 'int'),
    'iadd64':      ('iadd r{d}_r{d1}, r{d}_r{d1}, r20_r21', 'int64'),
    'imadd_wide':  ('imadd r{d}_r{d1}, r{d}, r21, r20_r21', 'int64'),
    'and':         ('and r{d}, r{d}, r20', 'int'),
    'xor':         ('xor r{d}, r{d}, r20', 'int'),
    'bfi':         ('bfi r{d}, r{d}, r20, r22', 'bitfield'),
    'bfeil':       ('bfeil r{d}, r{d}, r20, r22', 'bitfield'),
    'extr':        ('extr r{d}, r{d}, r20, r22', 'bitfield'),
    'shlhi':       ('shlhi r{d}, r{d}, r20, r22', 'bitfield'),
    'shrhi':       ('shrhi r{d}, r{d}, r20, r22', 'bitfield'),
    'asr':         ('asr r{d}, r{d}, r22', 'bitfield'),
    'asrh':        ('asrh r{d}, r{d}, r22', 'bitfield'),
    'bitrev':      ('bitrev r{d}, r{d}', 'bitfield'),
    'popcount':    ('popcount r{d}, r{d}', 'bitfield'),
    'ffs':         ('ffs r{d}, r{d}', 'bitfield'),
    'intl':        ('intl r{d}, r{d}, r20', 'bitfield'),
    'icmpsel':     ('icmpsel ult, r{d}, r{d}, r20, r{d}, r21', 'select'),
    'fcmpsel':     ('fcmpsel lt, r{d}, r{d}, r20, r{d}, r21', 'select'),
    'mov_imm':     ('mov_imm r{d}, 305419896, 0b0', 'move'),
    'iadd_imm':    ('iadd r{d}, r{d}, 1', 'int'),
    'iadd_cache':  ('iadd r{d}, r{d}, r20.cache', 'int'),
    'imadd_cache': ('imadd r{d}, r{d}, r21.cache, r20.cache', 'int'),
    # float ALU
    'fadd32':      ('fadd32 r{d}, r{d}, r20', 'float'),
    'fmul32':      ('fmul32 r{d}, r{d}, r21', 'float'),
    'fmadd32':     ('fmadd32 r{d}, r{d}, r21, r20', 'float'),
    'fadd32_imm':  ('fadd32 r{d}, r{d}, 0.125', 'float'),
    'fadd32_cache': ('fadd32 r{d}, r{d}, r20.cache', 'float'),
    'fadd32_cache2': ('fadd32 r{d}, r{d}.cache, r20.cache', 'float'),
    'fmul32_cache': ('fmul32 r{d}, r{d}, r21.cache', 'float'),
    'fmadd32_imm': ('fmadd32 r{d}, r{d}, 0.5, 0.125', 'float'),
    'fmadd32_cache': ('fmadd32 r{d}, r{d}, r21.cache, r20.cache', 'float'),
    'fadd16':      ('fadd16 r{d}l, r{d}l, r20l', 'half'),
    'fmul16':      ('fmul16 r{d}l, r{d}l, r21l', 'half'),
    'fmadd16':     ('fmadd16 r{d}l, r{d}l, r21l, r20l', 'half'),
    'convert_f2i': ('convert f_to_u32, r{d}, r{d}, rtz', 'convert'),
    'convert_i2f': ('convert u32_to_f, r{d}, r{d}, rte', 'convert'),
    # transcendental / special function unit
    'floor':       ('floor r{d}, r{d}', 'sfu'),
    'rint':        ('rint r{d}, r{d}', 'sfu'),
    'rcp':         ('rcp r{d}, r{d}', 'sfu'),
    'rsqrt':       ('rsqrt r{d}, r{d}', 'sfu'),
    'rsqrt_special': ('rsqrt_special r{d}, r{d}', 'sfu'),
    'log2':        ('log2 r{d}, r{d}', 'sfu'),
    'exp2':        ('exp2 r{d}, r{d}', 'sfu'),
    'sin_pt_1':    ('sin_pt_1 r{d}, r{d}', 'sfu'),
    'sin_pt_2':    ('sin_pt_2 r{d}, r{d}', 'sfu'),
    # SIMD-group / quad cross-lane
    'simd_shuffle':      ('simd_shuffle r{d}, r{d}, r22l', 'simd'),
    'simd_shuffle_down': ('simd_shuffle_down r{d}, r{d}, r22l', 'simd'),
    'simd_shuffle_xor':  ('simd_shuffle_xor r{d}, r{d}, r22l', 'simd'),
    'quad_shuffle':      ('quad_shuffle r{d}, r{d}, r22l', 'simd'),
    'simd_iadd':         ('simd_iadd r{d}, r{d}', 'simd'),
    'simd_prefix_iadd':  ('simd_prefix_iadd r{d}, r{d}', 'simd'),
    'simd_fadd':         ('simd_fadd r{d}, r{d}', 'simd'),
    'simd_min.u':        ('simd_min.u r{d}, r{d}', 'simd'),
    'quad_iadd':         ('quad_iadd r{d}, r{d}', 'simd'),
    'icmp_ballot':       ('icmp_ballot r{d}, ult, r{d}, r20', 'simd'),
    'fcmp_ballot':       ('fcmp_ballot r{d}, lt, r{d}, r20', 'simd'),
}
