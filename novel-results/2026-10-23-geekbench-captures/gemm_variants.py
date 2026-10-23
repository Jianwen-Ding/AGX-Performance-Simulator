"""Ablation variants of gemm_block_size_bias's inner loop (.LBB1_9), assembled with llvm-mc.

gemm_variants.py ASM OUTDIR : writes OUTDIR/<variant>.o for each variant (timing experiments; results are wrong).
"""
import sys, re, subprocess, os

asm_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
lines = open(asm_path).read().split('\n')
start = next(i for i, l in enumerate(lines) if l.startswith('gemm_block_size_bias:'))
end = next(i for i in range(start, len(lines)) if re.match(r'\s*\.size\s+gemm_block_size_bias', lines[i]))
loop = next(i for i in range(start, end) if lines[i].startswith('.LBB1_9:'))
loop_end = next(i for i in range(loop, end) if 'jmp_exec_any .LBB1_9' in lines[i])


def variant(name, edit):
    new = list(lines)
    for i in range(loop + 1, loop_end):
        new[i] = edit(new[i])
    path = os.path.join(out, name)
    open(path + '.asm', 'w').write('\n'.join(new))
    subprocess.run(['/scratch/llvm-agx-build/bin/llvm-mc', '-triple=agx', '-mcpu=agx-g14c', '-filetype=obj',
                    path + '.asm', '-o', path + '.o'], check=True)
    print(name)


ins = lambda l: l.strip().split(' ')[0] if l.startswith('\t') else ''
variant('base', lambda l: l)
variant('no_fma', lambda l: '' if ins(l) == 'fmadd32' else l)
# keep the loop counter (r79) increment
variant('no_addr_iadd', lambda l: '' if ins(l) == 'iadd' and 'r79' not in l else l)
variant('x_only', lambda l: l.replace('threadgroup_load i32, xy,', 'threadgroup_load i32, x,') if ins(l) == 'threadgroup_load' else l)
variant('no_inner_tg', lambda l: '' if ins(l) == 'threadgroup_load' else l)
# the loop body runs once per k pair: drop everything but the counter
variant('no_inner_loop', lambda l: l if ('r79' in l or 'while_icmp' in l or 'jmp_exec' in l or not l.startswith('\t')) else '')

# which threadgroup loads cost: the 8 broadcast xy loads (A, same address in every lane) or the 2 per-lane x loads (B)
variant('no_xy', lambda l: '' if ins(l) == 'threadgroup_load' and ', xy,' in l else l)
variant('no_x', lambda l: '' if ins(l) == 'threadgroup_load' and ', x,' in l else l)
_n = {'k': 0}
def _half_xy(l):
    if ins(l) == 'threadgroup_load' and ', xy,' in l:
        _n['k'] += 1
        return '' if _n['k'] % 2 == 0 else l
    return l
variant('half_xy', _half_xy)

# Mesa's form: the threadgroup loads issued back to back, then one wait before their first use (the first fmadd32)
def _wait_before_fma():
    seen = {'done': False}
    def f(l):
        if ins(l) == 'fmadd32' and not seen['done']:
            seen['done'] = True
            return '\twait 0\n' + l
        return l
    return f
variant('wait_before_fma', _wait_before_fma())
