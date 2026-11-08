"""Threadgroup-load data set: hardware times (measured 2026-10-23) against agx_dispatch, for checking model changes.

tgcheck.py [--dispatch PATH]
"""
import sys, os, json, subprocess, argparse, statistics, re

ap = argparse.ArgumentParser()
ap.add_argument('--dispatch', default='/scratch/agxsim-target/release/agx_dispatch')
a = ap.parse_args()
G = '/scratch/geekbench-extract'
P = G + '/probes/bcast-w1-f0-g4x1'

def dispatch(code, grid, block, regs, tg, unis, allocs):
    cmd = [a.dispatch, code, '--grid', grid, '--wg', block, '--regs', str(regs), '--tg-bytes', str(tg)]
    for u, v in unis: cmd += ['--uniform', f'{u}={v}']
    for va, n in allocs: cmd += ['--alloc', f'{va:#x}={n}']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return json.loads(r.stdout.strip().splitlines()[-1])['time_us']

rows = []
# hand-written probe loops (8 loads per iteration; 512 iterations), 4 and 32 groups per core; hw us
probe_unis = [(0, 0x10000000), (1, 0), (2, 4096), (3, 38), (4, 1), (5, 1)]
hw = {('p_pure8_bcast', 4): 101.1, ('p_inter1', 4): 152.7, ('p_alu_block', 4): 105.0, ('p_inter3', 4): 164.7,
      ('p_pure8_bcast', 32): 199.0, ('p_inter1', 32): 307.5, ('p_alu_block', 32): 215.5, ('p_inter3', 32): 341.7}
for (name, groups), t in hw.items():
    obj = f'{P}/{name}.o'
    code = f'{P}/{name}.bin'
    if not os.path.exists(code):
        import tempfile, importlib.util
        spec = importlib.util.spec_from_file_location('fc', '/save/workspace/ptx-frontend/ptx/tools/agx_ptx_formalization_check.py')
        fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
        with tempfile.TemporaryDirectory() as tmp:
            k = fc.kernel_info(None, obj, '/scratch/llvm-agx-build/bin/llvm-readelf', '/scratch/llvm-agx-build/bin/llvm-objcopy', tmp)[0]['probe']
        open(code, 'wb').write(k['code'])
    sim = dispatch(code, f'{38 * 32 * groups},1,1', f'{32 * groups},1,1', 64, 16384, probe_unis, [(0x10000000, 1 << 20)])
    rows.append((f'{name} g{groups}', t, sim))
# gemm_block_size_bias ablations (bench_gemm.sh firmware medians), sim on the same launch
gemm_hw = {'base': 229.6, 'no_fma': 213.6, 'no_addr_iadd': 219.7, 'x_only': 226.0, 'no_inner_tg': 119.5,
           'no_inner_loop': 76.4, 'no_xy': 140.5, 'no_x': 205.6, 'half_xy': 181.0}
for v, t in gemm_hw.items():
    code = f'{G}/variants/abl/{v}.k/code.bin'
    r = subprocess.run(['python3', '-I', f'{G}/tools/replay.py', f'{G}/runs/run05-profile.jsonl', f'{G}/kernels-new',
                        '--kernel', 'gemm_block_size_bias', '--limit', '1', '--code', code], capture_output=True, text=True)
    sim = float(re.search(r' sim ([\d.]+) us', r.stdout).group(1))
    rows.append((f'gemm {v}', t, sim))
# the PTX probe matrix
for l in open(f'{G}/probes/matrix1.txt'):
    m = re.match(r'(\S+)\s+hw\s+([\d.]+) us', l)
    if not m: continue
    name = m.group(1)
    d = f'{G}/probes/{name}'
    r = subprocess.run(['python3', '-I', f'{G}/tools/replay.py', f'{d}/one.jsonl', f'{d}/k', '--limit', '1'], capture_output=True, text=True)
    mm = re.search(r' sim ([\d.]+) us', r.stdout)
    if mm: rows.append((name, float(m.group(2)), float(mm.group(1))))
errs = []
for n, h, s in rows:
    e = (s - h) / h; errs.append(abs(e))
    print(f'{n:28s} hw {h:8.1f} sim {s:8.1f} err {e:+5.0%}')
print(f'{len(rows)} cases  median |err| {statistics.median(errs):.0%}  within 10%: {sum(e <= .1 for e in errs)}  within 25%: {sum(e <= .25 for e in errs)}')
