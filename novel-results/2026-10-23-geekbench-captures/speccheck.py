"""Launch-time specialization against the generic code on captured launches: per capture, the kernel compiled from its
module (generic) and as zluda_agx specializes it for the captured arguments and block (agx_specialize), both run on
the capture's memory (capvariant.py): the output hashes must match; prints both back-to-back times.

speccheck.py CAPDIR... [--b2b N]
"""
import sys, os, re, subprocess, tempfile, argparse, importlib.util, ast, json, shlex

G = '/scratch/geekbench-extract'
ap = argparse.ArgumentParser()
ap.add_argument('caps', nargs='+'); ap.add_argument('--b2b', type=int, default=100)
ap.add_argument('--sim', action='store_true', help='also time both codes with agx_dispatch (the simulator)')
ap.add_argument('--dispatch', default='/scratch/agxsim-target/release/agx_dispatch')
a = ap.parse_args()
spec = importlib.util.spec_from_file_location('fc', '/save/workspace/ZLUDA/ptx/tools/agx_ptx_formalization_check.py')
fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
ZOC = f'{G}/cargo-target/release/zoc'
SPEC = f'{G}/cargo-target/release/agx_specialize'
SIZES = {'u8': 1, 's8': 1, 'b8': 1, 'u16': 2, 's16': 2, 'b16': 2, 'f16': 2, 'u32': 4, 's32': 4, 'b32': 4, 'f32': 4,
         'u64': 8, 's64': 8, 'b64': 8, 'f64': 8}
modules = {f: open(os.path.join(G, 'ptx', f)).read() for f in sorted(os.listdir(os.path.join(G, 'ptx'))) if f.endswith('.ptx')}

def code_of(ptx_path, kernel, t):
    subprocess.run([ZOC, '--arch=agx-g14c', '--output-dir', t, ptx_path], check=True, capture_output=True)
    obj = os.path.join(t, os.path.basename(ptx_path)[:-4] + '.o')
    k = fc.kernel_info(None, obj, '/scratch/llvm-agx-build/bin/llvm-readelf', '/scratch/llvm-agx-build/bin/llvm-objcopy', t)[0][kernel]
    out = os.path.join(t, os.path.basename(ptx_path)[:-4] + '.bin')
    open(out, 'wb').write(k['code'])
    # the kernel's own register count (the captured one is the code that ran then)
    asm = open(obj[:-2] + '.asm').read()
    body = asm[asm.index(kernel + ':'):]
    gprs = int(re.search(r'general purpose registers: (\d+) x 32-bit', body).group(1))
    # the uniform layout (arguments, texture heap, grid size): the capture's uniforms are the generic code's
    header = []
    for l in body.splitlines()[1:]:
        if not l.strip().startswith('//'): break
        header.append(l)
    layout = re.findall(r'// (uniform registers: .*|texture heap at .*|sampler heap at .*)', '\n'.join(header))
    # the global data's symbols (zluda_agx specializes only when they are laid out alike)
    syms = subprocess.run(['/scratch/llvm-agx-build/bin/llvm-readelf', '-sW', obj], capture_output=True, text=True).stdout
    layout += sorted(l.split()[1] + ':' + l.split()[2] + ':' + l.split()[7] for l in syms.splitlines() if ' OBJECT ' in l)
    return f'{out}:{2 * gprs}', layout

def simulate(cap, code_halves):
    code, _, halves = code_halves.partition(':')
    line = [l for l in open(os.path.join(cap, 'replay.sh')) if 'exec' in l][0]
    args = shlex.split(line.split('agx_run}"', 1)[1])[:-1]
    out, i = [], 0
    while i < len(args):
        if args[i] == '--dump': i += 2; continue
        if args[i] == '--shared': out += ['--tg-bytes', args[i + 1]]; i += 2; continue
        out.append(code if args[i] == 'code.bin' else args[i]); i += 1
    regs = max(int(open(os.path.join(cap, 'regs.txt')).read()), int(halves or 0))
    r = subprocess.run([a.dispatch] + out + ['--regs', str(regs)], capture_output=True, text=True, cwd=cap, timeout=1800)
    return json.loads(r.stdout.strip().splitlines()[-1]).get('time_us')

sims = []
bad = 0
for cap in a.caps:
    line = open(os.path.join(cap, 'replay.sh')).read().splitlines()[2]
    m = re.search(r'kernel (\S+) grid \((.*?)\) block \((.*?)\).*arguments (\[.*\])', line)
    kernel, block, vals = m.group(1), [int(x) for x in m.group(3).split(',')], ast.literal_eval(m.group(4))
    # the module of the kernel (some names are in several): the parameter count, then the captured code
    cands = [f for f, s in modules.items() if re.search(r'\.entry\s+' + re.escape(kernel) + r'\s*\(', s)]
    nparams = lambda f: len(re.search(r'\.entry\s+' + re.escape(kernel) + r'\s*\((.*?)\)', modules[f], re.S).group(1).split(','))
    cands = [f for f in cands if nparams(f) == len(vals)] or cands
    if len(cands) > 1:
        with tempfile.TemporaryDirectory() as t:
            captured = open(os.path.join(cap, 'code.bin'), 'rb').read()
            same = [f for f in cands if open(code_of(os.path.join(G, 'ptx', f), kernel, t)[0].split(':')[0], 'rb').read() == captured]
        cands = same or cands
    mod = cands[0]
    src = modules[mod]
    ent = re.search(r'\.entry\s+' + re.escape(kernel) + r'\s*\((.*?)\)', src, re.S).group(1)
    sizes = [SIZES[re.search(r'\.param\s+\.(\w+)', p).group(1)] for p in ent.split(',')]
    hexargs = [int(v).to_bytes(n, 'little').hex() for v, n in zip(vals, sizes)]
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(f'{t}/g'); os.makedirs(f'{t}/s')
        r = subprocess.run([SPEC, os.path.join(G, 'ptx', mod), kernel, '--block', ','.join(map(str, block))] + hexargs,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'{os.path.basename(cap.rstrip('/')):50s} not specialized: {r.stderr.strip()}'); continue
        sp = os.path.join(t, 's', mod); open(sp, 'w').write(r.stdout)
        gen, lg = code_of(os.path.join(G, 'ptx', mod), kernel, f'{t}/g')
        spc, ls = code_of(sp, kernel, f'{t}/s')
        if sorted(lg) != sorted(ls):
            print(f'{os.path.basename(cap.rstrip("/")):50s} not checked: layout {set(lg) - set(ls)} -> {set(ls) - set(lg)}'); continue
        out = subprocess.run(['python3', '-I', f'{G}/tools/capvariant.py', cap, gen, spc, '--b2b', str(a.b2b)],
                             capture_output=True, text=True).stdout.splitlines()
        res = [re.search(r'([\d.]+) us\s+hash (\w+)', l) for l in out]
        if len(res) != 2 or not all(res):
            print(f'{os.path.basename(cap.rstrip('/')):50s} FAILED {out}'); bad += 1; continue
        (tg, hg), (ts, hs) = [(float(x.group(1)), x.group(2)) for x in res]
        ok = hg == hs
        bad += not ok
        simtxt = ''
        if a.sim and ok:
            sg, ss = simulate(cap, gen), simulate(cap, spc)
            if sg and ss:
                sims.append((ts / tg, ss / sg))
                simtxt = f'  sim {sg:9.1f} -> {ss:9.1f} us {ss / sg - 1:+5.0%}'
        print(f'{os.path.basename(cap.rstrip('/')):50s} generic {tg:9.1f} us  specialized {ts:9.1f} us  {ts / tg - 1:+5.0%}  '
              f'{"same" if ok else "DIFFERENT " + hg + " " + hs}{simtxt}', flush=True)
print(f'{len(a.caps)} captures, {bad} mismatches or failures')
if sims:
    import statistics
    err = [abs(sr - hr) for hr, sr in sims]
    agree = sum((hr < 0.97) == (sr < 0.97) for hr, sr in sims)
    print(f'simulator: speedup ratio |sim - hw| median {statistics.median(err):.2f}, '
          f'agrees on a >3% speedup in {agree} of {len(sims)}')
