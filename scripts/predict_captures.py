#!/usr/bin/env python3
"""predict_captures.py CAPTURE_DIR PROFILE.jsonl SKIP [agx_dispatch options]: for each launch AGX_RUNTIME_CAPTURE saved in
CAPTURE_DIR, the model's prediction (agx_dispatch over the captured memory) against the GPU time AGX_RUNTIME_PROFILE measured
for that same launch: the SKIP-th launch of its kernel, whose grid must match. Prints one JSON line per launch."""
import json, os, re, shlex, subprocess, sys, time, collections
DISPATCH = os.environ.get('AGX_DISPATCH', 'agx_dispatch')
cap, prof, skip = sys.argv[1], sys.argv[2], int(sys.argv[3]); extra = sys.argv[4:]
# KERNEL_REGS: a JSON file of each kernel's register count in 16-bit halves (else 256, all of them)
REGS = json.load(open(os.environ['KERNEL_REGS'])) if 'KERNEL_REGS' in os.environ else {}
by = collections.defaultdict(list)
for l in open(prof):
    r = json.loads(l); by[r['kernel']].append(r)
for d in sorted(os.listdir(cap)):
    rs = os.path.join(cap, d, 'replay.sh')
    if not os.path.exists(rs): continue
    text = open(rs).read()
    kernel = re.search(r'# kernel (\S+)', text).group(1)
    line = [l for l in text.splitlines() if 'exec ' in l][0]
    args = [a for a in shlex.split(line.split('exec ', 1)[1])[1:] if a != '$@']
    out, i = [], 0
    while i < len(args):
        if args[i] == '--dump': i += 2; continue
        out.append('--tg-bytes' if args[i] == '--shared' else args[i]); i += 1
    grid = [int(x) for x in out[out.index('--grid') + 1].split(',')]; wg = [int(x) for x in out[out.index('--wg') + 1].split(',')]
    n = int(re.search(r'-cap\d+-', d) is not None) and skip
    m = by[kernel][n] if n < len(by[kernel]) else None
    match = m and m['grid'][0] * m['block'][0] == grid[0] and m['block'][0] == wg[0]
    t = time.time()
    halves = REGS.get(kernel, 256)
    p = subprocess.run([DISPATCH] + out + ['--regs', str(halves)] + extra, cwd=os.path.join(cap, d), capture_output=True, text=True, errors='replace', env=dict(os.environ, AGX_EXPERIMENTAL_RET='buffer-pop2'))
    res = {}
    try: res = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception: res = {'error': (p.stdout + p.stderr)[-200:]}
    pred = res.get('time_us')
    meas = m['ns'] / 1000 if m else None
    print(json.dumps({'kernel': kernel.replace('kernel_gpu_', ''), 'threads': grid[0], 'matched': bool(match), 'measured_us': meas, 'predicted_us': pred,
                      'ratio': round(pred / meas, 3) if pred and meas else None, 'wgs_per_core': res.get('wgs_per_core'), 'waves': res.get('waves'), 'simulated_waves': res.get('simulated_waves'), 'sim_s': round(time.time() - t, 1), 'error': res.get('error')}), flush=True)
