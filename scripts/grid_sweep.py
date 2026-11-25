#!/usr/bin/env python3
"""grid_sweep.py CAPDIR F1,F2,... [env...]: shrink the grid's largest dimension to each fraction; hw (agx_run_hw b2b) vs agx_dispatch"""
import json, os, re, shlex, subprocess, sys
cap, fr = sys.argv[1], [float(x) for x in sys.argv[2].split(',')]
env = dict(os.environ, AGX_THREADS='2', **dict(e.split('=', 1) for e in sys.argv[3:]))
H = '/scratch/agx-r5/zcopy/target/release/agx_run_hw'; S = os.environ.get('DISP', '/scratch/agx-r7/bin/agx_dispatch.merge')
sh = open(os.path.join(cap, 'replay.sh')).read()
args = shlex.split([l for l in sh.splitlines() if 'exec' in l][0].split('agx_run}"', 1)[1])[:-1]
out, i = [], 0
while i < len(args):
    if args[i] == '--dump': i += 2; continue
    out.append(args[i]); i += 1
regs = open(os.path.join(cap, 'regs.txt')).read().strip()
st = re.search(r'uses (\d+) bytes of stack', sh); stack = ['--stack', st.group(1)] if st else []
gi = out.index('--grid'); g = [int(x) for x in out[gi + 1].split(',')]; wi = out.index('--wg'); w = [int(x) for x in out[wi + 1].split(',')]
d = max(range(len(g)), key=lambda k: g[k] // w[k])
print('#', os.path.basename(cap)[:70], 'grid', g, 'wg', w, 'dim', d, flush=True)
for f in fr:
    g2 = list(g); g2[d] = max(w[d], int(round(g[d] / w[d] * f)) * w[d])
    a2 = list(out); a2[gi + 1] = ','.join(map(str, g2))
    hw = None
    for _ in range(2):
        r = subprocess.run([H] + [x if x != '--shared' else '--shared' for x in a2] + ['--regs', regs] + stack + ['--back-to-back', '300', '--timeout-ms', '10000'], cwd=cap, capture_output=True, text=True, timeout=900)
        try: hw = json.loads(r.stdout.strip().splitlines()[-1])['b2b_median_fw_ns'] / 1e3
        except Exception: hw = None; print('  hw error', (r.stdout + r.stderr)[-200:]); break
    sa = [x if x != '--shared' else '--tg-bytes' for x in a2]
    r = subprocess.run([S] + sa + ['--regs', regs] + stack + ['--launch-profile', 'runtime'], cwd=cap, capture_output=True, text=True, env=env, timeout=3600)
    try:
        j = json.loads(r.stdout.strip().splitlines()[-1]); s = j['time_us']
        print(f"  f {f:5.3f} grid {g2} wgs/core {j['busiest_core_wgs']:4d} waves {j['waves']:3d} hw {hw if hw is None else round(hw,1)} sim {s:.1f} err {100*(s-hw)/hw if hw else float('nan'):+.0f}%", flush=True)
    except Exception: print('  sim error', (r.stdout + r.stderr)[-200:])
