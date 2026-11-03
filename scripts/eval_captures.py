#!/usr/bin/env python3
"""eval_captures.py [--agx-dispatch-arg X ...] TABLE...: the model's error on launches with a known warm GPU time.
A TABLE is CAPTURE_ROOT:CHECK_FILE, where CHECK_FILE has capcheck.py's lines "KERNEL hw 12.3 us ...": each capture directory
under the root (PID-capN-KERNEL) with a matching line is simulated. Prints one line per launch and the error summary."""
import json, os, re, shlex, statistics as st, subprocess, sys
D = os.environ.get('AGX_DISPATCH', '/scratch/agxsim-target/release/agx_dispatch')
args = sys.argv[1:]; extra = ['--launch-profile', 'zluda']; tables = []
while args:
    a = args.pop(0)
    if a == '--agx-dispatch-arg': extra.append(args.pop(0))
    else: tables.append(a)
def sim(d):
    line = [l for l in open(os.path.join(d, 'replay.sh')) if 'agx_run}"' in l][0]
    a = shlex.split(line.split('agx_run}"', 1)[1])[:-1]; out, i = [], 0
    while i < len(a):
        if a[i] == '--dump': i += 2; continue
        if a[i] == '--shared': out += ['--tg-bytes', a[i + 1]]; i += 2; continue
        out.append(a[i]); i += 1
    regs = open(os.path.join(d, 'regs.txt')).read().strip()
    p = subprocess.run([D] + out + ['--regs', regs] + extra, cwd=d, capture_output=True, text=True, timeout=900)
    if not p.stdout: raise RuntimeError(p.stderr.strip()[:200])
    return json.loads(p.stdout)['time_us']
errs = []
for t in tables:
    root, check = t.split(':'); hw = {}
    for l in open(check):
        m = re.match(r'(\S+)\s+hw\s+([\d.]+) us', l)
        if m: hw.setdefault(m.group(1), float(m.group(2)))
    for n in sorted(os.listdir(root)):
        k = re.sub(r'^\d+-cap\d+-', '', n)
        if k not in hw or not os.path.exists(os.path.join(root, n, 'replay.sh')): continue
        try:
            s = sim(os.path.join(root, n)); e = 100 * (s - hw[k]) / hw[k]; errs.append(e)
            print(f'{k[:40]:40s} hw {hw[k]:9.1f} sim {s:9.1f} {e:+6.0f}%', flush=True)
        except Exception as x:
            print(f'{k[:40]:40s} failed: {str(x)[:80]}', flush=True)
a = [abs(e) for e in errs]
if a: print(f'n={len(a)} median|err| {st.median(a):.0f}% mean|err| {st.mean(a):.0f}% within25 {sum(x <= 25 for x in a)} signed-median {st.median(errs):+.0f}%')
