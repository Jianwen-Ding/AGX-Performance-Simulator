"""Which instruction classes go with the model's errors: dynamic class mix of each captured launch (agx_cost
--annotate on one workgroup with the real memory) against log(hw / sim) from a capcheck output.

classmix.py CAPTURES CAPCHECK_OUTPUT
"""
import sys, os, re, json, math, shlex, subprocess, collections

caps, check = sys.argv[1], sys.argv[2]
err = {}
for l in open(check):
    m = re.match(r'(\S+)\s+hw\s+([\d.]+) us.*sim\s+([\d.]+) us', l)
    if m: err.setdefault(m.group(1), math.log(float(m.group(2)) / float(m.group(3))))
rows = []
for d in sorted(os.listdir(caps)):
    m = re.match(r'\d+-cap\d+-(.+)$', d)
    if not m or m.group(1) not in err: continue
    p = os.path.join(caps, d)
    line = [l for l in open(os.path.join(p, 'replay.sh')) if 'exec' in l][0]
    args = shlex.split(line.split('agx_run}"', 1)[1])[:-1]
    out, i = [], 0
    while i < len(args):
        if args[i] == '--dump': i += 2; continue
        if args[i] == '--shared': out += ['--tg-bytes', args[i + 1]]; i += 2; continue
        out.append(args[i]); i += 1
    cmd = ['/scratch/agxsim-target/release/agx_cost'] + out + ['--regs', open(os.path.join(p, 'regs.txt')).read().strip(), '--annotate', '--allow-unknown']
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=p, timeout=600)
    mix = collections.Counter()
    for l in r.stdout.splitlines():
        f = l.split()
        if len(f) >= 5 and f[0].isdigit():
            mix[f[4]] += int(f[1])
    tot = sum(mix.values())
    if not tot: continue
    rows.append((m.group(1), err[m.group(1)], {k: v / tot for k, v in mix.items()}, tot))
classes = sorted({c for _, _, mx, _ in rows for c in mx}, key=lambda c: -sum(mx.get(c, 0) for _, _, mx, _ in rows))
print(f'{len(rows)} launches; correlation of each class share with log(hw/sim) (positive: the model is too fast where the class is frequent)')
ys = [e for _, e, _, _ in rows]
my = sum(ys) / len(ys)
for c in classes[:25]:
    xs = [mx.get(c, 0) for _, _, mx, _ in rows]
    mx_ = sum(xs) / len(xs)
    sxy = sum((x - mx_) * (y - my) for x, y in zip(xs, ys)); sxx = sum((x - mx_) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0: continue
    print(f'{c:18s} mean share {mx_:6.1%}  corr {sxy / math.sqrt(sxx * syy):+.2f}')
json.dump([(k, e, mx, t) for k, e, mx, t in rows], open('/scratch/geekbench-extract/captures/classmix.json', 'w'))
