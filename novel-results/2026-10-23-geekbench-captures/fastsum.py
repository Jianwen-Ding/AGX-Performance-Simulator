"""Clock-robust GPU time of Geekbench runs: per launch configuration (kernel, grid, block, shared, arguments) the 10th
percentile of its firmware times x its number of launches, summed per kernel and in all. The plain sum of firmware
times moves with the GPU clock state (runs 12-14: the same code, convolve_2d's fastest launches 778-784 us, its median
790 us in one run and 1116 us in another).

fastsum.py RUN.jsonl [RUN2.jsonl ...] [--top N]
"""
import sys, json, collections, argparse

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+'); ap.add_argument('--top', type=int, default=20)
ap.add_argument('--pct', type=float, default=0.1)
a = ap.parse_args()

def load(f):
    g = collections.defaultdict(list)
    for l in open(f):
        r = json.loads(l)
        g[(r['kernel'], tuple(r['grid']), tuple(r['block']), r['shared'], tuple(r['params']))].append(r['ns'])
    k = collections.defaultdict(float)
    for key, v in g.items():
        v.sort()
        k[key[0]] += v[int(a.pct * (len(v) - 1))] * len(v)
    return k

runs = [load(f) for f in a.runs]
print('totals (s):', '  '.join(f'{sum(r.values()) / 1e9:.2f}' for r in runs))
names = sorted(runs[0], key=lambda n: -runs[0][n])[:a.top]
for n in names:
    print(f'{n:30s}' + ''.join(f' {r.get(n, 0) / 1e6:9.1f}' for r in runs) + ' ms')
