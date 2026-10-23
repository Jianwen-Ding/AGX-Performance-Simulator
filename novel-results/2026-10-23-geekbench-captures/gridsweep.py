"""A captured launch with fewer workgroups (the grid and its threadgroups uniform changed) on the GPU (back to back,
firmware) and in agx_dispatch: how the time grows with the workgroups per core.

gridsweep.py CAPDIR TG_UNIFORM WG1,WG2,... [--code CODE.bin] [--regs HALVES]
"""
import sys, os, json, shlex, subprocess, argparse
ap = argparse.ArgumentParser()
ap.add_argument('cap'); ap.add_argument('tgu', type=int); ap.add_argument('wgs')
ap.add_argument('--code'); ap.add_argument('--regs')
ap.add_argument('--b2b', type=int, default=100)
ap.add_argument('--sim-extra', default='', help='extra agx_dispatch options, space separated')
ap.add_argument('--no-hw', action='store_true')
a = ap.parse_args()
cap = os.path.abspath(a.cap)
line = [l for l in open(os.path.join(cap, 'replay.sh')) if 'exec' in l][0]
args = shlex.split(line.split('agx_run}"', 1)[1])[:-1]
regs = a.regs or open(os.path.join(cap, 'regs.txt')).read().strip()
code = os.path.abspath(a.code) if a.code else os.path.join(cap, 'code.bin')
wg = [int(x) for x in args[args.index('--wg') + 1].split(',')]
def build(n, hw):
    out, i = [], 0
    while i < len(args):
        x = args[i]
        if x == '--dump': i += 2; continue
        if x == '--map': out += ['--map', args[i + 1].split('=')[0] + '=' + os.path.join(cap, args[i + 1].split('=', 1)[1])]; i += 2; continue
        if x == '--grid': out += ['--grid', f'{n * wg[0]},1,1']; i += 2; continue
        if x == '--uniform' and args[i + 1].startswith(f'{a.tgu}='): out += ['--uniform', f'{a.tgu}={n}']; i += 2; continue
        if x == '--shared' and not hw: out += ['--tg-bytes', args[i + 1]]; i += 2; continue
        out.append(code if x == 'code.bin' else x); i += 1
    return out + ['--regs', regs]
first = True
for n in [int(x) for x in a.wgs.split(',')]:
    hwc = ['/scratch/geekbench-extract/cargo-target/release/agx_run_hw'] + build(n, True) + ['--back-to-back', str(a.b2b), '--timeout-ms', '10000']
    if a.no_hw:
        h = float('nan')
    else:
        if first:
            subprocess.run(hwc, capture_output=True, text=True); first = False
        h = json.loads(subprocess.run(hwc, capture_output=True, text=True).stdout.strip().splitlines()[-1])['b2b_median_fw_ns'] / 1e3
    j = json.loads(subprocess.run(['/scratch/agxsim-target/release/agx_dispatch'] + build(n, False) + a.sim_extra.split(), capture_output=True, text=True, cwd=cap).stdout.strip().splitlines()[-1])
    print(f'wgs {n:5d}  busiest {j["busiest_core_wgs"]:4d} waves {j["waves"]}  hw {h:8.1f} us  sim {j["time_us"]:8.1f} us  {j["time_us"] / h - 1:+5.0%}', flush=True)
