"""ALU throughput against occupancy: a dependent loop body (CODE, from probes/divperf/dv.ptx: a signed division by
a uniform and an add per iteration) run with ITERS_A and ITERS_B iterations; the difference per iteration in cycles of
the busiest core, on the GPU (agx_run_hw --back-to-back, firmware times) and in agx_dispatch, for grids that put 1
to 40 SIMD groups on the busiest core (the simulator's cluster split, which matches the measured placement).

aluocc.py [--code dv.bin] [--b2b N]
"""
import json, subprocess, argparse, math

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='/scratch/geekbench-extract/probes/divperf/dv.bin')
ap.add_argument('--b2b', type=int, default=200)
ap.add_argument('--regs', default='30', help='register allocation in 16-bit halves')
ap.add_argument('--iters', default='256,2048')
ap.add_argument('--configs', default='38x32,38x64,19x128,38x128,152x32,304x32,76x128,152x128,152x256,304x256')
ap.add_argument('--run-hw', default='/scratch/geekbench-extract/cargo-target/release/agx_run_hw')
ap.add_argument('--dispatch', default='/scratch/agxsim-target/release/agx_dispatch')
a = ap.parse_args()
ia, ib = map(int, a.iters.split(','))

def busiest(n):
    share = math.ceil(n / 4)
    return math.ceil(share / 9)

def common(g, b, it):
    return ['--grid', f'{g * b},1,1', '--wg', f'{b},1,1', '--regs', a.regs, '--uniform', '0=268435456', '--uniform', '1=0',
            '--uniform', '2=7', '--uniform', f'3={it}', '--uniform', f'4={g}', '--uniform', '5=1', '--uniform', '6=1',
            '--alloc', '0x10000000=4194304']

def hw(g, b, it):
    r = subprocess.run([a.run_hw, a.code] + common(g, b, it) + ['--back-to-back', str(a.b2b), '--timeout-ms', '10000'],
                       capture_output=True, text=True, timeout=600)
    return json.loads(r.stdout.strip().splitlines()[-1])['b2b_median_fw_ns'] / 1e3

def sim(g, b, it):
    r = subprocess.run([a.dispatch, a.code] + common(g, b, it), capture_output=True, text=True, timeout=3600)
    return json.loads(r.stdout.strip().splitlines()[-1])['time_us']

print(f'{"config":10s} {"groups":>6s} {"hw cyc/it":>10s} {"sim cyc/it":>10s} {"err":>6s}')
for c in a.configs.split(','):
    g, b = map(int, c.split('x'))
    groups = busiest(g) * math.ceil(b / 32)
    h = (hw(g, b, ib) - hw(g, b, ia)) / (ib - ia) * 1398
    s = (sim(g, b, ib) - sim(g, b, ia)) / (ib - ia) * 1398
    print(f'{c:10s} {groups:6d} {h:10.1f} {s:10.1f} {s / h - 1:+6.0%}', flush=True)
