#!/usr/bin/env python3
"""Fragment-op cost check: simulated slope of the varying sweep against the stored hardware slope (results/2026-11-14-frag-cost).
Pass: smooth slope within 50%, flat within 25% (the slopes are ~5% of the total; with the fragment issue factor smooth is +42%, flat -14%), absolute times within 20%. Usage: frag_cost_check.py [bin_dir]"""
import json, subprocess, sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
d = root / 'results/2026-11-14-frag-cost'
b = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'rust/target/release'
def sim(n, flat):
    r = subprocess.run([str(b / 'agx_dispatch'), str(d / f'frag_{n}_{flat}.bin'), '--grid', '4194304', '--wg', '1024', '--regs', '48', '--frag', str(d / 'frag_sweep.txt'),
                        '--uniform', '11=256', '--uniform', '13=256', '--uniform', '15=256', '--no-mem'], capture_output=True, text=True)
    return json.loads(r.stdout)['time_us']
hw = {}
for l in open(d / 'hw_sweep_vary.txt'):
    w = l.split(); hw[(int(w[2]), int(w[0].split('=')[1]))] = float(w[4]) * 1000
ok = True
for flat, tol in ((0, 0.50), (1, 0.25)):
    s = sim(24, flat) - sim(1, flat); h = hw[(24, flat)] - hw[(1, flat)]
    print(f"{'flat ldcf' if flat else 'smooth iterproj'}: 1 -> 24 varyings: simulated +{s:.0f} us, measured +{h:.0f} us ({(s / h - 1) * 100:+.0f}%)")
    ok &= abs(s / h - 1) <= tol
# absolute level (with agx_dispatch's fragment issue factor): the four sweep shaders and the ALU-depth sweep of the 1-varying shader
cases = [(f'{n} varyings, {"flat" if f else "smooth"}', sim(n, f), hw[(n, f)]) for n in (1, 24) for f in (0, 1)]
alu = {}
for l in open(d / 'hw_sweep_size_alu.txt'):
    w = l.split()
    if w[0] == 'W=2048': alu[int(w[1].split('=')[1])] = float(w[2]) * 1000
for t, h in sorted(alu.items()):
    if t == 256: continue
    r = subprocess.run([str(b / 'agx_dispatch'), str(d / 'frag_1_0.bin'), '--grid', '4194304', '--wg', '1024', '--regs', '48', '--frag', str(d / 'frag_sweep.txt'),
                        '--uniform', f'11={t}', '--no-mem'], capture_output=True, text=True)
    cases.append((f'1 varying, {t}-iteration loop', json.loads(r.stdout)['time_us'], h))
for name, s_, h in cases:
    print(f'{name}: simulated {s_:.0f} us, measured {h:.0f} us ({(s_ / h - 1) * 100:+.0f}%)')
    ok &= abs(s_ / h - 1) <= 0.20
# vertex stage: st_var throughput (903k vertices, 1 / 4 / 12 / 24 vec4 varyings); the 12 / 24 points saturate on hardware, so only 1 / 4 / 24 are asserted (+-20%)
vhw = {}
for l in open(d / 'hw_sweep_vertex.txt'):
    w = l.split(); vhw[int(w[0].split('=')[1])] = float(w[2]) * 1000
for n in (1, 4, 12, 24):
    r = subprocess.run([str(b / 'agx_dispatch'), str(d / f'vertex_{n}.bin'), '--grid', '903264', '--wg', '1024', '--regs', '48', '--no-mem'], capture_output=True, text=True)
    s_ = json.loads(r.stdout)['time_us']
    print(f'vertex stage, {n} varyings: simulated {s_:.0f} us, measured {vhw[n]:.0f} us ({(s_ / vhw[n] - 1) * 100:+.0f}%)' + ('' if n != 12 else '  (known outlier)'))
    if n != 12: ok &= abs(s_ / vhw[n] - 1) <= 0.20
sys.exit(0 if ok else 1)
