"""Simulator vs hardware on captured launches (AGX_RUNTIME_CAPTURE + AGX_RUNTIME_PROFILE of the same run).

capcheck.py CAPTURE_DIR PROFILE.jsonl [--dispatch PATH] [--only KERNEL] [--hw [--b2b N] | --hw-from FILE]
            [--sim-arg ARG ...]

--hw times each capture on the GPU back to back (agx_run_hw --back-to-back N, the median of the second half): with
N = 1000 (the default) the GPU's clock has ramped up by then, which a few launches do not do (30 back to back measured
the Q8_0 matrix-vector kernel of llama.cpp at 67 us, 1000 at 26 us). --sim-arg passes options to agx_dispatch, e.g.
`--sim-arg=--launch-profile --sim-arg=runtime` for the launch costs of the CUDA driver's commands.

For each capture (code, real memory as the launch found it, uniforms, grid): agx_dispatch with that memory mapped,
against the launch's firmware time from the profile (the line with the capture's command number). Prints one line per
launch and the median absolute error.
"""
import sys, os, re, json, shlex, subprocess, argparse, statistics

ap = argparse.ArgumentParser()
ap.add_argument('captures'); ap.add_argument('profile')
ap.add_argument('--dispatch', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../rust/target/release/agx_dispatch'))
ap.add_argument('--only')
ap.add_argument('--skip', help='regex: kernels to leave out')
ap.add_argument('--timeout', type=int, default=900)
ap.add_argument('--hw', action='store_true', help='time each capture on the GPU (agx_run_hw --back-to-back) instead of '
                'using the profile: the same launch with the same data, at a warm clock')
ap.add_argument('--run-hw', default=os.environ.get('AGX_RUN_HW', '/scratch/ptx-frontend-build/release/agx_run_hw'))
ap.add_argument('--b2b', type=int, default=1000)
ap.add_argument('--sim-arg', action='append', default=[])
ap.add_argument('--hw-repeat', type=int, default=1, help='with --hw: the median of this many back-to-back runs (a tiny kernel varies from one run to the next: set_rows 3.9-6.5 us, rope 5.6-8.7)')
ap.add_argument('--hw-idle-cmd', help='with --hw: a shell command that prints nothing while no other process uses the GPU '
                '(e.g. a list of the processes holding /dev/dri/renderD128 open); each timing waits for it to be empty and '
                'is taken again when it was not empty afterwards (timings beside other GPU work were up to 1.9x high)')
ap.add_argument('--hw-only', action='store_true', help='with --hw: print the hardware times only (no simulation)')
ap.add_argument('--hw-from', help='hardware times from an earlier --hw output of capcheck (no GPU runs)')
a = ap.parse_args()

# The captured launch itself ran after the GPU idled for the capture (its clock low): its time is that of the other
# launches with the same kernel, geometry and arguments (median), else the same kernel and geometry.
lines = {}
by_args, by_geom = {}, {}
for line in open(a.profile):
    r = json.loads(line)
    lines[(r['kernel'], r['command'])] = r
    by_args.setdefault((r['kernel'], tuple(r['grid']), tuple(r['block']), r['shared'], tuple(r['params'])), []).append(r)
    by_geom.setdefault((r['kernel'], tuple(r['grid']), tuple(r['block']), r['shared']), []).append(r)

def warm(r):
    for group in (by_args[(r['kernel'], tuple(r['grid']), tuple(r['block']), r['shared'], tuple(r['params']))],
                  by_geom[(r['kernel'], tuple(r['grid']), tuple(r['block']), r['shared'])]):
        ns = sorted(x['ns'] for x in group if x['command'] != r['command'])
        if ns:
            return ns[len(ns) // 2] / 1e3, len(ns)
    return None, 0

errs = []
for d in sorted(os.listdir(a.captures)):
    p = os.path.join(a.captures, d)
    m = re.match(r'\d+-cap\d+-(.+)$', d)
    if not m or not os.path.exists(os.path.join(p, 'command.txt')): continue
    kernel = m.group(1)
    if a.only and kernel != a.only: continue
    if a.skip and re.search(a.skip, kernel): continue
    command = int(open(os.path.join(p, 'command.txt')).read())
    regs = open(os.path.join(p, 'regs.txt')).read().strip()
    rec = lines.get((kernel, command))
    hw, n_warm = warm(rec) if rec else (None, 0)
    cold = rec['ns'] / 1e3 if rec else None
    sh = open(os.path.join(p, 'replay.sh')).read()
    line = [l for l in sh.splitlines() if 'exec' in l][0]
    args = shlex.split(line.split('agx_run}"', 1)[1])[:-1]   # drop "$@"
    out, i = [], 0
    while i < len(args):
        if args[i] == '--dump':
            i += 2; continue
        if args[i] == '--shared':
            out += ['--tg-bytes', args[i + 1]]; i += 2; continue
        out.append(args[i]); i += 1
    cmd = [a.dispatch] + out + ['--regs', regs] + a.sim_arg
    st_sim = re.search(r'the kernel uses (\d+) bytes of stack per thread', sh)
    if st_sim:
        cmd += ['--stack', st_sim.group(1)]
    if a.hw_from:
        prev = {}
        for l in open(a.hw_from):
            mm = re.match(r'(\S+)\s+hw\s+([\d.]+) us', l)
            if mm: prev.setdefault(mm.group(1), float(mm.group(2)))
        if kernel not in prev:
            continue
        hw, n_warm = prev[kernel], 0
    if a.hw:
        hcmd = [a.run_hw] + [x if x != '--tg-bytes' else '--shared' for x in out] + ['--regs', regs, '--back-to-back', str(a.b2b), '--timeout-ms', '10000']
        # a kernel with a stack needs it on the GPU (the CUDA driver notes it in replay.sh; agx_run has no --stack): without
        # it the GPU faults ("Bad address") at the first stack access
        st = re.search(r'the kernel uses (\d+) bytes of stack per thread', sh)
        if st:
            hcmd += ['--stack', st.group(1)]
        try:
            runs = []
            for _ in range(max(1, a.hw_repeat)):
                for _attempt in range(50):
                    if a.hw_idle_cmd:
                        import time
                        while subprocess.run(a.hw_idle_cmd, shell=True, capture_output=True, text=True).stdout.strip():
                            time.sleep(5)
                    hr = subprocess.run(hcmd, capture_output=True, text=True, cwd=p, timeout=600)
                    if not a.hw_idle_cmd or not subprocess.run(a.hw_idle_cmd, shell=True, capture_output=True, text=True).stdout.strip():
                        break
                hj = json.loads(hr.stdout.strip().splitlines()[-1])
                runs.append(hj['b2b_median_fw_ns'] / 1e3)
            hw, n_warm = statistics.median(runs), a.b2b // 2
        except Exception as e:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import gpu_safety
            gpu_safety.record(f'capcheck --hw {kernel}: {(hr.stdout + hr.stderr)[-120:] if "hr" in dir() else e}')
            print(f'{kernel:30s} HW FAILED {(hr.stdout + hr.stderr)[-200:] if "hr" in dir() else e}', flush=True); continue
    if a.hw and a.hw_only:
        print(f'{kernel:30s} hw {hw:9.1f} us (n {n_warm:5d}, runs {" ".join(f"{x:.1f}" for x in runs)})', flush=True); continue
    try:
        # A real program may read just outside its buffers (srad_cuda_1 reads 0x1c4 bytes below one): the GPU reads
        # whatever page is there, the capture has no data for it. Map the missing 16 KB page as zeros and retry.
        for _ in range(16):
            # calls (Cycles): the measured return model, as predict_captures.py
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=p, timeout=a.timeout,
                               env={'AGX_EXPERIMENTAL_RET': 'buffer-pop2', **os.environ})
            um = re.search(r'unmapped address (0x[0-9a-f]+)', r.stdout + r.stderr)
            if not um: break
            cmd += ['--alloc', f'{int(um.group(1), 16) & ~0x3fff:#x}=16384']
        j = json.loads(r.stdout.strip().splitlines()[-1])
        sim = j.get('time_us')
    except Exception as e:
        print(f'{kernel:30s} SIM FAILED {e} {r.stdout[-200:] if "r" in dir() else ""}'); continue
    if sim is None:
        print(f'{kernel:30s} SIM ERROR {json.dumps(j)[:200]}'); continue
    if hw is None:
        print(f'{kernel:30s} sim {sim:9.1f} us  (no other launch of it in the profile)'); continue
    e = (sim - hw) / hw
    errs.append(abs(e))
    print(f'{kernel:30s} hw {hw:9.1f} us (n {n_warm:5d}, captured {cold if cold is not None else 0:9.1f})  sim {sim:9.1f} us  err {e:+6.0%}   wgs/core {j.get("busiest_core_wgs")} waves {j.get("waves")}', flush=True)
if errs:
    print(f'launches {len(errs)}  median |err| {statistics.median(errs):.0%}  within 10%: {sum(e <= 0.1 for e in errs)}  within 25%: {sum(e <= 0.25 for e in errs)}')
