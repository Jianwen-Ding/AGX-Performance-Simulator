"""Time variants of a captured launch on the GPU: the capture's memory, uniforms and grid, another code.bin.

capvariant.py CAPDIR CODE.bin[:REGS] [CODE2.bin ...] [--b2b N]
REGS: the code's register count in 16-bit halves (2 x its 32-bit GPRs), when it needs more than the captured code.
Prints per code: back-to-back median firmware time (us) and a hash of every region after one plain run (to compare
the results of variants that should compute the same).
"""
import sys, os, json, shlex, subprocess, hashlib, tempfile, argparse

ap = argparse.ArgumentParser()
ap.add_argument('cap'); ap.add_argument('codes', nargs='+')
ap.add_argument('--b2b', type=int, default=40)
ap.add_argument('--run-hw', default='/scratch/geekbench-extract/cargo-target/release/agx_run_hw')
a = ap.parse_args()
cap = os.path.abspath(a.cap)
line = [l for l in open(os.path.join(cap, 'replay.sh')) if 'exec' in l][0]
args = shlex.split(line.split('agx_run}"', 1)[1])[:-1]
regs = open(os.path.join(cap, 'regs.txt')).read().strip()
warmed = False
for code in a.codes:
    code, _, own = code.partition(':')
    code = os.path.abspath(code)
    halves = str(max(int(regs), int(own))) if own else regs
    with tempfile.TemporaryDirectory() as t:
        out, dumps, i = [], [], 0
        while i < len(args):
            x = args[i]
            if x == '--dump':
                va, rest = args[i + 1].split('=', 1)
                n, _f = rest.split(':', 1)
                f = os.path.join(t, f'd{len(dumps)}.bin')
                dumps.append(f); out += ['--dump', f'{va}={n}:{f}']; i += 2; continue
            if x == '--map':
                va, f = args[i + 1].split('=', 1)
                out += ['--map', f'{va}={os.path.join(cap, f)}']; i += 2; continue
            out.append(code if x == 'code.bin' else x); i += 1
        base = [a.run_hw] + out + ['--regs', halves, '--timeout-ms', '10000']
        r1 = subprocess.run(base, capture_output=True, text=True, timeout=600)
        h = hashlib.sha256(b''.join(open(f, 'rb').read() for f in dumps if os.path.exists(f))).hexdigest()[:16]
        each = ' '.join(hashlib.sha256(open(f, 'rb').read()).hexdigest()[:8] if os.path.exists(f) else '-' for f in dumps)
        nodump = [x for j, x in enumerate(base) if not (x == '--dump' or (j > 0 and base[j - 1] == '--dump'))]
        if not warmed:
            # the GPU clock is low after the idle time of compiling: the first code would be timed slow
            subprocess.run(nodump + ['--back-to-back', '400'], capture_output=True, text=True, timeout=600)
            warmed = True
        r2 = subprocess.run(nodump + ['--back-to-back', str(a.b2b)], capture_output=True, text=True, timeout=600)
        try:
            us = json.loads(r2.stdout.strip().splitlines()[-1])['b2b_median_fw_ns'] / 1e3
            print(f'{os.path.basename(code):30s} {us:10.1f} us  hash {h}  regions {each}', flush=True)
        except Exception:
            print(f'{os.path.basename(code):30s} FAILED {(r1.stdout + r2.stdout + r2.stderr)[-200:]}', flush=True)
