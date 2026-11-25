#!/usr/bin/env python3
"""probe_fetch_overlap_groups.py: how much does cold-code instruction fetch overlap with memory waits when G SIMD groups
of one workgroup run the same code? K blocks of straight-line code, each: a dependent device load to a new 64-byte line
(address += loaded value (0) + 16 words), wait, then F independent fmadd32. One workgroup on one core, warm (back to back),
hardware vs agx_dispatch (runtime profile). Env: GS="1 2 4 8" F="0 16 48" K=64 B2B=1000.
Prints per row: hw us, model us, and the model with AGX_FETCH_OVERLAP=0 / 1 (the bounds of the overlap factor)."""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tg_alu as P
e = P.e
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rust', 'target', 'release')
HW = os.environ.get('AGX_RUN_HW', '/scratch/agx-r5/zcopy/target/release/agx_run_hw')
DISP = os.environ.get('AGX_DISPATCH', os.path.join(REL, 'agx_dispatch')); RUN = os.path.join(REL, 'agx_run')
K = int(os.environ.get('K', '64')); B2B = os.environ.get('B2B', '1000')


def program(f):
    code = e('iadd r0l, 0, 0') + P.mov(11, 0) + P.mov(10, 0)
    for r in range(13, 29):
        code += P.mov(r, 0x3f800000)
    for k in range(K):
        code += e('device_load 0, i32, x, r10, u0_u1, r11, unsigned') + e('wait 0')
        code += e('iadd r11, r11, r10') + e('iadd r11, r11, 16')
        for j in range(f):
            r = 13 + j % 16
            code += e(f'fmadd32 r{r}, r{r}, r28, r27')
    return code + e('stop') + bytes(64)


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return json.loads(r.stdout.strip().splitlines()[-1])


for f in [int(x) for x in os.environ.get('F', '0 16 48').split()]:
    for g in [int(x) for x in os.environ.get('GS', '1 2 4 8').split()]:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'k.bin'); code = program(f); open(p, 'wb').write(code)
            common = [p, '--grid', f'{32 * g},1,1', '--wg', f'{32 * g},1,1', '--regs', '64', '--alloc', '0x100000=1048576',
                      '--uniform', '0=0x100000', '--uniform', '1=0']
            chk = subprocess.run([RUN] + common, capture_output=True, text=True)
            if '"instructions"' not in chk.stdout:
                print(json.dumps({'F': f, 'G': g, 'error': (chk.stdout + chk.stderr)[-200:]})); continue
            hwc = [HW] + common + ['--back-to-back', B2B, '--timeout-ms', '10000']
            subprocess.run(hwc, capture_output=True, text=True)
            hw = run(hwc)['b2b_median_fw_ns'] / 1e3
            sim = {}
            for ov in ('0.3', '0', '1'):
                sim[ov] = run([DISP] + common + ['--launch-profile', 'runtime'], dict(os.environ, AGX_FETCH_OVERLAP=ov))['time_us']
            dbg = subprocess.run([DISP] + common + ['--launch-profile', 'runtime'], capture_output=True, text=True, env=dict(os.environ, AGX_DEBUG_TIME='1')).stderr
            line = [l for l in dbg.splitlines() if l.startswith('time:')][-1]
            core, fetch = [float(x) for x in line.split()[2:5:2]]
            need = (hw - sim['0']) / ((sim['1'] - sim['0']) or 1)
            print(json.dumps({'F': f, 'G': g, 'code_kb': round(len(code) / 1024, 1), 'hw_us': round(hw, 2), 'model_us': round(sim['0.3'], 2),
                              'ov0_us': round(sim['0'], 2), 'ov1_us': round(sim['1'], 2), 'core_us': round(core / 1e3, 2), 'fetch_us': round(fetch / 1e3, 2),
                              'needed_overlap': round(need, 2)}), flush=True)
