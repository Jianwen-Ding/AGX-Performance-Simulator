#!/usr/bin/env python3
"""probe_fetch_scaling.py: instruction fetch of long straight-line code. One workgroup of G SIMD groups runs a straight line
of `iadd` over code of growing size once, warm (back to back), against the model's prediction. Known-good encodings only,
each program run through agx_run first. Results as JSON lines."""
import json, os, re, subprocess, sys, tempfile
RUN = os.environ.get('AGX_RUN', 'agx_run'); HW = os.environ.get('AGX_RUN_HW', 'agx_run_hw'); DISP = os.environ.get('AGX_DISPATCH', 'agx_dispatch')
ASM = os.environ.get('AGX_ASM', 'agx_asm')
def e(text):
    return bytes.fromhex(subprocess.run([ASM], input=text + '\n', capture_output=True, text=True, check=True).stdout.split()[0])
stop = e('stop')
sizes = [int(x) for x in os.environ.get('SIZES', '4096 16384 65536 262144 1048576 4194304').split()]
wgs = [int(x) for x in os.environ.get('WGS', '32 384').split()]
for wg in wgs:
    for size in sizes:
        # independent 8-byte iadds: fetch, not dependencies, should bound them
        ins = [e(f'iadd r{5 + (i % 8)}, r{5 + ((i + 3) % 8)}, {1 + i % 7}') for i in range(8)]
        bytes_each = sum(len(x) for x in ins)
        code = b''.join(ins) * (size // bytes_each) + stop
        n = (size // bytes_each) * len(ins)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'code.bin'); open(p, 'wb').write(code)
            common = [p, '--grid', f'{wg},1,1', '--wg', f'{wg},1,1']
            r = subprocess.run([RUN] + common + ['--regs', '40'], capture_output=True, text=True)
            if '"instructions"' not in r.stdout: print(json.dumps({'size': size, 'error': (r.stdout + r.stderr)[-200:]})); continue
            h = subprocess.run([HW] + common + ['--regs', '40', '--back-to-back', '300'], capture_output=True, text=True)
            mhw = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
            m = subprocess.run([DISP] + common + ['--regs', '40'], capture_output=True, text=True)
            pred = json.loads(m.stdout)['time_us'] if m.stdout.startswith('{') else None
            hw = int(mhw.group(1)) / 1000 if mhw else None
            print(json.dumps({'wg': wg, 'code_kb': size // 1024, 'instructions': n, 'hw_us': hw, 'pred_us': pred, 'ratio': round(pred / hw, 3) if pred and hw else None,
                              'hw_ns_per_instr_per_group': round(hw * 1000 / n, 3) if hw else None, 'err': None if hw else (h.stdout + h.stderr)[-160:]}), flush=True)
