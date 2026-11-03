#!/usr/bin/env python3
"""probe_stack_cost.py: the cost of stack (scratch) memory. An LLVM-compiled kernel makes N volatile private loads and stores, over a
frame of FRAME bytes, against the model's prediction, warm (back to back). A real call's register saves and shading's structures live
there. Results as JSON lines."""
import json, os, re, subprocess, tempfile
B = os.environ.get('LLVM_BIN', '/scratch/llvm-agx-main-build/bin')
RUN, HW, DISP = os.environ.get('AGX_RUN', 'agx_run'), os.environ.get('AGX_RUN_HW', 'agx_run_hw'), os.environ.get('AGX_DISPATCH', 'agx_dispatch')
def kernel(n, frame, mode):
    w = frame // 4
    body = []
    for i in range(n):
        idx = (i * 37) % w
        g = f'%g{i} = getelementptr [{w} x i32], ptr addrspace(5) %a, i32 0, i32 {idx}'
        if mode == 'store': body += [g, f'store volatile i32 {i}, ptr addrspace(5) %g{i}']
        else: body += [g, f'%v{i} = load volatile i32, ptr addrspace(5) %g{i}']
    return f'target triple = "agx-unknown-unknown"\ndefine agx_kernel void @k() {{\n  %a = alloca [{w} x i32], addrspace(5)\n  ' + '\n  '.join(body) + '\n  ret void\n}\n'
for mode in ('load', 'store'):
    for frame in (256, 4096):
        for wg in (32, 384):
            n = 2000
            with tempfile.TemporaryDirectory() as d:
                ll, o, code = (os.path.join(d, x) for x in ('k.ll', 'k.o', 'code.bin'))
                open(ll, 'w').write(kernel(n, frame, mode))
                subprocess.run([B + '/llc', '-mtriple=agx', '-mcpu=agx-g14c', '-O0', '-filetype=obj', ll, '-o', o], check=True, capture_output=True)
                subprocess.run([B + '/llvm-objcopy', '-O', 'binary', '--only-section=.text', o, code], check=True)
                asm = subprocess.run([B + '/llc', '-mtriple=agx', '-mcpu=agx-g14c', '-O0', ll, '-o', '-'], capture_output=True, text=True).stdout
                ops = len(re.findall(r'\bstack_(?:load|store)\b', asm)); stack = int(re.search(r'stack size: (\d+)', asm).group(1))
                common = [code, '--grid', f'{wg},1,1', '--wg', f'{wg},1,1', '--regs', '256']
                r = subprocess.run([RUN] + common, capture_output=True, text=True)
                if '"instructions"' not in r.stdout: print(json.dumps({'mode': mode, 'error': r.stdout[-160:]})); continue
                h = subprocess.run([HW] + common + ['--stack', str(stack), '--back-to-back', '300'], capture_output=True, text=True)
                m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout); hw = int(m.group(1)) / 1000 if m else None
                p = subprocess.run([DISP] + common + ['--alloc', '0x100000=16', '--all-workgroups'], capture_output=True, text=True); pr = json.loads(p.stdout)['time_us'] if p.stdout.startswith('{') else None
                ins = json.loads(r.stdout)['instructions']
                print(json.dumps({'mode': mode, 'frame': frame, 'wg': wg, 'stack_ops': ops, 'stack': stack, 'code': os.path.getsize(code), 'hw_us': hw, 'pred_us': pr and round(pr, 1), 'ratio': round(pr / hw, 3) if pr and hw else None, 'hw_ns_per_op': round(hw * 1000 / ops, 2) if hw else None, 'err': None if hw else (h.stdout + h.stderr)[-200:]}), flush=True)
