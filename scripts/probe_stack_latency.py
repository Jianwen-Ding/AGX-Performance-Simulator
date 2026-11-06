#!/usr/bin/env python3
"""probe_stack_latency.py: latency of a stack_load whose word was stored P distinct words earlier. One SIMD group (wg 32,
lane-uniform offsets) runs a loop: stack_load a[j]; wait 0; add; stack_store a[j], with j cycling over P words of a 16 KB
frame (stride 37). Two trip counts give ns per iteration (the slope), hardware (back to back) vs agx_dispatch.
2026-11-06 (M2 Max, wg 32): hw 102.7 / 102.7 / 160.8 / 160.6 / 648 ns per iteration at P = 16 / 64 / 256 / 1024 / 4096 words,
model 103.6 / 103.6 / 168.0 / 168.6 / 417 (L1 and L2 right, beyond L2 low)."""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
B = os.environ.get('LLVM_BIN', '/scratch/llvm-agx-main-build/bin')
HW = os.environ.get('AGX_RUN_HW', '/scratch/zluda-air-target/release/agx_run_hw')
DISP = os.environ.get('AGX_DISPATCH', '/scratch/agxsim-target/release/agx_dispatch')
W = int(os.environ.get('FRAME_WORDS', '4096'))
def ir(p, n, wg_stride):
    return f'''target triple = "agx-unknown-unknown"
define agx_kernel void @k() {{
entry:
  %a = alloca [{W} x i32], addrspace(5)
  br label %init
init:
  %w = phi i32 [0, %entry], [%w1, %init]
  %pw = getelementptr [{W} x i32], ptr addrspace(5) %a, i32 0, i32 %w
  store volatile i32 %w, ptr addrspace(5) %pw
  %w1 = add i32 %w, 1
  %cw = icmp ult i32 %w1, {W}
  br i1 %cw, label %init, label %loop
loop:
  %i = phi i32 [0, %init], [%i1, %loop]
  %x = phi i32 [0, %init], [%x2, %loop]
  %m = mul i32 %i, {wg_stride}
  %j = and i32 %m, {p - 1}
  %p = getelementptr [{W} x i32], ptr addrspace(5) %a, i32 0, i32 %j
  %v = load volatile i32, ptr addrspace(5) %p
  %x2 = add i32 %v, %x
  store volatile i32 %x2, ptr addrspace(5) %p
  %i1 = add i32 %i, 1
  %c = icmp ult i32 %i1, {n}
  br i1 %c, label %loop, label %exit
exit:
  ret void
}}
'''
def build(d, p, n):
    ll, o, code = (os.path.join(d, f'{x}') for x in ('k.ll', 'k.o', f'k{p}_{n}.bin'))
    open(ll, 'w').write(ir(p, n, 37))
    subprocess.run([B + '/llc', '-mtriple=agx', '-mcpu=agx-g14c', '-O2', '-filetype=obj', ll, '-o', o], check=True)
    subprocess.run([B + '/llvm-objcopy', '-O', 'binary', '--only-section=.text', o, code], check=True)
    asm = subprocess.run([B + '/llc', '-mtriple=agx', '-mcpu=agx-g14c', '-O2', ll, '-o', '-'], capture_output=True, text=True).stdout
    assert 'call' not in asm.split('.Lfunc_end0')[0].replace('// @k', '')
    return code, int(re.search(r'stack size: (\d+)', asm).group(1))
mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
if mode in ('both', 'hw'):
    import gpu_safety; gpu_safety.require_budget(1)
Ps = [int(x) for x in os.environ.get('PS', '16 64 256 1024 4096').split()]
N1, N2 = 4000, 20000
with tempfile.TemporaryDirectory() as d:
    for p in Ps:
        rec = {'P_words': p, 'rows_kb': p * 128 / 1024}
        for key, run in (('hw', mode in ('both', 'hw')), ('sim', mode in ('both', 'sim'))):
            if not run: continue
            t = []
            for n in (N1, N2):
                code, stack = build(d, p, n)
                common = [code, '--grid', '32,1,1', '--wg', '32,1,1', '--regs', '16']
                if key == 'hw':
                    h = subprocess.run([HW] + common + ['--stack', str(stack), '--back-to-back', '100', '--timeout-ms', '10000'], capture_output=True, text=True)
                    m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
                    t.append(int(m.group(1)) / 1000 if m else None)
                    if not m: rec['hw_err'] = (h.stdout + h.stderr)[-200:]
                else:
                    r = subprocess.run([DISP] + common + ['--launch-profile', 'zluda', '--stack', str(stack), '--alloc', '0x100000=16'], capture_output=True, text=True)
                    t.append(json.loads(r.stdout)['time_us'])
            if None not in t:
                rec[key + '_us'] = t; rec[key + '_ns_per_iter'] = round((t[1] - t[0]) * 1000 / (N2 - N1), 2)
        print(json.dumps(rec), flush=True)
