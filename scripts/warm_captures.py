#!/usr/bin/env python3
"""warm_captures.py CAPTURE_DIR KERNEL_REGS.json [SECONDS]: each captured launch on the GPU with its clock held up: a small kernel keeps
the GPU busy in another process while the launch runs from its captured memory, restored each run. Without it, the same launch measures
up to 2.8x slower and varies by 2x: a launch of its own does not keep the clock up. Prints JSON lines of warm times; compare with
predict_captures.py. Kernels using hardware textures do not replay: capture with ZLUDA_AGX_SOFTWARE_TEXTURES=1."""
import json, os, re, shlex, subprocess, sys, tempfile
HW = os.environ.get('AGX_RUN_HW', 'agx_run_hw'); ASM = os.environ.get('AGX_ASM', 'agx_asm')
cap = sys.argv[1]; regs = json.load(open(sys.argv[2])); repeat = sys.argv[3] if len(sys.argv) > 3 else '8'
def enc(t): return bytes.fromhex(subprocess.run([ASM], input=t + '\n', capture_output=True, text=True, check=True).stdout.split()[0])
with tempfile.TemporaryDirectory() as tmp:
    k = os.path.join(tmp, 'k.bin'); open(k, 'wb').write(enc('iadd r5, r6, 1') * 40000 + enc('stop'))
    keeper = subprocess.Popen(['sh', '-c', f'while :; do {HW} {k} --grid 32,1,1 --wg 32,1,1 --regs 16 --back-to-back 1000 >/dev/null 2>&1; done'], start_new_session=True)
    try:
        for d in sorted(os.listdir(cap)):
            rs = os.path.join(cap, d, 'replay.sh')
            if not os.path.exists(rs): continue
            text = open(rs).read(); kernel = re.search(r'# kernel (\S+)', text).group(1)
            st = re.search(r'uses (\d+) bytes of stack', text)
            line = [l for l in text.splitlines() if 'exec ' in l][0]
            a = [x for x in shlex.split(line.split('exec ', 1)[1])[1:] if x != '$@']
            out, i = [], 0
            while i < len(a):
                if a[i] == '--dump': i += 2; continue
                out.append(a[i]); i += 1
            cmd = [HW] + out + ['--regs', str(regs.get(kernel, 256)), '--repeat', repeat] + (['--stack', st.group(1)] if st else [])
            p = subprocess.run(cmd, cwd=os.path.join(cap, d), capture_output=True, text=True, errors='replace', timeout=300)
            g = lambda n: (lambda m: int(m.group(1)) / 1000 if m else None)(re.search(f'"{n}":(\\d+)', p.stdout))
            print(json.dumps({'kernel': kernel.replace('kernel_gpu_', ''), 'threads': int(out[out.index('--grid') + 1].split(',')[0]), 'first_us': g('first_fw_ns'), 'min_us': g('min_fw_ns'), 'median_us': g('median_fw_ns'), 'err': None if g('median_fw_ns') else (p.stdout + p.stderr)[-200:]}), flush=True)
    finally:
        os.killpg(keeper.pid, 9); keeper.wait()
