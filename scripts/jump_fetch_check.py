#!/usr/bin/env python3
"""jump_fetch_check.py [HW.jsonl]: offline check of the jump-fetch rule (cost::JumpFetch). Re-simulates every code
shape of probe_jump_distance.py stored in results/2026-11-25-jump-fetch/hardware.jsonl (agx_dispatch, runtime profile,
the same N as the hardware slope) and compares ns per block. Fails if the mean |error| exceeds 8%."""
import json, os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
hwf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, 'results/2026-11-25-jump-fetch/hardware.jsonl')
rel = os.path.join(root, 'rust/target/release')
env = dict(os.environ, AGX_RUN=f'{rel}/agx_run', AGX_DISPATCH=f'{rel}/agx_dispatch', AGX_ASM=f'{rel}/agx_asm')
hw = {}
for l in open(hwf):
    r = json.loads(l)
    if 'hw_ns_per_block' not in r or r.get('fill') or r.get('script', 'probe_jump_distance.py') != 'probe_jump_distance.py': continue
    ns = tuple(sorted(int(k.split('_')[1]) for k in r if k.startswith('instructions_')))
    hw.setdefault((r['kind'], ns), []).append(r['hw_ns_per_block'])
errs = []
for (kind, (n1, n2)), v in sorted(hw.items()):
    out = subprocess.run([sys.executable, os.path.join(here, 'probe_jump_distance.py'), 'sim'], env=dict(env, KINDS=kind, N1=str(n1), N2=str(n2)),
                         capture_output=True, text=True, check=True).stdout
    sim = json.loads(out.strip().splitlines()[-1])['sim_ns_per_block']; h = sum(v) / len(v); e = (sim - h) / h
    errs.append(abs(e)); print(f'{kind:12} N {n1}-{n2}: hw {h:6.1f} sim {sim:6.1f} ns/block {e*100:+6.1f}%', flush=True)
errs.sort(); mean = sum(errs) / len(errs)
print(f'{len(errs)} shapes: mean |err| {mean*100:.1f}%, median {errs[len(errs)//2]*100:.1f}%, max {errs[-1]*100:.1f}%')
sys.exit(mean > 0.08)
