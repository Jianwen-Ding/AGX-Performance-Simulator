#!/usr/bin/env python3
"""Offline check of the retained instruction-benchmark evidence (no GPU)."""
import gzip, json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]/'results/2026-10-11-instruction-bench'
r = json.loads(gzip.open(R/'bench.json.gz').read())
fails = 0
def check(ok, msg):
    global fails
    fails += not ok; print(('ok   ' if ok else 'FAIL ')+msg)
ops = r['ops']
check(r.get('complete') and len(ops) == 60, f"60 instruction variants measured, run complete ({r['dispatches']} dispatches)")
check(all('cycles_per_step' in v[m] for v in ops.values() for m in ('latency', 'simd_issue', 'gpu')), 'every variant accepted in all three modes')
check(all(v[m]['max_abs_drift'] < 0.02 for v in ops.values() for m in ('latency', 'simd_issue')), 'single-group modes: reference phases before/after agree within 2% (no clock change)')
agree = [abs(v['gpu']['previous_dispatch_lanes']-v['gpu']['lanes_per_cycle_per_core'])/v['gpu']['lanes_per_cycle_per_core'] for v in ops.values()]
check(max(agree) <= 0.05, f'GPU mode: two independent dispatches agree within 5% for every variant (worst {max(agree):.3f})')
lat = lambda o: ops[o]['latency']['cycles_per_step']; iss = lambda o: ops[o]['simd_issue']['cycles_per_step']; ipc = lambda o: ops[o]['gpu']['lanes_per_cycle_per_core']/32
DVFS = (444, 612, 808, 968, 1110, 1236, 1338, 1398)
mis = [(o, m, v[m]['mhz']) for o, v in ops.items() for m in ('latency', 'simd_issue', 'gpu')
       if min(abs(v[m]['mhz']*1.049-f)/f for f in DVFS) > 0.015]
KNOWN_OFF_LEVEL = {('imadd_cache', 'gpu')}   # documented in the README: core medians straddle a clock change
check({(o, m) for o, m, _ in mis} == KNOWN_OFF_LEVEL, f'inferred clocks x 1.049 lie within 1.5% of a DVFS level except the documented {sorted(KNOWN_OFF_LEVEL)}')
check(abs(lat('iadd')-2) < 0.05, 'register-operand iadd chain has the same latency as the reference (2 units)')
bound = lambda o: 4*32/iss(o)
gb = {o: ops[o]['gpu']['lanes_per_cycle_per_core']/bound(o) for o in ('mov_imm', 'iadd_imm', 'fadd32_imm', 'fmadd32_imm', 'fadd16', 'fmul16', 'quad_shuffle')}
check(all(0.95 < x < 1.05 for x in gb.values()), f'one-register / 16-bit forms sit on the 4-group bound (ratios {min(gb.values()):.2f}-{max(gb.values()):.2f})')
check(all(ops[o]['gpu']['lanes_per_cycle_per_core']/bound(o) < 0.6 for o in ('iadd', 'fadd32', 'fmul32', 'and', 'xor')),
      'two-register-source forms are a core limit: < 0.6 of the 4-group bound')
check(ops['fmadd32_cache']['gpu']['lanes_per_cycle_per_core'] > 1.3*ops['fmadd32']['gpu']['lanes_per_cycle_per_core']
      and all(iss(o) > 1.9 for o in ('fadd32_cache', 'fmul32_cache', 'iadd_cache')), '.cache: raises fmadd32 throughput, halves single-group issue of fadd32/fmul32/iadd')
check(ops['fadd16']['gpu']['lanes_per_cycle_per_core'] > 115 and ops['fmadd16']['gpu']['lanes_per_cycle_per_core'] < 65, 'fp16 add at the bound, fp16 fma (3 register sources) about half')
sh = [ops[o]['gpu']['lanes_per_cycle_per_core']/bound(o) for o in ('simd_shuffle', 'simd_shuffle_down', 'simd_shuffle_xor')]
check(all(1.08 < x < 1.16 for x in sh), f'documented error: SIMD shuffles exceed the 4-group bound ({min(sh):.2f}-{max(sh):.2f})')
check(all(1.8 < ipc(o) < 2.2 for o in ('iadd', 'isub', 'and', 'xor', 'fadd32', 'fmul32')) and 1.2 < ipc('fmadd32') < 1.5,
      'two 32-bit register sources cap at ~2 instr/cycle/core, three (fmadd32) at ~1.4')
check(all(abs(lat(o)-2) < 0.1 for o in ('iadd', 'fadd32', 'fmul32', 'fmadd32', 'fadd16', 'fmadd16')), 'simple int/fp32/fp16/fma latency = 2 cycles')
check(all(abs(lat(o)-3) < 0.2 for o in ('imadd', 'bfi', 'popcount', 'convert_f2i')), 'imad / bitfield / convert latency = 3 cycles')
check(all(abs(iss(o)-2) < 0.15 for o in ('imadd', 'bfi', 'convert_f2i', 'simd_shuffle')), 'imad / bitfield / convert / shuffle issue every 2 cycles per SIMD group')
check(all(abs(lat(o)-iss(o)) < 0.3 for o in ('simd_iadd', 'simd_fadd', 'simd_min.u', 'simd_prefix_iadd')), 'SIMD reductions/scans do not overlap within a SIMD group (issue = latency)')
import subprocess
t = subprocess.run([sys.executable, str(Path(__file__).with_name('bench_report.py'))], capture_output=True, text=True).stdout
check(t.rstrip() == (R/'table.md').read_text().rstrip(), 'table.md regenerates exactly from bench.json.gz')
print(f'{fails} failures'); sys.exit(1 if fails else 0)
