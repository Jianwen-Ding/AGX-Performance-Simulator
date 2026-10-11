#!/usr/bin/env python3
"""Render the instruction benchmark table (markdown) from results/2026-10-11-instruction-bench/bench.json.gz."""
import gzip, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
r = json.loads(gzip.open(REPO/'results/2026-10-11-instruction-bench/bench.json.gz').read())
CLS = {'int': 'Integer', 'int64': '64-bit integer', 'bitfield': 'Bit/shift', 'select': 'Compare-select', 'move': 'Move',
       'float': 'FP32', 'half': 'FP16', 'convert': 'Convert', 'sfu': 'Transcendental / rounding', 'simd': 'SIMD-group / quad'}
head = ['Cycle values are in reference units (dependent 8-byte `iadd x, x, 1` step = 2); real cycles = value x 1.049',
        '(clocks inferred this way are the DVFS levels / 1.049). `mov_imm` has no source register: its latency column is its issue interval.', '']
lines = head+['| Instruction | Class | Latency (cycles) | Issue interval, 1 SIMD group (cycles) | Lanes / cycle / core | SIMD instr / cycle / core |',
         '| --- | --- | --- | --- | --- | --- |']
for op, v in r['ops'].items():
    t = v['template'].replace('r20', 'c0').replace('r21', 'c1').replace('r22', 'c2').replace('{d}', 'x').replace('{d1}', 'x1')
    g = v['gpu']
    lines.append(f"| `{t}` | {CLS[v['class']]} | {'n/a' if op == 'mov_imm' else format(v['latency']['cycles_per_step'], '.1f')} | {v['simd_issue']['cycles_per_step']:.2f} | "
                 f"{g['lanes_per_cycle_per_core']:.0f} | {g['lanes_per_cycle_per_core']/32:.2f} |")
text = '\n'.join(lines)+'\n'
if len(sys.argv) > 1: Path(sys.argv[1]).write_text(text)
print(text)
