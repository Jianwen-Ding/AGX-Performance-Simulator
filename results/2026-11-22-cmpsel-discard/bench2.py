# cmpsel operand-hint split: which of .cache (3rd source) / .discard (4th) costs the extra issue cycles
import sys
sys.path.insert(0, '/save/workspace/AGX-Performance-Simulator/scripts')
import coverage as CV; CV.install()
import coverage_catalog as C, run_bench as B, gpu_safety
base = 'icmpsel ult, r14, r6, r7, r8, r9'
forms = {
 'ic_d7':    'icmpsel ult, r14, r6, r7.discard, r8, r9',
 'ic_d89':   'icmpsel ult, r14, r6, r7, r8.discard, r9.discard',
 'ic_d9x':   'icmpsel ult, r14, r6, 0, r8, r9.discard',
 'ic16_d9':  'icmpsel ult, r14l, r6l, r7h, r8l, r9h.discard',
 'fc16_d9':  'fcmpsel lt, r14l, r6l, r7l, r8l, r9l.discard',
 'fc_c7':    'fcmpsel lt, r14, r6, r7.cache, r8, r9',
}
names = []
for k, t in forms.items():
    tmpl = C.bench_template(t); cls = C.bench_class(t, tmpl)
    B.OPS['cs_' + k] = (tmpl, cls); names.append('cs_' + k); print(k, tmpl, cls, flush=True)
gpu_safety.require_budget(1)
sys.argv = ['run_bench.py', '--ops', ','.join(names), '--modes', 'latency,simd_issue', '--output', '/scratch/agx-r7/cmpsel/run2']
B.main()
