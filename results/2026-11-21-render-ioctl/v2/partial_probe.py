"""Partial-render detector: point partial_bg at the normal bg (clear) program. A partial render then clears the tile instead of
reloading it, so pixels drawn before the flush disappear. Compare with the unmodified replay (last rep of a train burst)."""
import json, sys
sys.path.insert(0, '/save/workspace/AGX-Performance-Simulator/scripts')
import render_probes as rp, gpu_safety, clock_keeper
names = sys.argv[1:]
for n in names:
    tr = f'/scratch/render/v2/cap/{n}/submit-1'
    pa = rp.Pass(tr)
    res = {'name': n, 'partial_bg': hex(pa.cmd['partial_bg.usc']), 'bg': hex(pa.cmd['bg.usc'])}
    with gpu_safety.Tripwire('partial probe ' + n):
        for reps in (1, 15):
            with clock_keeper.keep(warm_s=0.08):
                pass
            a = pa.run(f'/scratch/render/v3/pp/{n}/base{reps}', reps=reps, train=True)
            b = pa.run(f'/scratch/render/v3/pp/{n}/pbg{reps}', fields={'partial_bg.usc': pa.cmd['bg.usc'], 'partial_bg.rsrc_spec': pa.cmd['bg.rsrc_spec']}, reps=reps, train=True)
            if not (a['ok'] and b['ok']):
                res[f'r{reps}'] = 'fail'; continue
            diff = sum(1 for i in range(0, len(a['image']), 4) if a['image'][i:i+4] != b['image'][i:i+4])
            res[f'r{reps}'] = {'diff_px': diff, 'frag_base': [x['frag_ns'] for x in a['reps']], 'vtx_base': [x['vtx_ns'] for x in a['reps']]}
    print(json.dumps(res), flush=True)
