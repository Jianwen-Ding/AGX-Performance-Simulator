import json, sys
sys.path.insert(0, '/save/workspace/AGX-Performance-Simulator/scripts')
import render_probes as rp, gpu_safety, clock_keeper
n = sys.argv[1]; nb = int(sys.argv[2])
pa = rp.Pass(f'/scratch/render/v2/cap/{n}/submit-1')
ref = None
with gpu_safety.Tripwire('pp2 ' + n):
    for b in range(nb):
        with clock_keeper.keep(warm_s=0.08):
            pass
        r = pa.run(f'/scratch/render/v3/pp2/{n}', fields={'partial_bg.usc': pa.cmd['bg.usc'], 'partial_bg.rsrc_spec': pa.cmd['bg.rsrc_spec']}, reps=15, train=True)
        if ref is None:
            ref = pa.run(f'/scratch/render/v3/pp2/{n}_ref', reps=1)['image']
        img = r['image']
        diff = sum(1 for i in range(0, len(img), 4) if img[i:i+4] != ref[i:i+4])
        f = [x['frag_ns'] for x in r['reps']]; v = [x['vtx_ns'] for x in r['reps']]
        print(b, 'diff', diff, 'frag', f[-5:], 'vtx', v[-5:], flush=True)
