import json, sys, collections, os
sys.path.insert(0, '/save/workspace/AGX-Performance-Simulator/scripts')
import render_check as rc, render_probes as rp, gpu_safety
os.makedirs('/scratch/render/v3/cmsweep', exist_ok=True)
sizes = [tuple(map(int, s.split('x'))) for s in sys.argv[1:]]
res = json.load(open('/scratch/render/v3/cmsweep/maps.json')) if os.path.exists('/scratch/render/v3/cmsweep/maps.json') else {}
code = rp.asm([('get_sr r0, sr20 (core_index)', 'get_sr r0, sr20 (core_index)'), 'wait_pix 12, 0', ('@R=0,F=0', '!09110004f0fc0003'), 'stop'])
for W, H in sizes:
    name = f'cm_{W}x{H}'
    if name in res: continue
    scene = f'/scratch/render/v3/cmsweep/{name}.txt'
    open(scene, 'w').write(f'size {W} {H}\nshader smooth\ndepth off\n' + ''.join(f'v {x} {y} 0.5 1 1 1 1 1\n' for x, y in [(-1,-1),(1,-1),(-1,1),(1,-1),(1,1),(-1,1)]) + 'draw 0 6\n')
    with gpu_safety.Tripwire('coremap ' + name):
        tr, _ = rc.capture(scene, name)
        pa = rp.Pass(tr)
        r = pa.run(f'/scratch/render/v3/cmsweep/run_{name}', {pa.fs_code_va(): code})
    img = r['image']
    tiles = collections.defaultdict(collections.Counter)
    for y in range(H):
        for x in range(W):
            tiles[(x // 32, y // 32)][img[4 * (y * W + x)]] += 1
    tx, ty = (W + 31) // 32, (H + 31) // 32
    grid = [[max(tiles[(x, y)], key=tiles[(x, y)].get) for x in range(tx)] for y in range(ty)]
    per = collections.Counter(c for row in grid for c in row)
    res[name] = {'W': W, 'H': H, 'grid': grid, 'max_per_core': max(per.values()), 'cores': len(per)}
    print(name, 'tiles', tx, ty, 'cores', len(per), 'max/core', max(per.values()), 'multi', sum(len(v) > 1 for v in tiles.values()), flush=True)
    json.dump(res, open('/scratch/render/v3/cmsweep/maps.json', 'w'))
