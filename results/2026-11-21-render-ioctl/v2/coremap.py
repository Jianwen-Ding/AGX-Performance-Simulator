"""Tile -> core map of fragment work: replace the fragment shader with one that stores sr20 (core id) / sr17 (slot) / sr53 into
the pixel (direct-ioctl replay of a captured pass, linear PBE readback)."""
import json, sys, collections
sys.path.insert(0, '/save/workspace/AGX-Performance-Simulator/scripts')
import render_probes as rp, gpu_safety
out = {}
for n in sys.argv[1:]:
    pa = rp.Pass(f'/scratch/render/v2/cap/{n}/submit-1')
    code = rp.asm([('get_sr r0, sr20 (core_index)', 'get_sr r0, sr20 (core_index)'),
                   ('get_sr r1, sr17', 'get_sr r1, sr17'),
                   ('get_sr r2, sr53 (simdgroup_index_in_threadgroup)', 'get_sr r2, sr53 (simdgroup_index_in_threadgroup)'),
                   ('get_sr r3, sr52 (thread_index_in_simdgroup)', 'get_sr r3, sr52 (thread_index_in_simdgroup)'),
                   'wait_pix 12, 0', ('@R=0,F=0', '!09110004f0fc0003'), 'stop'])
    with gpu_safety.Tripwire('coremap ' + n):
        r = pa.run(f'/scratch/render/v3/coremap/{n}', {pa.fs_code_va(): code})
    img = r['image']; W, H = pa.width, pa.height
    tiles = collections.defaultdict(collections.Counter)
    for y in range(H):
        for x in range(W):
            c, s = img[4 * (y * W + x)], img[4 * (y * W + x) + 1]
            tiles[(x // 32, y // 32)][c] += 1
    tm = {f'{k[0]},{k[1]}': dict(v) for k, v in sorted(tiles.items())}
    per_core = collections.Counter(max(v, key=v.get) for v in tiles.values())
    out[n] = {'tiles': tm, 'cores_used': len(per_core), 'max_tiles_per_core': max(per_core.values()), 'hist': sorted(per_core.values(), reverse=True)}
    print(n, 'ok', r['ok'], 'cores', len(per_core), 'max tiles/core', max(per_core.values()), 'multi-core tiles', sum(len(v) > 1 for v in tiles.values()), flush=True)
    tx = max(k[0] for k in tiles) + 1; ty = max(k[1] for k in tiles) + 1
    for y in range(ty):
        print('   ', ' '.join('%3s' % ('/'.join('%x' % c for c in sorted(tiles[(x, y)]))) for x in range(tx)))
json.dump(out, open('/scratch/render/v3/coremap.json', 'w'))
