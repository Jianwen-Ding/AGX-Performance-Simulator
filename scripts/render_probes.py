#!/usr/bin/env python3
"""Direct-ioctl render probes built on a traced Mesa pass (tests/render_trace.c + scripts/render_replay.py).

Each probe takes a captured submit, rewrites a few known fields or replaces the fragment / vertex code with our own
assembled AGX code (rust/target/release/agx_asm), replays the pass through drm_render_replay (no Mesa), and reads
the framebuffer back with the end-of-tile PBE switched to an uncompressed linear layout.

    python3 -I scripts/render_probes.py TRACE/submit-N lanes        # lane -> pixel map of fragment SIMD groups
    python3 -I scripts/render_probes.py TRACE/submit-N <probe> ...  # see PROBES below

Results: JSON on stdout and files in --out (default /scratch/render/v2/probes/<probe>).
"""
import json, struct, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_decode as rd
import render_replay as rr

ASM = HERE.parent / 'rust/target/release/agx_asm'


def asm(lines):
    """Assemble `TEXT` or (`TEXT`, want) or ('@F=V', '!HEX') requests with agx_asm; returns bytes."""
    if any(isinstance(l, str) and l.startswith('raw:') for l in lines):   # raw encodings pass through unchanged
        return b''.join(bytes.fromhex(l[4:]) if isinstance(l, str) and l.startswith('raw:') else asm([l]) for l in lines)
    reqs = []
    for l in lines:
        if isinstance(l, tuple):
            reqs.append('\t'.join(l))
        else:
            reqs.append(l)
    out = subprocess.run([str(ASM)], input='\n'.join(reqs) + '\n', capture_output=True, text=True, check=True).stdout.splitlines()
    code = b''
    for r, o in zip(reqs, out):
        if o.startswith('ERROR'):
            raise ValueError(f'{r}: {o}')
        code += bytes.fromhex(o.split('\t')[0])
    return code


class Pass:
    """Addresses of the interesting objects of one traced render pass."""

    def __init__(self, trace_dir):
        self.dir = Path(trace_dir)
        self.t = rd.Trace(trace_dir)
        self.cmd = rd.decode_render(self.t.render_raw(0))
        self.stream = rd.walk_vdm(self.t, self.cmd['vdm_ctrl_stream_base'])
        self.g = rd.gx()
        self.fs = []      # (ppp block, fragment shader dict)
        self.vs = []      # vdm_state blocks
        self.draws = []
        self.viewports = []
        for b in self.stream:
            if b['block'] == 'ppp' and isinstance(b['record'], dict):
                r = b['record']
                if 'Fragment shader' in r:
                    self.fs.append((b, r['Fragment shader']))
                if 'Viewport' in r:
                    self.viewports.append(b)
            elif b['block'] == 'vdm_state' and 'usc' in b:
                self.vs.append(b)
            elif b['block'] == 'index_list':
                self.draws.append(b)
        eot = rd.decode_usc(self.t, self.t.usc_base + (self.cmd['eot.usc'] & ~63))
        texw = [w for w in eot if w['_struct'] == 'USC Texture'][0]
        self.pbe_va = texw['Buffer']
        self.pbe = self.g.unpack('PBE', self.t.read(self.pbe_va, 24))
        self.fb_va = self.pbe['Buffer']
        self.width, self.height = self.cmd['width_px'], self.cmd['height_px']

    @staticmethod
    def code_va(t, usc):
        return t.usc_base + [w for w in usc if w['_struct'] == 'USC Shader'][0]['Code']

    def fs_code_va(self, i=0):
        return self.code_va(self.t, self.fs[i][1]['usc'])

    def vs_code_va(self, i=0):
        return self.code_va(self.t, self.vs[i]['usc'])

    def linear_pbe_patch(self, bpp=4):
        """End-of-tile PBE -> uncompressed linear; the partial-render background program's texture (it reloads the target
        after a partial render, HW: scenes whose vertex stage overflows) gets the same layout, or it reads garbage."""
        base = self.t.read(self.pbe_va, 24)
        new = self.g.pack('PBE', {'Layout': 'Linear', 'Compressed': False, 'Extended': False,
                                  'Stride': self.stride(bpp) - 16}, base)
        out = {self.pbe_va: new}
        try:
            pbg = rd.decode_usc(self.t, self.t.usc_base + (self.cmd['partial_bg.usc'] & ~63))
            for w in pbg:
                if w['_struct'] == 'USC Texture':
                    tb = self.t.read(w['Buffer'], 24)
                    if self.g.unpack('Texture', tb)['Address'] == self.fb_va:
                        out[w['Buffer']] = self.g.pack('Texture', {'Layout': 'Linear', 'Compressed': False, 'Compression': 0,
                                                                   'Extended': False, 'Stride': self.stride(bpp) - 16}, tb)
        except KeyError:
            pass
        return out

    def stride(self, bpp=4):
        # linear PBE rows are whole 16-byte units ((field >> 4) + 1) * 16 (hardware, 33-px target)
        return (self.width * bpp + 15) & ~15

    def run(self, out, patches=None, fields=None, reps=1, extra_dumps=(), bpp=4, train=False):
        p = dict(self.linear_pbe_patch(bpp))
        p.update(patches or {})
        st = self.stride(bpp)
        n = st * self.height
        r = rr.replay(self.dir, out, p, fields, [(self.fb_va, n)] + list(extra_dumps), reps, train=train)
        r['image'] = self.unpad(r['dumps'].get(self.fb_va), bpp)
        return r

    def unpad(self, img, bpp=4):
        if img is None:
            return None
        st = self.stride(bpp)
        return b''.join(img[y * st:y * st + self.width * bpp] for y in range(self.height))


def lanes_probe(pa, out):
    """Fragment code: (lane, simdgroup index, sr0, sr1) as i8 into the tile. Tells which pixels a SIMD group covers."""
    code = asm([('get_sr r0, sr52 (thread_index_in_simdgroup)', 'get_sr r0, sr52 (thread_index_in_simdgroup)'),
                ('get_sr r1, sr53 (simdgroup_index_in_threadgroup)', 'get_sr r1, sr53 (simdgroup_index_in_threadgroup)'),
                ('get_sr r2, sr0 (threadgroup_position_in_grid.x)', 'get_sr r2, sr0 (threadgroup_position_in_grid.x)'),
                ('get_sr r3, sr1 (threadgroup_position_in_grid.y)', 'get_sr r3, sr1 (threadgroup_position_in_grid.y)'),
                'wait_pix 12, 0', ('@R=0,F=0', '!09110004f0fc0003'), 'stop'])
    r = pa.run(out, {pa.fs_code_va(): code})
    img = r['image']
    W, H = pa.width, pa.height
    px = [[tuple(img[4 * (y * W + x):4 * (y * W + x) + 4]) for x in range(W)] for y in range(H)]
    return r, px


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('trace'); ap.add_argument('probe'); ap.add_argument('--out')
    a = ap.parse_args()
    pa = Pass(a.trace)
    out = Path(a.out or f'/scratch/render/v2/probes/{a.probe}')
    if a.probe == 'lanes':
        r, px = lanes_probe(pa, out)
        print(json.dumps({'ok': r['ok'], 'reps': r['reps']}))
        for y in range(min(pa.height, 16)):
            print(' '.join(f'{px[y][x][0]:2d}' for x in range(min(pa.width, 32))))
        print('sr53/sr0/sr1 samples:', sorted({p[1:3] for row in px for p in row})[:20])
        (out / 'lanes.json').write_text(json.dumps(px))


if __name__ == '__main__':
    main()


def cf_probe(pa, out, regs):
    """Hardware coefficient registers: per pixel, ldcf cfN -> (A, B, C) f32 bits (3 replays per register)."""
    res = {}
    for n in regs:
        for k in range(3):
            code = asm([('@b16_8=%d' % n, '!a181004000000000'), 'wait_pix 12, 0',
                        ('@R=%d,F=2,mask=1' % (2 * k), '!09110004f0fc0003'), 'stop'])
            r = pa.run(f'{out}/cf{n}_{k}', {pa.fs_code_va(): code})
            img = r['image']
            res[(n, k)] = [struct.unpack_from('<I', img, 4 * p)[0] for p in range(pa.width * pa.height)]
    return res


def run_warm(pa, out, patches=None, fields=None, bursts=2, reps=15):
    """Top-clock timing of a pass: the clock keeper warms the GPU (~80 ms), then `reps` copies go back to back (train mode);
    vtx / frag = median of reps 3.. of each burst, the lower burst kept. (Zero timestamps on later reps of a burst, seen once,
    came from GPU recoveries caused by a killed keeper, not from a timestamp limit; the keeper now stops gracefully.)"""
    import clock_keeper
    best, last = None, None
    for b in range(bursts):
        with clock_keeper.keep(warm_s=0.08):
            pass
        r = pa.run(out, patches, fields, reps=reps, train=True)
        if not r['ok']:
            return r
        good = [x for x in r['reps'][3:] if x['vtx_ns'] > 0 and x['frag_ns'] > 0]
        if not good:
            continue
        v = sorted(x['vtx_ns'] for x in good)[len(good) // 2]
        f = sorted(x['frag_ns'] for x in good)[len(good) // 2]
        if best is None or v + f < best[0] + best[1]:
            best = (v, f)
        last = r
    last = last or r
    last['warm_vtx_ns'], last['warm_frag_ns'] = best if best else (None, None)
    return last
