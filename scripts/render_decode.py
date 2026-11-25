#!/usr/bin/env python3
"""Decode a traced render submission (tests/render_trace.c output) with Mesa's open field description
(agx-notes/mesa/src/asahi/genxml/cmdbuf.xml): the drm_asahi_cmd_render record, the VDM control stream
(PPP state records, VDM state, index lists, links, barriers), USC pipeline words of the vertex, fragment,
background and end-of-tile programs, and the vertex / index data the draws read.

    python3 -I scripts/render_decode.py TRACE/submit-N [--json OUT] [--quiet]

Library use (simulator front end, scripts/render_*.py):
    import render_decode as rd
    t = rd.Trace(dir); cmd = rd.decode_render(t.render_raw(0)); stream = rd.walk_vdm(t, cmd['vdm_ctrl_stream_base'])

Pure Python (no numpy). The XML is parsed once; struct packing is word:bit based exactly as gen_pack.py.
"""
import json, os, struct, sys, xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
XML_CANDIDATES = [os.environ.get('AGX_CMDBUF_XML', ''), '/save/workspace/agx-notes/mesa/src/asahi/genxml/cmdbuf.xml',
                  '/imports/0/agx-notes/mesa/src/asahi/genxml/cmdbuf.xml']


# ---------------------------------------------------------------- genxml
class GenXML:
    def __init__(self, path=None):
        path = path or next(p for p in XML_CANDIDATES if p and Path(p).exists())
        root = ET.parse(path).getroot()
        self.enums = {}
        for e in root.iter('enum'):
            self.enums[e.get('name')] = {int(v.get('value'), 0): v.get('name') for v in e.iter('value')}
        self.structs = {}
        for s in root.iter('struct'):
            fields = []
            for f in s.iter('field'):
                st = f.get('start')
                if ':' in st:
                    w, b = st.split(':'); start = int(w) * 32 + int(b)
                else:
                    start = int(st)
                fields.append(dict(name=f.get('name'), start=start, size=int(f.get('size')), type=f.get('type'),
                                   modifier=f.get('modifier'), default=f.get('default')))
            self.structs[s.get('name')] = dict(size=int(s.get('size', '4')), fields=fields)

    def size(self, name):
        return self.structs[name]['size']

    def unpack(self, name, data, off=0):
        s = self.structs[name]
        raw = int.from_bytes(bytes(data[off:off + s['size']]).ljust(s['size'], b'\0'), 'little')
        out = {}
        for f in s['fields']:
            v = (raw >> f['start']) & ((1 << f['size']) - 1)
            t, m = f['type'], f['modifier']
            if m:
                if m.startswith('groups('):
                    g = int(m[7:-1]); v = (v if v else (1 << f['size'])) * g
                elif m.startswith('minus('):
                    v = v + int(m[6:-1])
                elif m.startswith('shr('):
                    v = v << int(m[4:-1])
                elif m == 'log2':
                    v = 1 << v
            if t == 'bool':
                v = bool(v)
            elif t == 'float':
                v = struct.unpack('<f', struct.pack('<I', v))[0]
            elif t == 'half':
                v = struct.unpack('<e', struct.pack('<H', v))[0]
            elif t in self.enums:
                v = self.enums[t].get(v, v)
            out[f['name']] = v
        return out

    def pack(self, name, values, base=None):
        """Inverse of unpack. `values` maps field name -> decoded value; fields not given keep `base`'s bits
        (bytes of the original record), so overlapping alternative fields (PBE stride vs levels) are not clobbered."""
        s = self.structs[name]
        raw = int.from_bytes(bytes(base).ljust(s['size'], b'\0')[:s['size']], 'little') if base is not None else 0
        for f in s['fields']:
            if f['name'] not in values:
                continue
            v, t, m = values[f['name']], f['type'], f['modifier']
            if t == 'bool':
                v = int(bool(v))
            elif t == 'float':
                v = struct.unpack('<I', struct.pack('<f', v))[0]
            elif t == 'half':
                v = struct.unpack('<H', struct.pack('<e', v))[0]
            elif t in self.enums and isinstance(v, str):
                v = {n: k for k, n in self.enums[t].items()}[v]
            if m:
                if m.startswith('groups('):
                    g = int(m[7:-1]); v = (v // g) & ((1 << f['size']) - 1)
                elif m.startswith('minus('):
                    v = v - int(m[6:-1])
                elif m.startswith('shr('):
                    v = v >> int(m[4:-1])
                elif m == 'log2':
                    v = v.bit_length() - 1
            mask = ((1 << f['size']) - 1) << f['start']
            raw = (raw & ~mask) | ((int(v) << f['start']) & mask)
        return raw.to_bytes(s['size'], 'little')


GX = None


def gx():
    global GX
    if GX is None:
        GX = GenXML()
    return GX


# ---------------------------------------------------------------- trace memory
class Trace:
    """GPU memory of one traced submit: bo-<va>.bin files (optionally the .after versions)."""

    def __init__(self, d, after=False):
        self.dir = Path(d)
        self.man = json.loads((self.dir / 'manifest.json').read_text())
        self.usc_base = int(self.man['usc_exec_base'], 16)
        self.regions = []
        for b in self.man['binds']:
            va = int(b['va'], 16)
            p = self.dir / f'bo-{va:x}{".after" if after else ""}.bin'
            if p.exists():
                self.regions.append((va, p.read_bytes()))
        self.regions.sort()
        self.cmdbuf = (self.dir / 'cmdbuf.bin').read_bytes()

    def find(self, va):
        for base, data in self.regions:
            if base <= va < base + len(data):
                return base, data
        return None, None

    def read(self, va, n):
        base, data = self.find(va)
        if base is None:
            raise KeyError(f'unmapped VA 0x{va:x}')
        o = va - base
        if o + n > len(data):
            raise KeyError(f'read past region at 0x{va:x}+{n}')
        return data[o:o + n]

    def has(self, va):
        return self.find(va)[0] is not None

    def u32(self, va):
        return struct.unpack('<I', self.read(va, 4))[0]

    def commands(self):
        out, off = [], 0
        cb = self.cmdbuf
        while off + 8 <= len(cb):
            ctype, size, vb, cbar = struct.unpack_from('<HHHH', cb, off)
            out.append(dict(type=ctype, size=size, vdm_barrier=vb, cdm_barrier=cbar, payload=cb[off + 8:off + 8 + size]))
            off += 8 + size
        return out

    def render_raw(self, idx=0):
        return [c for c in self.commands() if c['type'] == 0][idx]['payload']


# ---------------------------------------------------------------- drm_asahi_cmd_render
RENDER_LAYOUT = [
    ('flags', 'I'), ('isp_zls_pixels', 'I'), ('vdm_ctrl_stream_base', 'Q'),
    ('vertex_helper.binary', 'I'), ('vertex_helper.cfg', 'I'), ('vertex_helper.data', 'Q'),
    ('fragment_helper.binary', 'I'), ('fragment_helper.cfg', 'I'), ('fragment_helper.data', 'Q'),
    ('isp_scissor_base', 'Q'), ('isp_dbias_base', 'Q'), ('isp_oclqry_base', 'Q'),
    ('depth.base', 'Q'), ('depth.comp_base', 'Q'), ('depth.stride', 'I'), ('depth.comp_stride', 'I'),
    ('stencil.base', 'Q'), ('stencil.comp_base', 'Q'), ('stencil.stride', 'I'), ('stencil.comp_stride', 'I'),
    ('zls_ctrl', 'Q'), ('ppp_multisamplectl', 'Q'), ('sampler_heap', 'Q'), ('ppp_ctrl', 'I'),
    ('width_px', 'H'), ('height_px', 'H'), ('layers', 'H'), ('sampler_count', 'H'),
    ('utile_width_px', 'B'), ('utile_height_px', 'B'), ('samples', 'B'), ('sample_size_B', 'B'),
    ('isp_merge_upper_x', 'I'), ('isp_merge_upper_y', 'I'),
    ('bg.usc', 'I'), ('bg.rsrc_spec', 'I'), ('eot.usc', 'I'), ('eot.rsrc_spec', 'I'),
    ('partial_bg.usc', 'I'), ('partial_bg.rsrc_spec', 'I'), ('partial_eot.usc', 'I'), ('partial_eot.rsrc_spec', 'I'),
    ('isp_bgobjdepth', 'I'), ('isp_bgobjvals', 'I'),
    ('ts_vtx.start.handle', 'I'), ('ts_vtx.start.offset', 'I'), ('ts_vtx.end.handle', 'I'), ('ts_vtx.end.offset', 'I'),
    ('ts_frag.start.handle', 'I'), ('ts_frag.start.offset', 'I'), ('ts_frag.end.handle', 'I'), ('ts_frag.end.offset', 'I'),
]
RENDER_FMT = '<' + ''.join(t for _, t in RENDER_LAYOUT)


def decode_render(raw):
    n = struct.calcsize(RENDER_FMT)
    assert len(raw) >= n, (len(raw), n)
    vals = struct.unpack_from(RENDER_FMT, raw)
    d = {k: v for (k, _), v in zip(RENDER_LAYOUT, vals)}
    d['_size'] = n
    g = gx()
    d['_zls_ctrl'] = g.unpack('ZLS Control', struct.pack('<Q', d['zls_ctrl']))
    d['_isp_zls_pixels'] = g.unpack('CR ISP ZLS Pixels', struct.pack('<I', d['isp_zls_pixels']))
    d['_ppp_ctrl'] = g.unpack('CR PPP Control', struct.pack('<I', d['ppp_ctrl']))
    d['_merge'] = [struct.unpack('<f', struct.pack('<I', d[k]))[0] for k in ('isp_merge_upper_x', 'isp_merge_upper_y')]
    return d


def pack_render(d):
    return struct.pack(RENDER_FMT, *[d[k] for k, _ in RENDER_LAYOUT])


# ---------------------------------------------------------------- USC words
USC_TAGS = {0x38: ('USC Preshader', 8), 0x58: ('USC Fragment Properties', 4), 0x88: ('USC No Preshader', 2),
            0x0d: ('USC Shader', 6), 0x1d: ('USC Uniform', 8), 0x3d: ('USC Uniform High', 8), 0x4d: ('USC Shared', 4),
            0x8d: ('USC Registers', 4), 0x9d: ('USC Sampler', 8), 0xdd: ('USC Texture', 8)}


def decode_usc(t, va, limit=64):
    """USC pipeline words until (No) Preshader, which ends the list (Mesa's agxdecode_usc)."""
    words, g = [], gx()
    for _ in range(limit):
        tag = t.read(va, 1)[0]
        if tag not in USC_TAGS:
            words.append({'_struct': 'unknown', '_tag': tag, '_va': va}); break
        name, n = USC_TAGS[tag]
        w = g.unpack(name, t.read(va, n)); w['_struct'] = name; w['_va'] = va
        w['_hex'] = t.read(va, n).hex()
        words.append(w); va += n
        if tag in (0x38, 0x88):
            break
    return words


# ---------------------------------------------------------------- PPP record
PPP_ORDER_A = [('Fragment control', 'Fragment control'), ('Fragment control 2', 'Fragment control'),
               ('Fragment front face', 'Fragment face'), ('Fragment front face 2', 'Fragment face 2'),
               ('Fragment front stencil', 'Fragment stencil'), ('Fragment back face', 'Fragment face'),
               ('Fragment back face 2', 'Fragment face 2'), ('Fragment back stencil', 'Fragment stencil'),
               ('Depth bias/scissor', 'Depth bias/Scissor')]
PPP_ORDER_B = [('W clamp', 'W Clamp'), ('Output select', 'Output Select'), ('Varying counts 32', 'Varying Counts'),
               ('Varying counts 16', 'Varying Counts'), ('Cull', 'Cull'), ('Cull 2', 'Cull 2')]
PPP_ORDER_C = [('Occlusion query', 'Fragment occlusion query'), ('Occlusion query 2', 'Fragment occlusion query 2'),
               ('Output unknown', 'Output Unknown'), ('Output size', 'Output Size'), ('Varying word 2', 'Varying 2')]


def decode_ppp(t, va, words):
    g = gx(); data = t.read(va, words * 4); o = 0
    hdr = g.unpack('PPP Header', data, 0); o = 4
    rec = {'_header': hdr, '_va': va, '_words': words, '_hex': data.hex()}

    def take(key, sname):
        nonlocal o
        if hdr[key]:
            rec[key] = g.unpack(sname, data, o); o += g.size(sname)
    for k, s in PPP_ORDER_A:
        take(k, s)
    if hdr['Region clip']:
        rec['Region clip'] = []
        for _ in range(hdr['Viewport count']):
            rec['Region clip'].append(g.unpack('Region clip', data, o)); o += 8
    if hdr['Viewport']:
        o += 4  # viewport control
        rec['Viewport'] = []
        for _ in range(hdr['Viewport count']):
            rec['Viewport'].append(g.unpack('Viewport', data, o)); o += 24
    for k, s in PPP_ORDER_B:
        take(k, s)
    if hdr['Fragment shader']:
        fs = {f'w{i}': g.unpack(f'Fragment Shader Word {i}', data, o + 4 * i) for i in range(4)}
        o += 16
        pipe = t.usc_base + fs['w1']['Pipeline']
        try:
            fs['usc'] = decode_usc(t, pipe)
        except KeyError as e:
            fs['usc'] = str(e)
        cfb = fs['w2']['CF bindings']
        if cfb and fs['w0']['CF binding count']:
            try:
                raw = t.read(t.usc_base + cfb, 4 + 4 * fs['w0']['CF binding count'])
                fs['cf_header'] = g.unpack('CF binding header', raw, 0)
                fs['cf'] = [g.unpack('CF binding', raw, 4 + 4 * i) for i in range(fs['w0']['CF binding count'])]
            except KeyError as e:
                fs['cf'] = str(e)
        rec['Fragment shader'] = fs
    for k, s in PPP_ORDER_C:
        take(k, s)
    rec['_consumed'] = o
    rec['_ok'] = o == words * 4
    return rec


# ---------------------------------------------------------------- VDM stream
def walk_vdm(t, va, max_blocks=4096):
    g = gx(); out = []; stack = []
    for _ in range(max_blocks):
        w0 = t.u32(va); bt = w0 >> 29
        if bt == 0:  # PPP state update
            h = g.unpack('PPP State', t.read(va, 8))
            ptr = (h['Pointer (hi)'] << 32) | h['Pointer (lo)']
            blk = {'block': 'ppp', '_va': va, 'pointer': ptr, 'words': h['Size (words)'], '_hex': t.read(va, 8).hex()}
            try:
                blk['record'] = decode_ppp(t, ptr, h['Size (words)'])
            except KeyError as e:
                blk['record'] = str(e)
            out.append(blk); va += 8
        elif bt == 1:
            h = g.unpack('VDM Barrier', t.read(va, 4)); out.append({'block': 'barrier', '_va': va, **h, '_hex': t.read(va, 4).hex()})
            if h['Returns']:
                if not stack:
                    break
                va = stack.pop(); continue
            va += 4
        elif bt == 2:
            h = g.unpack('VDM State', t.read(va, 4)); n = 4
            blk = {'block': 'vdm_state', '_va': va, 'present': h}
            for key, sname in [('Restart index present', 'VDM State Restart Index'),
                               ('Vertex shader word 0 present', 'VDM State Vertex Shader Word 0'),
                               ('Vertex shader word 1 present', 'VDM State Vertex Shader Word 1'),
                               ('Vertex outputs present', 'VDM State Vertex Outputs'),
                               ('Tessellation present', 'VDM State Tessellation'),
                               ('Vertex unknown present', 'VDM State Vertex Unknown'),
                               ('Tessellation scale present', 'VDM State Tessellation Scale')]:
                if h[key]:
                    blk[sname] = g.unpack(sname, t.read(va + n, 4)); n += 4
            if h['Vertex shader word 1 present']:
                pipe = t.usc_base + blk['VDM State Vertex Shader Word 1']['Pipeline']
                try:
                    blk['usc'] = decode_usc(t, pipe)
                except KeyError as e:
                    blk['usc'] = str(e)
            if not h['Tessellation scale present']:
                n = (n + 7) & ~7
            blk['_hex'] = t.read(va, n).hex()
            out.append(blk); va += n
        elif bt == 3:
            h = g.unpack('Index List', t.read(va, 4)); n = 4
            blk = {'block': 'index_list', '_va': va, **h}
            for key, sname, sz in [('Index buffer present', 'Index List: Buffer lo', 4), ('Index count present', 'Index List: Count', 4),
                                   ('Instance count present', 'Index List: Instances', 4), ('Start present', 'Index List: Start', 4),
                                   ('Indirect buffer present', 'Index List: Indirect buffer', 8),
                                   ('Index buffer size present', 'Index List: Buffer size', 4)]:
                if h[key]:
                    blk[sname] = g.unpack(sname, t.read(va + n, sz)); n += sz
            blk['_hex'] = t.read(va, n).hex()
            out.append(blk); va += n
        elif bt == 4:
            h = g.unpack('VDM Stream Link', t.read(va, 8))
            tgt = (h['Target hi'] << 32) | h['Target lo']
            out.append({'block': 'link', '_va': va, 'target': tgt, 'with_return': h['With return']})
            if h['With return']:
                stack.append(va + 8)
            va = tgt
        elif bt == 6:
            out.append({'block': 'terminate', '_va': va}); break
        else:
            out.append({'block': f'unknown{bt}', '_va': va, '_hex': t.read(va, 8).hex()}); break
    return out


def summarize(stream):
    lines = []
    for b in stream:
        k = b['block']
        if k == 'ppp':
            r = b['record']
            if isinstance(r, str):
                lines.append(f"PPP @{b['pointer']:#x} {b['words']}w: {r}"); continue
            keys = [k2 for k2 in r if not k2.startswith('_')]
            lines.append(f"PPP @{b['pointer']:#x} {b['words']}w ok={r['_ok']} {keys}")
        elif k == 'vdm_state':
            lines.append('VDM state ' + str({x: y for x, y in b.items() if x.startswith('VDM State')}))
        elif k == 'index_list':
            lines.append(f"DRAW {b['Primitive']} idx={b['Index buffer present']} count={b.get('Index List: Count')} inst={b.get('Index List: Instances')} start={b.get('Index List: Start')}")
        else:
            lines.append(f"{k} " + str({x: y for x, y in b.items() if not x.startswith('_') and x != 'block'}))
    return lines


def default(o):
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    return str(o)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('trace'); ap.add_argument('--json'); ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    t = Trace(a.trace)
    res = {'commands': []}
    for c in t.commands():
        e = {'type': c['type'], 'size': c['size'], 'vdm_barrier': c['vdm_barrier'], 'cdm_barrier': c['cdm_barrier']}
        if c['type'] == 0:
            cmd = decode_render(c['payload'])
            e['render'] = cmd
            e['stream'] = walk_vdm(t, cmd['vdm_ctrl_stream_base'])
            for key in ('bg', 'eot', 'partial_bg', 'partial_eot'):
                u = cmd[f'{key}.usc']
                try:
                    e[f'{key}_usc'] = decode_usc(t, t.usc_base + (u & ~63))
                except KeyError as ex:
                    e[f'{key}_usc'] = str(ex)
        elif c['type'] in (2, 3, 4):
            e['attachments'] = [dict(zip(('pointer', 'size', 'pad', 'flags'), struct.unpack_from('<QQII', c['payload'], o)))
                                for o in range(0, len(c['payload']), 24)]
        res['commands'].append(e)
    if not a.quiet:
        for e in res['commands']:
            print(f"cmd type={e['type']} size={e['size']} vdm_barrier={e['vdm_barrier']} cdm_barrier={e['cdm_barrier']}")
            if 'render' in e:
                r = e['render']
                for k, v in r.items():
                    if k.startswith('_'):
                        print(f'  {k}: {v}')
                    else:
                        print(f'  {k} = {v:#x}' if isinstance(v, int) else f'  {k} = {v}')
                for line in summarize(e['stream']):
                    print('   ', line)
                for key in ('bg', 'eot', 'partial_bg', 'partial_eot'):
                    print(f'  {key}_usc:', e[f'{key}_usc'] if isinstance(e[f'{key}_usc'], str) else
                          [{k: v for k, v in w.items() if k not in ('_hex',)} for w in e[f'{key}_usc']])
            if 'attachments' in e:
                print('  attachments', [(hex(x['pointer']), x['size']) for x in e['attachments']])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=default))


if __name__ == '__main__':
    main()
