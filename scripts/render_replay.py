#!/usr/bin/env python3
"""Replay a traced render submission through direct ioctls (tests/drm_render_replay.c), optionally mutated.

    python3 -I scripts/render_replay.py TRACE/submit-N OUT [--reps N] [--dump VA:SIZE ...] [--patch VA=HEX ...]
                                                           [--field name=value ...]

Library:  import render_replay as rr
          res = rr.replay(trace_dir, out_dir, patches={va: bytes}, fields={'width_px': 64}, dumps=[(va, size)], reps=3)
          res = {'reps': [{'vtx_ns','frag_ns',...}], 'dumps': {va: bytes}, 'ok': bool}

Safety (hang budget, scripts/gpu_safety.py): every run is bounded (syncobj wait 2 s, process timeout 20 s); a wait failure
or timeout is recorded against the boot's budget and further runs refuse once the budget is spent. Mutations only change
data the captured pass already uses (vertex data, state words within known fields); pointers stay inside captured buffers.
"""
import json, os, struct, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpu_safety
import render_decode as rd

REPO = HERE.parent
BIN = Path(os.environ.get('AGX_RENDER_REPLAY', '/scratch/render/v2/drm_render_replay'))


def build():
    src = REPO / 'tests/drm_render_replay.c'
    if not BIN.exists() or BIN.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(['gcc', '-std=gnu11', '-O2', '-Wall', '-o', str(BIN), str(src)], check=True)


def replay(trace_dir, out_dir, patches=None, fields=None, dumps=(), reps=1, wait_ms=2000, cmdbuf=None, after=False, train=False):
    build()
    gpu_safety.require_budget()
    trace_dir, out_dir = Path(trace_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = rd.Trace(trace_dir, after=after)
    man = t.man
    # Memory images, patched.
    images = {}
    for b in man['binds']:
        va = int(b['va'], 16)
        p = trace_dir / f'bo-{va:x}{".after" if after else ""}.bin'
        images[va] = bytearray(p.read_bytes())
    for pva, data in (patches or {}).items():
        for va, img in images.items():
            if va <= pva and pva + len(data) <= va + len(img):
                img[pva - va:pva - va + len(data)] = data
                break
        else:
            raise KeyError(f'patch outside captured memory: {pva:#x}+{len(data)}')
    # Command buffer, with render fields overridden.
    cb = bytearray(cmdbuf if cmdbuf is not None else t.cmdbuf)
    if fields:
        off = 0
        while off + 8 <= len(cb):
            ctype, size = struct.unpack_from('<HH', cb, off)
            if ctype == 0:
                d = rd.decode_render(bytes(cb[off + 8:off + 8 + size]))
                for k, v in fields.items():
                    assert k in d, k
                    d[k] = v
                cb[off + 8:off + 8 + rd.struct.calcsize(rd.RENDER_FMT)] = rd.pack_render(d)
            off += 8 + size
    lines = [f"usc_base {man['usc_exec_base'][2:]}", f'wait_ms {wait_ms}', f'reps {reps}'] + (['train 1'] if train else [])
    ids = {}
    for b in man['binds']:
        h = b['handle']
        if h not in ids:
            ids[h] = len(ids)
            lines.append(f"bo {ids[h]} {b['bo_size']}")
        va = int(b['va'], 16)
        f = out_dir / f'in-{va:x}.bin'
        f.write_bytes(images[va])
        lines.append(f"fill {ids[h]} {b['bo_offset']} {f}")
    for b in man['binds']:
        lines.append(f"bind {ids[b['handle']]} {b['va'][2:]} {b['range']} {b['bo_offset']} {b['flags']}")
    (out_dir / 'cmdbuf.bin').write_bytes(cb)
    lines.append(f"cmdbuf {out_dir / 'cmdbuf.bin'}")
    for va, size in dumps:
        lines.append(f'dump {va:x} {size} {out_dir}/out-{va:x}.bin')
    (out_dir / 'plan.txt').write_text('\n'.join(lines) + '\n')
    try:
        p = subprocess.run([str(BIN), '/dev/dri/renderD128', str(out_dir / 'plan.txt')], capture_output=True, text=True,
                           timeout=20 + reps * wait_ms / 1000)
        out, code = p.stdout, p.returncode
    except subprocess.TimeoutExpired as e:
        gpu_safety.record(f'render_replay timeout {trace_dir}')
        return {'ok': False, 'error': 'process-timeout', 'reps': [], 'dumps': {}}
    recs = [json.loads(l) for l in out.splitlines() if l.startswith('{')]
    if code == 2 or any(r.get('failure') == 'syncobj-wait' for r in recs):
        gpu_safety.record(f'render_replay wait-fail {trace_dir} fields={fields} patches={len(patches or {})}')
    res = {'ok': code == 0 and any(r.get('completed') for r in recs), 'reps': [r for r in recs if 'rep' in r],
           'log': recs, 'stderr': p.stderr[-2000:], 'dumps': {}}
    for va, size in dumps:
        f = out_dir / f'out-{va:x}.bin'
        if f.exists():
            res['dumps'][va] = f.read_bytes()
    # The inputs stay next to the plan: rust/target/release/agx_render PLAN simulates exactly what the GPU got.
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('trace'); ap.add_argument('out')
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--dump', action='append', default=[])
    ap.add_argument('--patch', action='append', default=[])
    ap.add_argument('--field', action='append', default=[])
    a = ap.parse_args()
    dumps = [(int(d.split(':')[0], 16), int(d.split(':')[1], 0)) for d in a.dump]
    patches = {int(x.split('=')[0], 16): bytes.fromhex(x.split('=')[1]) for x in a.patch}
    fields = {x.split('=')[0]: int(x.split('=')[1], 0) for x in a.field}
    r = replay(a.trace, a.out, patches, fields, dumps, a.reps)
    print(json.dumps({k: v for k, v in r.items() if k != 'dumps'}))


if __name__ == '__main__':
    main()
