#!/usr/bin/env python3
"""Offline render-model regression on stored direct-ioctl replays (no GPU): `make verify-render-model`.

    python3 -I scripts/render_fixtures.py pack NAME... [--runs /scratch/render/v2/runs]   # store replays as fixtures
    python3 -I scripts/render_fixtures.py verify                                           # simulate each, compare with its hardware output

A fixture is one replay directory (plan.txt, the memory images and command buffer the GPU was given, the hardware's framebuffer
out-<va>.bin) plus its record from the results file (measured vertex / fragment times), tarred and gzipped under
results/2026-11-21-render-ioctl/fixtures/. The simulator runs under the same 120 s / 8 GB limits as render_check.
"""
import io, json, os, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIX = REPO / 'results/2026-11-21-render-ioctl/fixtures'
sys.path.insert(0, str(HERE))


def records():
    out = {}
    for f in (REPO / 'results/2026-11-21-render-ioctl').glob('*.jsonl'):
        for l in open(f):
            r = json.loads(l)
            out[r['name']] = r
    return out


def pack(names, runs):
    FIX.mkdir(parents=True, exist_ok=True)
    recs = records()
    for n in names:
        d = Path(runs) / n
        with tarfile.open(FIX / f'{n}.tar.gz', 'w:gz') as t:
            for f in sorted(d.iterdir()):
                if f.name.endswith('.sim'):
                    continue
                t.add(f, arcname=f'{n}/{f.name}')
            meta = json.dumps({k: recs[n].get(k) for k in ('name', 'W', 'H', 'hw_vtx_ns', 'hw_frag_ns', 'trace')}).encode()
            ti = tarfile.TarInfo(f'{n}/record.json'); ti.size = len(meta); t.addfile(ti, io.BytesIO(meta))
        print(n, (FIX / f'{n}.tar.gz').stat().st_size)


def verify():
    import render_check as rc, render_compare as cmp_
    ok = True
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for tz in sorted(FIX.glob('*.tar.gz')):
            with tarfile.open(tz) as t:
                t.extractall(tmp, filter='data')
            n = tz.name[:-7]; d = Path(tmp) / n
            rec = json.loads((d / 'record.json').read_text())
            plan = (d / 'plan.txt').read_text()
            plan = re.sub(r'(\S+/)(in-[0-9a-f]+\.bin|cmdbuf\.bin|out-[0-9a-f]+\.bin)', lambda m: str(d / m.group(2)), plan)
            (d / 'plan.txt').write_text(plan)
            sim, dt = rc.simulate(d / 'plan.txt')
            if 'error' in sim:
                print(n, 'SIM ERROR', sim['error']); ok = False; continue
            W, H = rec['W'], rec['H']
            outs = [f for f in d.glob('out-*.bin')]
            stride = (W * 4 + 15) & ~15
            unpad = lambda b: b''.join(b[y * stride:y * stride + W * 4] for y in range(H))
            c = cmp_.compare(unpad(outs[0].read_bytes()), unpad(Path(str(outs[0]) + '.sim').read_bytes()), W, H, 4, 1)
            ev = abs(sim['predicted_vtx_ns'] / rec['hw_vtx_ns'] - 1); ef = abs(sim['predicted_frag_ns'] / rec['hw_frag_ns'] - 1)
            good = c['within_tol'] == c['pixels']
            ok &= good
            rows.append((ev, ef))
            print(f"{n:24s} pixels {c['exact']}/{c['pixels']} exact, max diff {c['max_diff']}  vtx {rec['hw_vtx_ns']} vs {sim['predicted_vtx_ns']:.0f}"
                  f"  frag {rec['hw_frag_ns']} vs {sim['predicted_frag_ns']:.0f}  sim {dt:.2f}s {'OK' if good else 'FAIL'}")
    ev = sorted(r[0] for r in rows); ef = sorted(r[1] for r in rows)
    print(json.dumps({'fixtures': len(rows), 'functional_ok': ok, 'vtx_median_err': round(ev[len(ev) // 2], 3), 'frag_median_err': round(ef[len(ef) // 2], 3)}))
    return ok


if __name__ == '__main__':
    if sys.argv[1] == 'pack':
        runs = sys.argv[sys.argv.index('--runs') + 1] if '--runs' in sys.argv else '/scratch/render/v2/runs'
        pack([a for a in sys.argv[2:] if not a.startswith('--') and a != runs], runs)
    else:
        sys.exit(0 if verify() else 1)
