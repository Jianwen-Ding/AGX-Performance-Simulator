#!/usr/bin/env python3
"""Render-model check: hardware (direct-ioctl replay of a captured pass, optionally mutated) vs agx_render.

    python3 -I scripts/render_check.py scene SCENE.txt [--name N]      # capture with gles_scene, replay, simulate, compare
    python3 -I scripts/render_check.py trace TRACE/submit-N [--name N]  # an existing capture
    python3 -I scripts/render_check.py suite SUITE.json                 # many cases, results to results/.../cases.jsonl

Each case: GL readback (scene runs only) vs hardware replay (checks the replay itself), hardware replay vs simulator
(pixels exact / within 1, max diff), firmware vertex / fragment times vs agx_render's prediction.
Freeze safety: the GPU side is bounded by render_replay (2 s waits, hang budget); the simulator runs under a 120 s
timeout and an 8 GB address-space limit, so a model bug that loops or balloons is a failed case, not a stuck machine.
"""
import json, os, resource, struct, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_decode as rd
import render_probes as rp
import render_compare as rc
import gpu_safety

REPO = HERE.parent
WORK = Path(os.environ.get('RENDER_WORK', '/scratch/render/v2'))
SCENE_BIN = WORK / 'gles_scene'
SHIM = WORK / 'render_trace.so'
AGX_RENDER = REPO / 'rust/target/release/agx_render'
SIM_TIMEOUT = int(os.environ.get('RENDER_SIM_TIMEOUT', '120'))
SIM_MEM = int(os.environ.get('RENDER_SIM_MEM_GB', '8')) << 30


def build_tools():
    if not SHIM.exists() or SHIM.stat().st_mtime < (REPO / 'tests/render_trace.c').stat().st_mtime:
        subprocess.run(['gcc', '-O2', '-fPIC', '-shared', '-o', str(SHIM), str(REPO / 'tests/render_trace.c'), '-ldl'], check=True)
    if not SCENE_BIN.exists() or SCENE_BIN.stat().st_mtime < (REPO / 'tests/gles_scene.c').stat().st_mtime:
        subprocess.run(['gcc', '-std=gnu11', '-O2', '-I', os.environ.get('MESA_INCLUDE', '/save/workspace/agx-notes/mesa/include'),
                        '-o', str(SCENE_BIN), str(REPO / 'tests/gles_scene.c'), '-ldl'], check=True)


def capture(scene, name):
    """Run the scene under the trace shim; returns (trace dir of the last frame, GL readback bytes or None)."""
    build_tools()
    cap = WORK / 'cap' / name
    if cap.exists():
        subprocess.run(['rm', '-rf', str(cap)], check=True)
    cap.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RENDER_TRACE_DIR=str(cap), RENDER_TRACE_MAX='2', LD_PRELOAD=str(SHIM))
    gl = WORK / 'cap' / f'{name}.gl.rgba'
    p = subprocess.run([str(SCENE_BIN), str(scene), str(gl), '2'], env=env, capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise RuntimeError(f'gles_scene failed: {p.stderr[-500:]}')
    subs = sorted(cap.glob('submit-*'))
    if not subs:
        raise RuntimeError('no render submit captured')
    return subs[-1], (gl.read_bytes() if gl.exists() else None)


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (SIM_MEM, SIM_MEM))


def simulate(plan, extra=()):
    t0 = time.time()
    try:
        p = subprocess.run([str(AGX_RENDER), str(plan), *extra], capture_output=True, text=True, timeout=SIM_TIMEOUT, preexec_fn=limits)
    except subprocess.TimeoutExpired:
        return {'error': f'simulator timeout {SIM_TIMEOUT}s'}, time.time() - t0
    lines = [l for l in p.stdout.splitlines() if l.startswith('{')]
    if p.returncode or not lines:
        return {'error': (lines[-1] if lines else p.stderr[-500:]) or f'exit {p.returncode}'}, time.time() - t0
    return json.loads(lines[-1]), time.time() - t0


def flip_rows(img, W, H):
    return b''.join(img[4 * W * (H - 1 - y):4 * W * (H - y)] for y in range(H))


def check(trace, name, patches=None, fields=None, gl=None, reps=5, tol=1):
    pa = rp.Pass(trace)
    out = WORK / 'runs' / name
    # timing at the top clock (clock keeper warm-up, short back-to-back bursts); the image of the last burst
    r = rp.run_warm(pa, out, patches, fields)
    rec = {'name': name, 'trace': str(trace), 'hw_ok': r['ok'], 'W': pa.width, 'H': pa.height}
    if not r['ok']:
        rec['hw_error'] = r.get('error') or r.get('log')
        return rec
    rec['hw_vtx_ns'], rec['hw_frag_ns'] = r['warm_vtx_ns'], r['warm_frag_ns']
    rec['hw_reps_cold'] = [(x['vtx_ns'], x['frag_ns']) for x in r['reps'][:3]]
    if gl is not None and not patches and not fields:
        c = rc.compare(gl, r["image"], pa.width, pa.height, 4, 0)  # Mesa FBOs are y-up in memory: no flip
        rec['replay_vs_gl_exact'] = c['exact']
    sim, dt = simulate(out / 'plan.txt')
    rec['sim_s'] = round(dt, 3)
    if 'error' in sim:
        rec['sim_error'] = sim['error']
        return rec
    simimg = pa.unpad((out / f'out-{pa.fb_va:x}.bin.sim').read_bytes())
    c = rc.compare(r['image'], simimg, pa.width, pa.height, 4, tol)
    rec.update({'pixels': c['pixels'], 'exact': c['exact'], 'within_tol': c['within_tol'], 'max_diff': c['max_diff'],
                'first_bad': c['first_bad'][:4]})
    rec['sim'] = {k: v for k, v in sim.items() if k != 'log'}
    for k in ('predicted_vtx_ns', 'predicted_frag_ns'):
        rec[k] = sim.get(k)
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['scene', 'trace', 'suite', 'resim'])
    ap.add_argument('path'); ap.add_argument('--name'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.mode == 'scene':
        name = a.name or Path(a.path).stem
        tr, gl = capture(a.path, name)
        print(json.dumps(check(tr, name, gl=gl)))
    elif a.mode == 'trace':
        print(json.dumps(check(a.path, a.name or Path(a.path).parent.name)))
    elif a.mode == 'resim':
        # offline regression: re-simulate every stored replay of a cases file, compare with the stored hardware output
        for l in open(a.path):
            r = json.loads(l)
            if 'hw_vtx_ns' not in r:
                print(json.dumps(r)); continue
            out = WORK / 'runs' / r['name']
            pa = rp.Pass(r['trace'])
            sim, dt = simulate(out / 'plan.txt')
            if 'error' in sim:
                r.update({'sim_error': sim['error']}); [r.pop(k, None) for k in ('pixels', 'exact')]
            else:
                hw = pa.unpad((out / f'out-{pa.fb_va:x}.bin').read_bytes())
                c = rc.compare(hw, pa.unpad((out / f'out-{pa.fb_va:x}.bin.sim').read_bytes()), pa.width, pa.height, 4, 1)
                r.update({'pixels': c['pixels'], 'exact': c['exact'], 'within_tol': c['within_tol'], 'max_diff': c['max_diff'],
                          'first_bad': c['first_bad'][:4], 'sim_s': round(dt, 3), 'predicted_vtx_ns': sim.get('predicted_vtx_ns'),
                          'predicted_frag_ns': sim.get('predicted_frag_ns'), 'sim': {k: v for k, v in sim.items() if k != 'log'}})
            print(json.dumps(r), flush=True)
    else:
        suite = json.loads(Path(a.path).read_text())
        outp = Path(a.out) if a.out else None
        for case in suite:
            try:
                with gpu_safety.Tripwire(f"render case {case['name']}"):
                    if 'scene' in case:
                        tr, gl = capture(case['scene'], case['name'])
                    else:
                        tr, gl = case['trace'], None
                    rec = check(tr, case['name'], gl=gl)
            except RuntimeError as e:
                if 'GPU recovered' in str(e):
                    print(json.dumps({'name': case['name'], 'error': str(e)}), flush=True)
                    sys.exit(3)   # a recovery stops the whole suite: no further submissions on a struggling GPU
                rec = {'name': case['name'], 'error': str(e)}
            except Exception as e:
                rec = {'name': case['name'], 'error': str(e)}
            print(json.dumps(rec), flush=True)
            if outp:
                with outp.open('a') as f:
                    f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
