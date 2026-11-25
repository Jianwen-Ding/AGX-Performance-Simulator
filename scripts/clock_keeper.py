#!/usr/bin/env python3
"""Hold the GPU at its top clock while measuring (playground novel-results/2026-11-10-gpu-clock-ladder: the clock climbs 8 states
over ~35 ms of continuous work and idle gaps pin it low, so a short render pass measured alone runs at 1.4-3x its top-clock time).

    with clock_keeper.keep():
        ... measure ...
SAFETY (2026-11-21): the keeper must never be killed while its commands are in flight. SIGKILL of drm_compute tore its VM down
under the GPU: gCDM_CS faults at its control stream (unmapped), the firmware then recovered ~1000 times a second for 3 minutes.
The loop now ends on a stop file and the running drm_compute finishes (waits for its fence, frees its VM) before keep() returns.
A separate process loops tests/drm_compute.c over a dependent-iadd kernel of ONE SIMD group, 1000 commands back to back per run
(AGX_TRAIN): it occupies one SIMD-group slot of one core, so the measured work loses at most ~1/38 of a core. The keeper is killed
(process group) on exit, also on exceptions; it is bounded by its own 3 s waits and the hang budget like every other GPU run.
"""
import contextlib, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORK = Path(os.environ.get('RENDER_WORK', '/scratch/render/v2'))


def _build():
    exe = WORK / 'drm_compute'
    if not exe.exists() or exe.stat().st_mtime < (REPO / 'tests/drm_compute.c').stat().st_mtime:
        subprocess.run(['gcc', '-std=gnu11', '-O2', '-o', str(exe), str(REPO / 'tests/drm_compute.c'), '-lpthread'], check=True)
    k = WORK / 'keeper.bin'
    if not k.exists():
        asm = subprocess.run([str(REPO / 'rust/target/release/agx_asm')], input='iadd r5, r6, 1\nstop\n', capture_output=True, text=True, check=True).stdout.split()
        k.write_bytes(bytes.fromhex(asm[0]) * 20000 + bytes.fromhex(asm[1]))
    return exe, k


@contextlib.contextmanager
def keep(warm_s=0.06):
    exe, k = _build()
    env = dict(os.environ, AGX_TRAIN='200', AGX_TRAIN_BATCH='64')
    run = WORK / f'keeper.run.{os.getpid()}'
    run.write_text('1')
    p = subprocess.Popen(['sh', '-c', f'while [ -e {run} ]; do {exe} /dev/dri/renderD128 {k} 32 /dev/null 1 >/dev/null 2>&1; done'],
                         env=env, start_new_session=True)
    try:
        time.sleep(warm_s)   # >= 40 ms of continuous work reaches the top state
        yield p
    finally:
        run.unlink(missing_ok=True)      # graceful: the loop ends after the running drm_compute completes its commands
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # still running after 10 s means the GPU is stuck: leave it (a kill would fault the GPU) and report it
            raise RuntimeError('clock keeper did not finish within 10 s; not killed (would fault the GPU)')


if __name__ == '__main__':
    with keep():
        time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 1)
