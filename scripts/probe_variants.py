#!/usr/bin/env python3
"""Run instruction variants the simulator cannot execute, with every surrounding instruction checked in agx_run first.

A probe is `prefix + X + suffix`. X is either a *surrogate* (an encoding agx_run executes and the GPU has run) or the
*variant* under test. Before a variant reaches the GPU:

  1. the variant differs from the surrogate only inside the probe's allowed bit mask of X (the rest of the program
     is byte-identical);
  2. the surrogate program passes the decode gate and runs to its end in agx_run with the harness layout, and its
     output and table equal the probe's host oracle for the surrogate (addresses in range, stores where expected,
     no loop);
  3. the hang budget has room.

The variant then runs twice through tests/drm_compute.c (record_suite.run: whole program from build_main, final stop
checked, faults recorded in the hang budget). The first fault stops the batch. Evidence (code, input table, output,
table after, firmware time) goes to --out as one JSON per variant; analysis is the caller's (see probe_atomics.py).
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import coverage as cov                                    # noqa: E402  (agx_asm-backed encoders, gate, simulate)

SCRATCH = Path(os.environ.get('AGX_SCRATCH', '/scratch/apple-m2-playground')) / 'probe-variants'


class Probe:
    """prefix/suffix: bytes around the probed instruction. surrogate / variant: instruction bytes of equal length.
    allowed: bit mask (int over the instruction) the variant may change. oracle(out, table_after) -> bool for the
    surrogate run."""

    def __init__(self, name, prefix, surrogate, variant, suffix, allowed, n, wg, regs, table, oracle, shared=0):
        self.__dict__.update(name=name, prefix=prefix, surrogate=surrogate, variant=variant, suffix=suffix,
                             allowed=allowed, n=n, wg=wg, regs=regs, table=table, oracle=oracle, shared=shared)

    def body(self, x):
        return self.prefix + x + self.suffix


def check(p):
    """steps 1-2; returns the simulated (out, table) of the surrogate program"""
    import run_drm_compute as drm
    if len(p.surrogate) != len(p.variant):
        raise RuntimeError(f'{p.name}: variant and surrogate lengths differ')
    diff = int.from_bytes(p.surrogate, 'little') ^ int.from_bytes(p.variant, 'little')
    if diff & ~p.allowed:
        raise RuntimeError(f'{p.name}: variant changes bits outside the allowed mask: {diff & ~p.allowed:#x}')
    cov.gate(p.body(p.surrogate))
    out, tab = cov.simulate(drm.build_main(p.body(p.surrogate)), p.n, p.wg, p.regs, p.shared, p.table)
    if not p.oracle(out, tab):
        raise RuntimeError(f'{p.name}: the surrogate program does not match its host oracle in agx_run')
    return out, tab


def run(probes, out_dir, dry=False, repeats=2):
    """check every probe offline first, then run the variants on the GPU; returns the list of evidence records"""
    import gpu_safety, record_suite as RS, run_bench as B, run_drm_compute as drm
    cov.install()
    sims = {p.name: check(p) for p in probes}
    print(json.dumps({'checked_in_agx_run': len(sims)}), flush=True)
    if dry:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    R = B.Runner(SCRATCH / 'run')
    recs = []
    for p in probes:
        path = out_dir / f'{p.name}.json'
        if path.exists():
            recs.append(json.loads(path.read_text())); continue
        gpu_safety.require_budget(1)
        code = drm.build_main(p.body(p.variant))
        rec = {'name': p.name, 'variant_hex': p.variant.hex(), 'surrogate_hex': p.surrogate.hex(), 'code': code.hex(),
               'n': p.n, 'wg': p.wg, 'regs': p.regs, 'table_in': p.table, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            res = [RS.run(R, code, p.n, p.wg, p.table, p.regs, p.shared) for _ in range(repeats)]
        except Exception as ex:
            rec['fault'] = str(ex)[:600]
            path.write_text(json.dumps(rec) + '\n')
            print(json.dumps({'probe': p.name, 'fault': rec['fault'][:200]}), flush=True)
            sys.exit(f'stopped after the fault in {p.name}: check the kernel log and a control dispatch before resuming')
        rec.update(out=res[0][0], table_after=res[0][1], fw_ns=[r[2] for r in res],
                   deterministic=all(r[0] == res[0][0] and r[1] == res[0][1] for r in res))
        path.write_text(json.dumps(rec) + '\n')
        print(json.dumps({'probe': p.name, 'ok': True, 'deterministic': rec['deterministic'], 'fw_ns': rec['fw_ns']}), flush=True)
        recs.append(rec)
    return recs
