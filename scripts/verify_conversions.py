#!/usr/bin/env python3
"""Offline: exhaustive / edge-dense convert recordings (safe-01 part 1) replayed in the simulator.

results/2026-10-16-conversions/hw_samples.json.gz keeps, per form (80: modes 0, 1, 4-11 x round 0 / 1 x 16/32-bit
destination and source), ~12k sampled inputs with their hardware words (both ends of the input list plus random
picks) and the SHA-256 of the whole hardware result array. This check
  1. regenerates the inputs (record_convert.form_inputs) and replays the sampled ones through agx_run with the exact
     recorded kernel (one 2^18-lane chunk per form): every word must equal the hardware word;
  2. applies the numpy model (convert_check.model) to the samples;
  3. when /scratch/agxsim-data/convert/<form>.npz exists, re-hashes it and runs the model over every input.
"""
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings('ignore')
import coverage as COV
import record_convert as RC, convert_check as CC, record_transcendentals as T, run_drm_compute as drm

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = json.load(gzip.open(ROOT/'results/2026-10-16-conversions/hw_samples.json.gz', 'rt'))
    bad = full = 0
    for f in RC.FORMS:
        nm = RC.name(*f)
        e = d['forms'][nm]
        x = RC.form_inputs(f)
        if len(x) != e['inputs']:
            print(f'FAIL {nm}: input count changed'); bad += 1; continue
        idx = np.array(e['idx']); hw = np.array(e['hw'], np.uint32); xs = x[idx]
        C = T.CHUNK
        table = [int(v) for v in xs]+[0]*(C-len(xs))+[0]*C
        _, tab = COV.simulate(drm.build_main(RC.body(*f)), C, T.WG, 64, 0, table)
        sim = np.array(tab[C:C+len(xs)], np.uint32)
        mo = CC.model(*f, xs).astype(np.uint32)
        nb = int(np.sum(sim != hw)) + int(np.sum(mo != hw))
        extra = ''
        p = RC.OUT/(nm+'.npz')
        if p.exists():
            fh = np.load(p)['hw'].astype(np.uint32)
            ok = hashlib.sha256(fh.tobytes()).hexdigest() == e['hw_sha256'] and (CC.model(*f, x).astype(np.uint32) == fh).all()
            nb += not ok; full += 1; extra = f', full {len(x)} {"ok" if ok else "FAIL"}'
        bad += nb
        print(f'{"ok  " if not nb else "FAIL"} {nm:45s} {len(xs):6d} samples in agx_run{extra}', flush=True)
    print(f'{d["total_inputs"]} inputs recorded; {full} forms re-checked in full; {bad} failures')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
