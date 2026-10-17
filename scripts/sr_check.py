#!/usr/bin/env python3
"""Offline: the functional simulator's special registers against the M2's, lane for lane.

results/2026-10-16-special-registers holds six direct-DRM kernels (the 256-selector scan inside 0-3 nested ifs;
playground scripts/run_sr_divergence.py) and the GPU's readback (hardware.json.gz: per pattern, the active lanes and
every selector's value in each). Each kernel runs in agx_run with the launch's register allocation (32), its VM slot
(read back from the hardware) and pipeline offset (0x1000); every selector except SR17 / SR20 (placement) and SR113
(unexplained, read as AGX_SR_DEFAULT) must match on every active lane.
"""
import gzip, json, os, struct, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'results/2026-10-16-special-registers'
RUN = os.environ.get('AGX_RUN', str(ROOT/'rust/target/release/agx_run'))
SENTINEL, LANES, OUT_VA, TABLE_VA = 0xcdcdcdcd, 64, 0x1100100000, 0x1100200000


def main():
    hw = json.load(gzip.open(D/'hardware.json.gz', 'rt'))
    patterns = hw['patterns']
    expected = {'uniform': 64, 'depth1': 32, 'depth2': 16, 'depth3': 8, 'quad': 16, 'quad-depth2': 8}
    if set(patterns) != set(expected):
        raise ValueError('expected all six retained special-register patterns')
    for name, p in patterns.items():
        active = p['active_lanes']
        if len(active) != expected[name] or len(set(active)) != len(active) or any(type(i) is not int or not 0 <= i < LANES for i in active):
            raise ValueError(f'{name}: invalid/empty active lanes')
        if set(p['values']) != {str(s) for s in range(256)} or any(len(v) != len(active) for v in p['values'].values()):
            raise ValueError(f'{name}: missing selectors or incomplete lane values')
    total = bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp/'tab.bin').write_bytes(struct.pack('<I', SENTINEL)*(256*LANES))
        out = [SENTINEL]*(LANES+16)+[0xa5a50000+j for j in range(16)]
        (tmp/'out.bin').write_bytes(struct.pack(f'<{len(out)}I', *out))
        for name, p in patterns.items():
            cmd = [RUN, str(D/f'{name}.bin'), '--grid', str(LANES), '--wg', '32', '--uniform', '4=0', '--uniform', f'5={LANES}',
                   '--uniform', f'6={OUT_VA & 0xffffffff}', '--uniform', f'7={OUT_VA >> 32}', '--uniform', f'8={TABLE_VA & 0xffffffff}',
                   '--uniform', f'9={TABLE_VA >> 32}', '--map', f'{OUT_VA}={tmp/"out.bin"}', '--map', f'{TABLE_VA}={tmp/"tab.bin"}',
                   '--dump', f'{TABLE_VA}={256*LANES*4}:{tmp/"sim.tab"}', '--alloc-regs', '32', '--vm-slot', str(p['values']['21'][0]),
                   '--pipeline', '0x1000']
            r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, AGX_SR_DEFAULT='0'))
            if r.returncode:
                print(f'FAIL {name}: agx_run: {(r.stdout + r.stderr).strip()[:300]}')
                bad += 1
                continue
            t = struct.unpack(f'<{256*LANES}I', (tmp/'sim.tab').read_bytes())
            act = p['active_lanes']
            miss = []
            for s in range(256):
                if s in (17, 20, 113):
                    continue
                total += 1
                sim = [t[s*LANES+i] for i in act]
                if sim != p['values'][str(s)]:
                    miss.append(s)
            inactive_ok = all(t[s*LANES+i] == SENTINEL for s in range(256) for i in range(LANES) if i not in act)
            bad += len(miss) + (not inactive_ok)
            print(f'{"ok  " if not miss and inactive_ok else "FAIL"} {name:12s} {len(act):2d} active lanes, 253 selectors'
                  + (f', mismatching {miss}' if miss else '') + ('' if inactive_ok else ', inactive lanes wrote'))
    print(f'{total} selector x pattern comparisons, {bad} failures')
    return 1 if bad or total != 1518 else 0


if __name__ == '__main__':
    sys.exit(main())

