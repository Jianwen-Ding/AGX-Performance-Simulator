#!/usr/bin/env python3
"""Compare a hardware replay output (out-<va>.bin) with the simulator's (out-<va>.bin.sim) as RGBA8 pixels.

    python3 -I scripts/render_compare.py OUT_DIR WIDTH HEIGHT [--va HEX] [--bpp 4] [--tol 1]
Prints {"pixels", "exact", "within_tol", "max_diff", "first_bad": [...]}.
"""
import json, sys
from pathlib import Path


def compare(hw, sim, W, H, bpp=4, tol=1):
    n = W * H
    exact = within = 0; mx = 0; bad = []
    for i in range(n):
        a, b = hw[bpp * i:bpp * i + bpp], sim[bpp * i:bpp * i + bpp]
        d = max(abs(x - y) for x, y in zip(a, b))
        mx = max(mx, d)
        exact += d == 0
        within += d <= tol
        if d > tol and len(bad) < 12:
            bad.append({'x': i % W, 'y': i // W, 'hw': a.hex(), 'sim': b.hex()})
    return {'pixels': n, 'exact': exact, 'within_tol': within, 'max_diff': mx, 'first_bad': bad}


def main():
    d = Path(sys.argv[1]); W, H = int(sys.argv[2]), int(sys.argv[3])
    args = sys.argv[4:]
    opt = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
    outs = sorted(d.glob('out-*.bin')) if '--va' not in opt else [d / f"out-{opt['--va']}.bin"]
    hw = outs[0].read_bytes(); sim = Path(str(outs[0]) + '.sim').read_bytes()
    print(json.dumps(compare(hw, sim, W, H, int(opt.get('--bpp', 4)), int(opt.get('--tol', 1)))))


if __name__ == '__main__':
    main()
