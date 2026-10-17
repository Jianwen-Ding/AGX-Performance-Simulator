#!/usr/bin/env python3
"""Benchmark functional get_sr execution with exact output checks; no GPU needed.

Build agx_run first. --baseline compares an earlier agx_run binary.
Timing includes process startup, execution and a dump of the global-X result.
"""
import argparse
import json
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'rust/target/release/agx_run')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--groups', type=int, default=1024)
    parser.add_argument('--reads', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.groups, args.reads, args.repeats) < 1:
        parser.error('groups, reads and repeats must be positive')
    # kernels::get_sr(80, 2), followed by a store of global X at word global X.
    sr = (0x72 | 2 << 7 | 4 << 9 | (80 & 63) << 16 | (80 >> 6) << 26).to_bytes(4, 'little')
    # kernels::store(2, 8, 2): the control store's R/A/O decode fields patched
    # from half indices 4/12/2 to 4/16/4, including the base's extension nibble.
    store = int.from_bytes(bytes.fromhex('45112c0e00c01200'), 'little')
    store = (store & ~((15 << 16) | (15 << 20) | (15 << 32) | (15 << 36))) | 4 << 20 | 1 << 36
    code = sr * args.reads + store.to_bytes(8, 'little') + bytes.fromhex('8800')
    n = args.groups * 32
    expected = struct.pack(f'<{n}I', *range(n))
    result = {'groups': args.groups, 'reads_per_group': args.reads, 'repeats': args.repeats}
    with tempfile.TemporaryDirectory(prefix='agx-sr-bench-') as tmp:
        tmp = Path(tmp)
        (tmp / 'code.bin').write_bytes(code)
        targets = ([('baseline', args.baseline)] if args.baseline else []) + [('current', args.run)]
        for label, exe in targets:
            samples = []
            for _ in range(args.repeats + 1):  # discard one warm-up run
                start = time.perf_counter()
                run = subprocess.run([str(exe.resolve()), str(tmp / 'code.bin'), '--grid', str(n), '--wg', '32',
                    '--regs', '8', '--uniform', '8=65536', '--uniform', '9=0', '--alloc', f'65536={4 * n}',
                    '--dump', f'65536={4 * n}:{tmp / "out.bin"}'], capture_output=True, text=True, timeout=120, check=True)
                samples.append(1000 * (time.perf_counter() - start))
                if (tmp / 'out.bin').read_bytes() != expected:
                    raise AssertionError(f'{label}: global-X output differs')
                if json.loads(run.stdout)['instructions'] != (args.reads + 2) * args.groups:
                    raise AssertionError(f'{label}: instruction count differs')
            result[label + '_median_ms'] = statistics.median(samples[1:])
    if args.baseline:
        result['speedup'] = result['baseline_median_ms'] / result['current_median_ms']
    print(json.dumps(result))


if __name__ == '__main__':
    main()
