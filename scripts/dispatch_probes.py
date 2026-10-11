#!/usr/bin/env python3
"""Whole-dispatch validation of agx_dispatch (direct DRM, known-safe encodings only).

Each case is a main body (counted loops of benchmarked forms, or the grid-stride streaming body of the
load-latency finding) wrapped in the harness's control main and dispatched with a given grid, workgroup
and register count. The firmware start/end timestamps give the dispatch duration; `repeats` dispatches run
back to back and the minimum is taken (the GPU then sits at its top DVFS level, 1398 MHz). The same binary
goes through `rust/target/release/agx_dispatch` at 1398 MHz with the harness's uniforms.

Usage: dispatch_probes.py [--cases NAME,...] [--repeats 12] [--name dispatch.json]
"""
import argparse, json, statistics, subprocess, time
from pathlib import Path
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm
import mem_probes as M

e, mov = agx_loop.encode, B.mov
ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT/'rust/target/release/agx_dispatch'


def alu(op, chains, steps, trips):
    return B.loop(op, chains, steps, trips)


def stream(trips, steps, stride_words):
    return agx_loop.stream_body(trips, steps, stride_words)


def diverge(k, n_then, n_else, trips):
    """Counted loop whose body is if (lane < k) { n_then fadd32 on r5..r12 } else { n_else fadd32 on r13..r20 },
    Mesa's structured form with jmp_exec_none over an empty side (else: lanes > k-1, signed, so k = 0 works)."""
    head = e('and r23, r1, 31')+mov(22, k)+mov(21, (k-1) & 0xffffffff)
    head += b''.join(mov(r, 0x3fc00000) for r in range(5, 21))
    head += mov(3, 0)+mov(4, trips)+e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    then = b''.join(e(f'fadd32 r{5 + i % 8}, r{5 + i % 8}, 0.125') for i in range(n_then))
    els = b''.join(e(f'fadd32 r{13 + i % 8}, r{13 + i % 8}, 0.125') for i in range(n_else))
    jt, je = 6+len(then), 6+len(els)
    body = e('if_icmp r0l, ult, r23, r22, 1')+e(f'jmp_exec_none pc+{jt}', f'jmp_exec_none pc+{jt}')+then
    body += e('else_icmp r0l, sgt, r23, r21, 1')+e(f'jmp_exec_none pc+{je}', f'jmp_exec_none pc+{je}')+els+e('pop_exec r0l, 1')
    body += e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
    return head+body+e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}')+e('pop_exec r0l, 1')+e('iadd r2, r5, 0')


def diverge_gate():
    """Emulator: lanes < k ran the then-side every trip, the others the else-side; all lanes re-enabled."""
    import struct
    for k in (0, 8, 16, 32):
        b = diverge(k, 3, 2, 4)+e('iadd r2, r5, 0')
        r = agx_loop.emulate(b, 32, lane_ids=True, num_registers=32)
        f = lambda x: struct.unpack('<f', struct.pack('<I', x))[0]
        # r5 gets 0.125 per then-step on chain 0 (steps 0 of 3 -> 1 per trip); r2 = r5
        want = [1.5+(0.125*4 if lane < k else 0) for lane in range(32)]
        got = [f(x) for x in r['r2']]
        if got != want or not all(r['exec']):
            raise RuntimeError(f'diverge gate k={k}: {got[:4]}... want {want[:4]}')


def cases():
    """Each case is sized to run for several ms at the top clock (shorter dispatches often stay at a low DVFS
    level for the whole run: the minimum of 12 is then not the top clock)."""
    out = []
    # empty main: per-group launch cost (huge grids)
    for n, wg in ((1 << 22, 256), (1 << 24, 256), (1 << 24, 1024), (1 << 24, 64), (1 << 23, 32)):
        out.append(('empty', mov(2, 7), n, wg, 64, 0))
    # throughput-bound ALU
    for n, wg, regs, trips in ((1216, 32, 64, 20000), (4864, 128, 64, 6000), (38912, 256, 64, 1000), (155648, 256, 64, 2000),
                               (155648, 1024, 64, 2000), (155648, 256, 128, 2000), (155648, 256, 192, 2000),
                               (1 << 20, 256, 64, 400)):
        out.append(('fmadd_imm8', alu('fmadd32_imm', 8, 64, trips), n, wg, regs, 0))
    # latency-bound ALU (one dependent chain)
    for n, wg, trips in ((1216, 32, 20000), (9728, 256, 6000), (38912, 256, 3000), (155648, 256, 1000)):
        out.append(('iadd_dep', alu('iadd_imm', 1, 64, trips), n, wg, 64, 0))
    # shared-unit bound
    for n, wg, trips in ((4864, 128, 3000), (38912, 256, 600), (155648, 256, 200)):
        out.append(('rsqrt8', alu('rsqrt', 8, 64, trips), n, wg, 64, 0))
    # register-operand bound
    for n, wg, trips in ((38912, 256, 1500), (155648, 256, 400)):
        out.append(('fmadd_reg8', alu('fmadd32', 8, 64, trips), n, wg, 64, 0))
    # divergent if/else: lanes < k take 64 fadds, the others 32 (both sides run when 0 < k < 32)
    for k in (0, 8, 16, 32):
        out.append((f'diverge{k}', diverge(k, 64, 32, 600), 155648, 256, 64, 0))
    # grid-stride streaming reads: table of n*trips*steps words (64 MiB, 160 MiB, 240 MiB)
    for n, trips, steps in ((38912, 54, 8), (155648, 34, 8), (155648, 50, 8)):
        out.append(('stream', stream(trips, steps, n), n, 256, 64, n*trips*steps))
    return out


def predict(path, n, wg, regs, table_words):
    argv = [str(DISPATCH), str(path), '--grid', str(n), '--wg', str(wg), '--regs', str(regs), '--clock', '1398',
            '--uniform', 'u4=0', '--uniform', f'u5={n}', '--uniform', 'u7=1', '--uniform', 'u9=2']
    r = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    if r.returncode: raise RuntimeError(r.stderr)
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', default='')
    ap.add_argument('--repeats', type=int, default=12)
    ap.add_argument('--name', default='dispatch.json')
    ap.add_argument('--no-predict', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('/scratch/apple-m2-playground/sim/dispatch'))
    a = ap.parse_args()
    diverge_gate()
    gpu_safety.require_budget(1)
    R = B.Runner(a.output)
    rows = []
    want = set(a.cases.split(',')) if a.cases else None
    for k, (name, body, n, wg, regs, words) in enumerate(cases()):
        if want and name not in want: continue
        R.run(alu('fmadd32_imm', 8, 64, 4000), 155648, 3, 256, gprs=64)   # warm-up: settle the top clock
        code = drm.build_main(body)
        path = a.output/f'case{k}.bin'
        path.write_bytes(code)
        if words:
            fw, _ = M.run(R, body, n, a.repeats, wg, words+B.TS_BASE+7*n+4096)
        else:
            fw, _ = R.run(body, n, a.repeats, wg, gprs=regs)
        row = {'case': name, 'grid': n, 'wg': wg, 'regs': regs, 'table_words': words, 'fw_ns': fw,
               'min_us': min(fw)/1e3, 'median_us': statistics.median(fw)/1e3}
        if not a.no_predict:
            p = predict(path, n, wg, regs, words)
            row.update({'pred_us': p['time_us'], 'pred': p, 'err': (p['time_us']-row['min_us'])/row['min_us']})
        rows.append(row)
        print(json.dumps({x: (round(v, 2) if isinstance(v, float) else v) for x, v in row.items() if x not in ('fw_ns', 'pred')}), flush=True)
        (a.output/a.name).write_text(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'rows': rows})+'\n')


if __name__ == '__main__':
    main()
