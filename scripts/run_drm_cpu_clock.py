#!/usr/bin/env python3
"""An in-shader clock: read a CPU-written timer from shared memory (direct DRM).

tests/drm_compute.c with AGX_CPU_CLOCK=0 runs a CPU thread that keeps writing table
word 0 = spin counter and word 1 = low 32 bits of CNTVCT_EL0 (ARM generic timer;
CNTFRQ is printed) for as long as the submits run. The shader reads both words, runs a
counted loop of K trips, reads them again, and stores (counter0, timer0, counter1,
timer1) per lane at table word 64 + 4*gid.

Variants: the load cache policy (Mesa's normal u4=4 or coherent u4=7, agx_pack.c:742),
whether the CPU cleans each write to the point of coherency (dc cvac), and K.
Only decoded, Mesa-emitted instruction forms are used.
"""
import argparse, datetime, json, os, shlex, struct, subprocess, sys
from pathlib import Path

import agx_loop
import patch_agx_sr as patcher
import run_drm_compute as drm
from run_drm_opcode_describe import get_sr, WAIT
from run_drm_sync_probe import finish, decodes

REPO = Path(__file__).resolve().parents[1]
mov, enc = agx_loop.chains.mov_imm, agx_loop.encode
SENT = 0xcdcdcdcd
RES = 32*16   # results after up to 16 clock lines (128 B apart)
LINE = 32


def load(reg, idx_reg, coherent):
    b = enc(f'device_load 0, i32, x, r{reg}, u8_u9, r{idx_reg}, unsigned')
    x = int.from_bytes(b, 'little')
    if coherent: x = (x & ~(7 << 44)) | 7 << 44     # Mesa: u4 = coherent ? 7 : 4
    return x.to_bytes(8, 'little')


def loop(trips):
    """Counted loop (agx_loop shape) of 4 dependent iadd per trip; uses r2..r4."""
    gpu, _ = patcher.load_model(agx_loop.APPLEGPU)
    head = mov(3, 0)+mov(4, trips)+mov(2, 0)+enc('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    body = b''.join(agx_loop.chains.encode_step('iadd8', 2, gpu) for _ in range(4))
    body += enc('iadd r3, 1, r3.sx')+enc('while_icmp r0l, ult, r3, r4, 1')
    return head+body+enc(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}')+enc('pop_exec r0l, 1')


BARRIERS = {'none': b'', 'mem_barrier': bytes.fromhex('f596'), 'device_barrier_1': bytes.fromhex('f59d'),
            'device_barrier_2': bytes.fromhex('f59e'), 'unknown_barrier_1': bytes.fromhex('f533'),
            'mesa_device_scope': bytes.fromhex('f596f533f59af59ef503f599f59d')}   # agx_compile.c:1514-1534 order


def body(trips, coherent, reads=2, second_line=False, between='none'):
    b = get_sr(80, 24)
    out = b''
    regs = []
    for k in range(reads):
        r0, r1 = 5+2*k, 6+2*k
        line = LINE*k if second_line else 0
        out += mov(26, line)+mov(27, line+1)+load(r0, 26, coherent)+load(r1, 27, coherent)+WAIT
        regs += [r0, r1]
        if k < reads-1:
            if trips: out += loop(trips)
            out += BARRIERS[between]
    b += out+mov(25, 4)+mov(28, RES)+enc('imadd r29, r24, r25, r28')
    for r in regs:
        b += agx_loop.store(r, 8, 29)+enc('iadd r29, r29, 1')
    return b+WAIT+finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--trips', default='0,1000,10000,100000')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='/dev/dri/renderD128')
    p.add_argument('--only', help="python expression over coherent, flush, trips, second_line, between")
    a = p.parse_args()
    gate = agx_loop.emulate(loop(5), 32)
    if gate['counts'].get('jmp_exec_any') != 5 or gate['exec'] != [True]*32: sys.exit('emulator gate failed')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = a.output or Path('/scratch/apple-m2-playground/drm-compute/cpu-clock')/stamp
    out.mkdir(parents=True, exist_ok=False)
    exe = out/'drm_compute'
    subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', str(REPO/'tests/drm_compute.c'), '-o', str(exe)], check=True)
    words = RES+4*a.count
    results = []
    configs = [(c, f, t, sl, bw) for c in (False, True) for f in (0, 1) for t in [int(x) for x in a.trips.split(',')]
               for sl, bw in [(False, 'none'), (True, 'none')]+[(False, k) for k in BARRIERS if k != 'none']]
    if a.only: configs = [x for x in configs if eval(a.only, {}, dict(coherent=x[0], flush=x[1], trips=x[2], second_line=x[3], between=x[4]))]
    for coherent, flush, trips, second_line, between in configs:
            if True:
                bb = body(trips, coherent, second_line=second_line, between=between); decodes(bb, set())
                (out/'c.bin').write_bytes(drm.build_main(bb)+bytes(16))
                (out/'t.bin').write_bytes(struct.pack('<I', SENT)*words)
                env = dict(os.environ, AGX_CPU_CLOCK='0', AGX_CPU_CLOCK_FLUSH=str(flush), AGX_CPU_CLOCK_LINES='2', AGX_TABLE_OUT=str(out/'t.out'))
                r = subprocess.run([str(exe), a.device, str(out/'c.bin'), str(a.count), str(out/'rb'), str(a.repeats), '64', str(out/'t.bin')],
                                   capture_output=True, env=env, timeout=60)
                ev = [json.loads(l) for l in r.stdout.decode().splitlines() if l.startswith('{')]
                if r.returncode: sys.exit('dispatch failed: '+r.stdout.decode()[-300:])
                smp = [e for e in ev if 'rep' in e]; cal = next(e['cpu_clock'] for e in ev if 'cpu_clock' in e)
                t = struct.unpack(f'<{words}I', (out/'t.out').read_bytes())
                lanes = [t[RES+4*i:RES+4*i+4] for i in range(a.count)]
                rec = {'coherent': coherent, 'flush': flush, 'trips': trips, 'second_line': second_line, 'between': between, 'fw_ns_last': smp[-1]['fw_ts'][1]-smp[-1]['fw_ts'][0],
                       'fw_ns': [e['fw_ts'][1]-e['fw_ts'][0] for e in smp], 'cpu_clock': cal,
                       'lane0': lanes[0], 'lanes_agree': all(l == lanes[0] for l in lanes)}
                c0, t0, c1, t1 = lanes[0]
                rec['counter_delta'] = (c1-c0) & 0xffffffff
                rec['timer_delta_ticks'] = (t1-t0) & 0xffffffff
                rec['timer_delta_ns'] = rec['timer_delta_ticks']*1e9/cal['cntfrq']
                rec['any_sentinel'] = SENT in lanes[0]
                results.append(rec)
                print(json.dumps({k: rec[k] for k in ('coherent', 'flush', 'trips', 'second_line', 'between', 'counter_delta', 'timer_delta_ticks', 'lanes_agree', 'any_sentinel')}
                                 | {'timer_us': round(rec['timer_delta_ns']/1e3, 2), 'fw_us': round(rec['fw_ns_last']/1e3, 2)}), flush=True)
    (out/'record.json').write_text(json.dumps({'utc': stamp, 'count': a.count, 'repeats': a.repeats, 'results': results})+'\n')
    print(json.dumps({'output': str(out)}))


if __name__ == '__main__':
    main()
