#!/usr/bin/env python3
"""Device-memory throughput with 1..32 SIMD groups per core (direct DRM; scale_probes.py timing).

The timed phase is a counted loop of `steps` device loads (or stores), `wait 0` after every 8. Each of
8 index registers r5..r12 holds a word index into the data region (after the timestamp region of the
table, base u8_u9):

    idx_j = DATA + lane * S + j * J + warp * W          (lane = r1 & 31, warp = SR53)

so one load instruction touches 32 words S words apart (S = 0: one word, 1: one 128-byte line,
32: 32 lines). With an advance A (loads only) each index advances by A words after every wait and
wraps to table words 0..mask (mask = 2^k - 1), so groups stream through 4*(mask+1) bytes instead of
re-reading the same lines (2 ALU instructions per index per 8 memory instructions). The result is core cycles per memory instruction of every group, reference
units, as in scale_probes.

Loads are `device_load 0, i32, x, rD, u8_u9, rI, unsigned` (pointer-chase form); stores the control's
`device_store` form (agx_loop.store); both already ran on hardware.
"""
import argparse, json, struct, subprocess, time
from pathlib import Path
import agx_loop, gpu_safety, run_bench as B, run_drm_compute as drm
import scale_probes as S

e, mov = agx_loop.encode, B.mov
CORES = 38
IDX = list(range(5, 13))
DST = list(range(13, 21))


def mem_loop(steps, kind, s, j, w, data, advance=0, mask=0, glob=False, private=False, pad=0, pad_kind='indep', inter=0):
    """Counted loop (trips in r29) of `steps` memory instructions, wait after every 8.
    private: each workgroup streams through its own window: base r24_r25 = table + 4*(data + SR0*(mask+1)),
    index = (local linear index SR51 * s + j*J + advance...) & mask (validated register-pair base form)."""
    head = mov(3, 0)+e(f'iadd r4, r{S.R_OPTRIPS}, 0')
    if private:
        head += e('iadd r24, u8, 0')+e('iadd r25, u9, 0')+B.get_sr(0, 26)+mov(27, mask+1)+mov(28, data)
        head += e('imadd r26, r26, r27, r28')+e('iadd r24_r25, r24_r25, r26, lsl 2')+B.get_sr(51, 23)
        head += mov(21, s)
        for k, r in enumerate(IDX):
            head += mov(27, k*j)+e(f'imadd r{r}, r23, r21, r27')+mov(27, mask)+e(f'and r{r}, r{r}, r27')
        head += mov(21, advance)+mov(22, mask)
        for r in DST:
            head += mov(r, 0x5a5a5a5a)
        head += e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
        body = b''
        for k in range(steps):
            body += e(f'device_load 0, i32, x, r{DST[k % 8]}, r24_r25, r{IDX[k % 8]}, unsigned')
            body += b''.join(e(f'iadd r{26 + m % 2}, r{26 + m % 2}, 1') for m in range(inter))   # r26/r27: head temporaries only
            if k % 8 == 7 or k == steps-1:
                body += e('wait 0')
                for r2 in IDX:
                    body += e(f'iadd r{r2}, r{r2}, r21')+e(f'and r{r2}, r{r2}, r22')
        body += e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
        return head+body+e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}')+e('pop_exec r0l, 1')
    head += (e('iadd r23, r1, 0') if glob else e('and r23, r1, 31'))+mov(21, s)+mov(22, w)
    for k, r in enumerate(IDX):
        head += mov(27, data+k*j)+e(f'imadd r27, r{S.R_LOCAL}, r22, r27')+e(f'imadd r{r}, r23, r21, r27')
        if mask:   # streaming window: table words 0..mask (loads only; includes the timestamp region)
            head += mov(27, mask)+e(f'and r{r}, r{r}, r27')
    if advance:
        head += mov(21, advance)+mov(22, mask)
    for r in DST:
        head += mov(r, 0x5a5a5a5a)
    head += e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
    body = b''
    for k in range(steps):
        r = IDX[k % 8]
        if kind == 'load':
            body += e(f'device_load 0, i32, x, r{DST[k % 8]}, u8_u9, r{r}, unsigned')
        elif kind == 'store':
            body += agx_loop.store(DST[k % 8], 8, r)
        elif kind in VEC:
            fmt, mask, regs = VEC[kind]
            d0 = DST[0] if k % 2 == 0 else DST[4]
            names = {'4': f'r{d0}_r{d0+1}_r{d0+2}_r{d0+3}', '2': f'r{d0}_r{d0+1}', 'h': f'r{d0}l', 'b': f'r{d0}l'}[regs]
            body += e(f'device_load 0, {fmt}, {mask}, {names}, u8_u9, r{r}, unsigned')
        elif kind == 'atomic':          # device atomic add without return: [u8_u9 + idx*4] += value
            body += agx_loop.atomic_add(8, r, DST[k % 8])
        elif kind == 'atomic_ret':      # device atomic add returning the old value (Mesa encoding, atomics finding)
            import run_drm_atomic as DA
            body += DA.atomic(0, DST[k % 8], r, 21, True)
        elif kind in ('tgatomic', 'tgatomic_ret'):   # threadgroup atomic add at the byte address in r<idx>l
            import run_drm_tg_atomic as TA
            body += TA.local_atomic(0, DST[k % 8], r, 21, kind == 'tgatomic_ret')
        else:                           # threadgroup load/store: byte address in the 16-bit base register
            from run_drm_tg_layout import tg
            body += tg(kind == 'tgload', 'i32', DST[k % 8], base=r, index=0)
        if k % 8 == 7 or k == steps-1:
            body += e('wait 0')
            body += pad_alu(pad, pad_kind)
            if advance:
                for r2 in IDX:
                    body += e(f'iadd r{r2}, r{r2}, r21')+e(f'and r{r2}, r{r2}, r22')
    body += e('iadd r3, 1, r3.sx')+e('while_icmp r0l, ult, r3, r4, 1')
    return head+body+e(f'jmp_exec_any pc-{len(body)}', f'jmp_exec_any pc-{len(body)}')+e('pop_exec r0l, 1')


# vector / narrow device loads (index counts format elements; vec_probe.py)
VEC = {'load_v4': ('i32', 'xyzw', '4'), 'load_v2': ('i32', 'xy', '2'), 'load_i16': ('i16', 'x', 'h'), 'load_i8': ('i8', 'x', 'b')}


def pad_alu(n, kind):
    """n ALU instructions after each wait: 'dep' one chain on r28, 'indep' round-robin over r21..r28,
    'idx' r5..r12 += 0 (writes the load index registers without changing them)."""
    out = b''
    for i in range(n):
        if kind == 'dep':
            out += e('iadd r28, r28, 1')
        elif kind == 'indep':
            out += e(f'iadd r{21 + i % 8}, r{21 + i % 8}, 1')
        else:
            out += e(f'iadd r{5 + i % 8}, r{5 + i % 8}, 0')
    return out


def run(R, body, count, repeats, local_x, table_words, shared_bytes=0):
    """Runner.run with a larger table (timestamp region + data region) and optional threadgroup memory."""
    (R.out/'c.bin').write_bytes(drm.build_main(body)+bytes(16))
    env = {'PATH': '/usr/bin', 'AGX_LOCAL_X': str(local_x), 'AGX_CPU_CLOCK': '0', 'AGX_TABLE_OUT': str(R.out/'t.out')}
    if shared_bytes:
        from run_drm_tg_layout import shared_word
        env['AGX_SHARED_WORD'] = str(shared_word(shared_bytes) if shared_bytes < 65536 else 0x4d | 1 << 8 | 0x24 << 10)
    (R.out/'t.bin').write_bytes(bytes(4*table_words))
    argv = [str(R.exe), '/dev/dri/renderD128', str(R.out/'c.bin'), str(count), str(R.out/'rb'), str(repeats), '64', str(R.out/'t.bin')]
    try:
        r = subprocess.run(argv, capture_output=True, timeout=60, env=env)
    except subprocess.TimeoutExpired:
        gpu_safety.record('mem_probes timeout'); raise RuntimeError('host timeout')
    smp = [json.loads(l) for l in r.stdout.decode().splitlines() if '"rep"' in l]
    if not (r.returncode == 0 and len(smp) == repeats and all(x['fw_ts'][0] and x['fw_ts'][1] for x in smp)):
        gpu_safety.record('mem_probes fault'); raise RuntimeError('dispatch fault: '+r.stdout.decode()[-300:])
    n = B.TS_BASE+7*count
    t = struct.unpack(f'<{n}I', (R.out/'t.out').read_bytes()[:4*n])
    return [x['fw_ts'][1]-x['fw_ts'][0] for x in smp], [[t[B.TS_BASE+k*count+i] for k in range(7)] for i in range(0, count, 32)]


def gate():
    """Emulate one SIMD group's load loop (idle-group trip count path) against a table."""
    import array
    for adv, mask in ((0, 0), (40, 4095)):
        table = array.array('I', range(8192))
        body = S.group_setup(1, 0, 0)+mov(S.R_OPTRIPS, 3)+mem_loop(16, 'load', 1, 64, 0, 2048, adv, mask)+e('iadd r2, r13, 0')
        r = agx_loop.emulate(body, 32, lane_ids=True, num_registers=32, table=table)
        # last load into r13 is step 8 of trip 3: index for j = 0
        want = [((2048+lane) + (2*3-1)*adv) & mask if mask else 2048+lane for lane in range(32)]
        if r['r2'] != want or r['counts'].get('device_load') != 48:
            raise RuntimeError(f'mem gate failed adv={adv}: {r["r2"][:3]} want {want[:3]}, loads {r["counts"].get("device_load")}')


def gate_private():
    """Private mode uses get_sr (not emulated): check every instruction decodes to its intended length."""
    b = mem_loop(16, 'load', 1, 32, 0, 4096, 256, 4095, private=True)
    _, g = agx_loop._tools()
    p = 0
    while p < len(b):
        n = g.opcode_to_number(b[p:p+16]); d = g.get_instruction_descriptor(n)
        if d is None: raise RuntimeError(f'undecodable at {p}')
        p += d.decode_size(n)
    if p != len(b): raise RuntimeError('length mismatch')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('what', choices=['run', 'gate'])
    ap.add_argument('--kind', default='load', choices=['load', 'store', 'atomic', 'atomic_ret', 'tgatomic', 'tgatomic_ret', 'tgload', 'tgstore', 'load_v4', 'load_v2', 'load_i16', 'load_i8'])
    ap.add_argument('--shared-bytes', type=int, default=32768, help='threadgroup memory declared for tg kinds')
    ap.add_argument('--groups', default='1,4,8,16,32')
    ap.add_argument('--stride', default='0,1,2,4,8,16,32', help='S: words between lanes')
    ap.add_argument('--j', type=int, default=0, help='J: words between the 8 index registers')
    ap.add_argument('--w', type=int, default=0, help='W: words between SIMD groups')
    ap.add_argument('--advance', type=int, default=0)
    ap.add_argument('--private', action='store_true',
                    help='each workgroup streams its own window of --mask-bytes (per-core footprint = window); '
                         'J defaults to the workgroup lanes and --advance to 8 x workgroup lanes')
    ap.add_argument('--global', dest='glob', action='store_true',
                    help='lane index = global thread index (r1): groups on every core touch distinct words; with it, '
                         'J defaults to the lane count and --advance to 8 x lanes (grid stride)')
    ap.add_argument('--mask-bytes', default='0', help='streaming window sizes in bytes (powers of two)')
    ap.add_argument('--inter', type=int, default=0, help='private mode: ALU instructions after every load')
    ap.add_argument('--pad', default='0', help='ALU instructions after each wait (list)')
    ap.add_argument('--pad-kind', default='indep', choices=['dep', 'indep', 'idx'])
    ap.add_argument('--trips', type=int, default=200)
    ap.add_argument('--ref-trips', type=int, default=500)
    ap.add_argument('--steps', default='32,128')
    ap.add_argument('--name', default='mem.json')
    ap.add_argument('--output', type=Path, default=Path('/scratch/apple-m2-playground/sim/mem'))
    a = ap.parse_args()
    gate(); gate_private()
    if a.what == 'gate': print('gate ok'); return
    gpu_safety.require_budget(1)
    R = B.Runner(a.output)
    s1, s2 = map(int, a.steps.split(','))
    path = a.output/a.name
    out = json.loads(path.read_text()) if path.exists() else {'rows': []}
    out.update({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'steps': [s1, s2], 'units': 'reference'})
    for pad in map(int, a.pad.split(',')):
      for wb in map(int, a.mask_bytes.split(',')):
        mask = wb//4-1 if wb else 0
        for st in map(int, a.stride.split(',')):
            for G in map(int, a.groups.split(',')):
                count = CORES*32*G
                ts_end = B.TS_BASE+7*count
                data = (ts_end+4095)//4096*4096
                span = data+31*st+7*a.j+(G-1)*a.w+1
                words = max(span, mask+1 if mask else 0)+4096
                if words > (1 << 27): raise SystemExit('table too large')
                jj = a.j or (count*st if a.glob else 32*G*st if a.private else 0)
                adv = a.advance or (8*count*st if a.glob and mask else 8*32*G*st if a.private else 0)
                span = data+(count if a.glob else 32)*st+7*jj+(G-1)*a.w+1
                words = max(span, mask+1 if mask else 0)+4096
                if a.private:
                    words = data+CORES*2*(mask+1)+4096     # room for up to 2 workgroups per core
                if words > (1 << 27): raise SystemExit('table too large')
                if a.kind.startswith('tg') and False: pass
                if a.kind.startswith('tg'):   # byte offsets: lane*4*S + j*J + warp*W (J, W in bytes); must fit the declaration
                    top = 31*4*st+7*jj+(G-1)*a.w+4
                    if top > a.shared_bytes: raise SystemExit(f'threadgroup offsets reach {top} > {a.shared_bytes}')
                    ph = lambda s: mem_loop(s, a.kind, 4*st, jj, a.w, 0, 0, 0)
                else:
                    ph = lambda s: mem_loop(s, a.kind, st, jj, a.w, data, adv, mask, a.glob, a.private, pad, a.pad_kind, a.inter)
                body = S.program(None, 0, G, s1, s2, a.trips, a.ref_trips, count, 2000, phase=ph)
                best = None
                for att in range(1, 5):
                    fw, ts = run(R, body, count, 2+att, 32*G, words, a.shared_bytes if a.kind.startswith('tg') else 0)
                    m = S.analyse(ts, G, a.trips, a.ref_trips, s1, s2)
                    if m['stable_cores'] >= 10: best = m; break
                    best = best or m
                row = {'kind': a.kind, 'inter': a.inter, 'pad': pad, 'pad_kind': a.pad_kind, 'stride': st, 'j': jj, 'w': a.w, 'advance': adv, 'global': a.glob, 'private': a.private, 'window_bytes': wb,
                       'groups': G, 'trips': a.trips, 'attempts': att, **{k: v for k, v in best.items() if k != 'slots'}}
                out['rows'].append(row)
                print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k not in ('spread',)}), flush=True)
                path.write_text(json.dumps(out)+'\n')


if __name__ == '__main__':
    main()
