"""Threadgroup-load throughput probes: hardware (agx_bench_kernel, firmware times) vs simulator (agx_dispatch).

tgprobe.py [--only NAME]   prints one line per probe: hw us, sim us, error.

Each probe is a PTX kernel: every thread fills a 16 KiB shared array, then runs ITERS iterations of L loads of a
pattern (bcast: every lane the same address; lane: lane-distinct consecutive words) of width W (1 = .f32,
2 = .v2.f32), each component added into an accumulator plus F FMAs, and stores the sum.
The grid puts G SIMD groups on each of the 38 cores (block 32*G/WG... one workgroup of G groups per core).
"""
import sys, os, json, subprocess, tempfile, argparse, itertools

ZOC = '/scratch/geekbench-extract/cargo-target/release/zoc'
BENCH = '/scratch/geekbench-extract/cargo-target/release/agx_bench_kernel'
TOOLS = '/scratch/geekbench-extract/tools'


def ptx(pattern, width, loads, fmas, iters):
    v = '.v2' if width == 2 else ''
    regs = '{%a<A>, %b<A>}' if width == 2 else '%a<A>'
    body = []
    for l in range(loads):
        dst = f'{{%x{l}, %y{l}}}' if width == 2 else f'%x{l}'
        # address: bcast = (i * L + l) * 8 & 0x3ff8 (same in every lane); lane = tid * 4 * W + same offset
        body.append(f'add.u32 %q, %i, {l};')
        body.append('shl.b32 %q, %q, 3;')
        body.append('and.b32 %q, %q, 8184;')
        if pattern == 'lane':
            body.append('add.u32 %q, %q, %lanebytes;')
            body.append('and.b32 %q, %q, 16376;')
        body.append(f'ld.shared{v}.f32 {dst}, [sm+%q_];'.replace('%q_', '0') if False else f'add.u32 %p_, %smb, %q;')
        body.append(f'ld.shared{v}.f32 {dst}, [%p_];')
        comps = [f'%x{l}'] + ([f'%y{l}'] if width == 2 else [])
        for c, src in enumerate(comps):
            body.append(f'add.f32 %acc{(l * 2 + c) % 8}, %acc{(l * 2 + c) % 8}, {src};')
        for f in range(fmas):
            src = f'%x{l}' if (width == 1 or f % 2 == 0) else f'%y{l}'
            body.append(f'fma.rn.f32 %acc{(l * fmas + f) % 8}, {src}, %k, %acc{(l * fmas + f) % 8};')
    body = '\n        '.join(body)
    xs = ', '.join(f'%x{l}' for l in range(loads)) + ((', ' + ', '.join(f'%y{l}' for l in range(loads))) if width == 2 else '')
    return f""".version 7.0
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 out, .param .u32 n)
{{
    .shared .align 16 .b8 sm[16384];
    .reg .u32 %tid, %i, %q, %p_, %smb, %lanebytes, %n, %t;
    .reg .f32 {xs}, %k, %acc<8>, %s;
    .reg .u64 %o, %oa;
    .reg .pred %p;
    mov.u32 %tid, %tid.x;
    mov.u32 %smb, sm;
    and.b32 %lanebytes, %tid, 31;
    shl.b32 %lanebytes, %lanebytes, {3 if width == 2 else 2};
    // fill: every thread writes its words of the 16 KiB
    mov.u32 %t, %tid;
FILL:
    shl.b32 %q, %t, 2;
    add.u32 %q, %q, %smb;
    cvt.rn.f32.u32 %s, %t;
    st.shared.f32 [%q], %s;
    add.u32 %t, %t, %ntid.x;
    setp.lt.u32 %p, %t, 4096;
    @%p bra FILL;
    bar.sync 0;
    ld.param.u32 %n, [n];
    mov.f32 %k, 0f3F800000;
    mov.f32 %acc0, 0f00000000; mov.f32 %acc1, 0f00000000; mov.f32 %acc2, 0f00000000; mov.f32 %acc3, 0f00000000;
    mov.f32 %acc4, 0f00000000; mov.f32 %acc5, 0f00000000; mov.f32 %acc6, 0f00000000; mov.f32 %acc7, 0f00000000;
    mov.u32 %i, 0;
LOOP:
        {body}
    add.u32 %i, %i, {loads};
    setp.lt.u32 %p, %i, %n;
    @%p bra LOOP;
    add.f32 %s, %acc0, %acc1; add.f32 %s, %s, %acc2; add.f32 %s, %s, %acc3;
    add.f32 %s, %s, %acc4; add.f32 %s, %s, %acc5; add.f32 %s, %s, %acc6; add.f32 %s, %s, %acc7;
    ld.param.u64 %o, [out];
    cvta.to.global.u64 %o, %o;
    mov.u32 %t, %ctaid.x;
    mad.lo.u32 %t, %t, %ntid.x, %tid;
    mul.wide.u32 %oa, %t, 4;
    add.u64 %o, %o, %oa;
    st.global.f32 [%o], %s;
    ret;
}}
"""


def run(name, pattern, width, loads, fmas, groups, wgs_per_core, iters=4096):
    d = os.path.join('/scratch/geekbench-extract/probes', name)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, 'p.ptx')
    open(src, 'w').write(ptx(pattern, width, loads, fmas, iters))
    subprocess.run([ZOC, '--arch=agx-g14c', '--output-dir', d, src], check=True, capture_output=True)
    obj = os.path.join(d, 'p.o')
    block, grid = 32 * groups, 38 * wgs_per_core
    prof = os.path.join(d, 'profile.jsonl')
    if os.path.exists(prof): os.remove(prof)
    env = dict(os.environ, AGX_RUNTIME_PROFILE=prof, AGX_RUNTIME_ZOC=ZOC)
    r = subprocess.run([BENCH, obj, 'probe', '--grid', str(grid), '--block', str(block), '--iters', '200', '--warmup', '50',
                        f'buf:{grid * block * 4}', f'u32:{iters}'], capture_output=True, text=True, env=env, timeout=300)
    if 'ms_per_launch' not in r.stdout:
        return f'{name:34s} HW FAILED {r.stdout[-200:]} {r.stderr[-200:]}'
    lines = open(prof).read().splitlines()
    ns = sorted(json.loads(l)['ns'] for l in lines[-200:])
    hw = ns[len(ns) // 2] / 1e3
    # the simulator on the same launch (one profile line)
    kd = os.path.join(d, 'k')
    subprocess.run(['python3', '-I', os.path.join(TOOLS, 'kextract.py'), d, kd], check=True, capture_output=True)
    one = os.path.join(d, 'one.jsonl')
    open(one, 'w').write(lines[-1] + '\n')
    s = subprocess.run(['python3', '-I', os.path.join(TOOLS, 'replay.py'), one, kd, '--limit', '1'], capture_output=True, text=True)
    try:
        sim = float(s.stdout.split(' sim ')[1].split(' us')[0])
    except Exception:
        return f'{name:34s} hw {hw:8.1f} us  SIM FAILED {s.stdout[-200:]}{s.stderr[-200:]}'
    per = lambda t: t * 1e3 * 1.398 / (iters / loads * loads) / (groups * wgs_per_core)  # core cycles per load per group
    return f'{name:34s} hw {hw:8.1f} us  sim {sim:8.1f} us  err {(sim - hw) / hw:+5.0%}   hw {hw * 1398 / (iters * groups * wgs_per_core):6.2f} cyc/load·group'


ap = argparse.ArgumentParser()
ap.add_argument('--only')
a = ap.parse_args()
cases = []
for pattern, width, fmas in [('bcast', 1, 0), ('bcast', 2, 0), ('lane', 1, 0), ('lane', 2, 0), ('bcast', 2, 2), ('lane', 1, 2)]:
    for groups, wgs in [(4, 1), (4, 2), (4, 4), (16, 1), (32, 1)]:
        cases.append((f'{pattern}-w{width}-f{fmas}-g{groups}x{wgs}', pattern, width, 4, fmas, groups, wgs))
for c in cases:
    if a.only and a.only not in c[0]: continue
    print(run(*c), flush=True)
