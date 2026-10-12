#!/usr/bin/env python3
"""Realistic SPIR-V compute workloads: compile, run on a Vulkan device, check against numpy, time, and hand the
driver's compiled AGX code to the simulator.

    spirv_task.py build                                    compile spirv/shaders/*.comp -> spirv/build/*.spv (glslangValidator, spirv-val)
    spirv_task.py run  WORKLOAD|all [--size small|large] [--repeat N] [--out DIR] [--no-export]
    spirv_task.py extract BLOB [--out DIR]                 split an exported shader-object binary into code segments (agx_asm)
    spirv_task.py sim  WORKLOAD --code-dir DIR [--layout spirv/root_layout.json] [--hardware RUN_DIR]   functional simulator
    spirv_task.py predict WORKLOAD --code main.bin [--size S] [--regs H] [--uniform I=V ...]   agx_dispatch for the launch
    spirv_task.py list

`run` writes DIR/<workload>-<size>/ with the spec, inputs, read-back outputs and result.json: functional verdict
(exact for integer workloads, error bounds for float ones), GPU time per run (timestamps), throughput, and, when
the device has VK_EXT_shader_object / VK_KHR_pipeline_executable_properties, the driver's compiled binary
(shader.bin) and statistics / IR (props.json). The same command runs on any Vulkan device; for a CPU check use Mesa's
lavapipe: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json.

Workloads (tasks/spirv-*.md): reduce, scan, sgemm, conv2d, histogram, nbody, bitonic, sha256.
"""
import argparse, hashlib, json, os, shutil, statistics, struct, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHADERS, BUILD = ROOT / 'spirv/shaders', ROOT / 'spirv/build'
VK_RUN = BUILD / 'vk_run'


def push(*vals):
    """push constants: ints as u32, floats as f32"""
    return b''.join(struct.pack('<f', v) if isinstance(v, float) else struct.pack('<I', v & 0xffffffff) for v in vals)


def f32(a): return np.ascontiguousarray(a, dtype=np.float32)
def u32(a): return np.ascontiguousarray(a, dtype=np.uint32)


class W:
    """a workload instance: buffers {binding: (init array or byte size, read back?)}, dispatches [(gx, gy, gz, push)],
    check(outputs) -> (ok, detail), metrics for throughput"""
    def __init__(self, shader, buffers, dispatches, check, wg, work=None):
        self.shader, self.buffers, self.dispatches, self.check, self.wg, self.work = shader, buffers, dispatches, check, wg, work or {}


def rel_err(got, ref):
    scale = max(float(np.max(np.abs(ref))), 1e-30)
    return float(np.max(np.abs(got.astype(np.float64) - ref))) / scale


# ---------------------------------------------------------------------------------------------------------------

def w_reduce(size, rng):
    n = (1 << 14) if size == 'small' else (1 << 24)          # elements; 1024 per workgroup
    groups = n // 1024
    fd = f32(rng.standard_normal(n)); ud = u32(rng.integers(0, 1 << 32, n, dtype=np.uint64))
    def check(o):
        fo, uo = o[2].view(np.float32), o[3].view(np.uint32)
        fref = fd.astype(np.float64).reshape(groups, 1024).sum(1)
        uref = (ud.astype(np.uint64).reshape(groups, 1024).sum(1) & 0xffffffff).astype(np.uint32)
        total_u = int(ud.astype(np.uint64).sum() & 0xffffffff)
        ok_u = np.array_equal(uo[1:1 + groups], uref) and int(uo[0]) == total_u
        ef = float(np.max(np.abs(fo[1:1 + groups] - fref) / (np.abs(fd.reshape(groups, 1024)).astype(np.float64).sum(1) + 1e-30)))
        et = abs(float(fo[0]) - float(fd.astype(np.float64).sum())) / float(np.abs(fd).astype(np.float64).sum())
        return ok_u and ef < 1e-6 and et < 1e-6, {'uint_exact': ok_u, 'float_partial_err': ef, 'float_total_err': et}
    return W('reduce', {0: (fd, False), 1: (ud, False), 2: (4 * (groups + 1), True), 3: (4 * (groups + 1), True)},
             [(groups, 1, 1, push(groups, 0)), (1, 1, 1, push(groups, 1))], check, 256, {'bytes': 8 * n})


def w_scan(size, rng):
    n = (1 << 14) if size == 'small' else (1 << 20)          # nblocks = n / 1024 <= 1024
    nb = n // 1024
    d = u32(rng.integers(0, 1 << 20, n))
    def check(o):
        got = o[0].view(np.uint32)
        ref = (np.concatenate([np.zeros(1, np.uint64), np.cumsum(d.astype(np.uint64))[:-1]]) & np.uint64(0xffffffff)).astype(np.uint32)
        bad = np.flatnonzero(got[:n] != ref)
        return bad.size == 0, {'mismatches': int(bad.size), 'first': int(bad[0]) if bad.size else None}
    return W('scan', {0: (d, True), 1: (4 * 1024, False)},
             [(nb, 1, 1, push(0, nb)), (1, 1, 1, push(1, nb)), (n // 256, 1, 1, push(2, nb))], check, 256, {'bytes': 4 * 4 * n})


def w_sgemm(size, rng):
    M = N = K = 128 if size == 'small' else 1024
    a = f32(rng.uniform(-1, 1, (M, K))); b = f32(rng.uniform(-1, 1, (K, N))); c = f32(rng.uniform(-1, 1, (M, N)))
    alpha, beta = 1.5, -0.5
    def check(o):
        got = o[2].view(np.float32).reshape(M, N)
        ref = alpha * (a.astype(np.float64) @ b.astype(np.float64)) + beta * c.astype(np.float64)
        bound = (abs(alpha) * (np.abs(a).astype(np.float64) @ np.abs(b).astype(np.float64)) + abs(beta) * np.abs(c)) * (K + 2) * 2.0 ** -24
        excess = float(np.max(np.abs(got - ref) / bound))
        return excess <= 1.0, {'max_err_over_bound': excess, 'rel_err': rel_err(got, ref)}
    return W('sgemm', {0: (a, False), 1: (b, False), 2: (c, True)}, [(N // 64, M // 64, 1, push(M, N, K, alpha, beta))],
             check, 256, {'flops': 2 * M * N * K, 'bytes': 4 * (M * K + K * N + 2 * M * N)})


def w_conv2d(size, rng):
    Wd, H = (64, 48) if size == 'small' else (2048, 2048)
    img = f32(rng.uniform(0, 1, (H, Wd)))
    g = np.exp(-0.5 * (np.arange(-2, 3) / 1.2) ** 2); wgt = f32(np.outer(g, g) / np.outer(g, g).sum())
    def check(o):
        got = o[2].view(np.float32)[:Wd * H].reshape(H, Wd)
        pad = np.pad(img.astype(np.float64), 2, mode='edge')
        ref = sum(wgt[dy, dx] * pad[dy:dy + H, dx:dx + Wd] for dy in range(5) for dx in range(5))
        e = rel_err(got, ref)
        return e < 1e-6, {'rel_err': e}
    return W('conv2d', {0: (img, False), 1: (wgt, False), 2: (4 * Wd * H, True)},
             [((Wd + 15) // 16, (H + 15) // 16, 1, push(Wd, H))], check, 256, {'flops': 50 * Wd * H, 'bytes': 8 * Wd * H})


def w_histogram(size, rng):
    nwords = (1 << 12) if size == 'small' else (1 << 22)
    per = 16
    # skewed byte distribution (contention on a few bins, like real data)
    bytes_ = np.minimum(rng.geometric(0.02, 4 * nwords) - 1, 255).astype(np.uint8)
    words = bytes_.view(np.uint32)
    groups = (nwords + 256 * per - 1) // (256 * per)
    def check(o):
        got = o[1].view(np.uint32)[:256]
        ref = np.bincount(bytes_, minlength=256).astype(np.uint32)
        return np.array_equal(got, ref), {'mismatched_bins': int(np.sum(got != ref))}
    return W('histogram', {0: (words, False), 1: (4 * 256, True)}, [(groups, 1, 1, push(nwords, per))], check, 256, {'bytes': 4 * nwords})


def w_nbody(size, rng):
    n = 512 if size == 'small' else 16384
    pos = np.zeros((n, 4), np.float32); pos[:, :3] = rng.standard_normal((n, 3)); pos[:, 3] = rng.uniform(0.5, 1.5, n) / n
    vel = np.zeros((n, 4), np.float32); vel[:, :3] = 0.1 * rng.standard_normal((n, 3))
    dt, eps2 = 0.01, 0.01
    def check(o):
        gp, gv = o[2].view(np.float32).reshape(n, 4), o[3].view(np.float32).reshape(n, 4)
        p64 = pos.astype(np.float64)
        acc = np.zeros((n, 3))
        for s in range(0, n, 256):
            d = p64[None, :, :3] - p64[s:s + 256, None, :3]
            r2 = (d * d).sum(-1) + eps2
            acc[s:s + 256] = (d * (p64[None, :, 3] * r2 ** -1.5)[..., None]).sum(1)
        v = vel[:, :3] + acc * dt
        ev = rel_err(gv[:, :3] - vel[:, :3], v - vel[:, :3])     # error in the velocity update (the computed part)
        ep = rel_err(gp[:, :3], p64[:, :3] + v * dt)
        return ev < 1e-4 and ep < 1e-6 and np.array_equal(gp[:, 3], pos[:, 3]), {'dv_rel_err': ev, 'pos_rel_err': ep}
    return W('nbody', {0: (pos, False), 1: (vel, False), 2: (16 * n, True), 3: (16 * n, True)},
             [(n // 256, 1, 1, push(n, dt, eps2))], check, 256, {'flops': 20 * n * n, 'interactions': n * n})


def w_bitonic(size, rng):
    n = (1 << 12) if size == 'small' else (1 << 20)
    # unique keys: an odd multiplier is a bijection mod 2^32, then shuffled
    keys = u32((rng.permutation(n).astype(np.uint64) * np.uint64(2654435761) + np.uint64(rng.integers(1 << 32))) & np.uint64(0xffffffff))
    vals = u32(np.arange(n))
    ds = [(n // 1024, 1, 1, push(0, 0, 0))]
    k = 2048
    while k <= n:
        j = k // 2
        while j >= 1024:
            ds.append((n // 2 // 512, 1, 1, push(1, k, j))); j //= 2
        ds.append((n // 1024, 1, 1, push(2, k, 0))); k *= 2
    def check(o):
        gk, gv = o[0].view(np.uint32)[:n], o[1].view(np.uint32)[:n]
        ok = np.array_equal(gk, np.sort(keys)) and np.array_equal(keys[gv], gk)
        return ok, {'sorted': bool(np.all(gk[1:] >= gk[:-1])), 'payload_consistent': bool(np.array_equal(keys[gv], gk))}
    return W('bitonic', {0: (keys, True), 1: (vals, True)}, ds, check, 512, {'keys': n, 'dispatches': len(ds)})


def w_sha256(size, rng):
    n = 256 if size == 'small' else 1 << 18
    msgs = rng.integers(0, 256, (n, 55), dtype=np.uint8)
    blocks = np.zeros((n, 64), np.uint8); blocks[:, :55] = msgs; blocks[:, 55] = 0x80; blocks[:, 62:] = [0x01, 0xb8]   # 440 bits
    words = blocks.view('>u4').astype(np.uint32)
    def check(o):
        got = o[1].view(np.uint32).reshape(n, 8)
        ref = np.array([np.frombuffer(hashlib.sha256(bytes(m)).digest(), '>u4') for m in msgs], dtype=np.uint32)
        bad = int(np.sum(np.any(got != ref, axis=1)))
        return bad == 0, {'bad_digests': bad}
    return W('sha256', {0: (words, False), 1: (32 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'hashes': n})


WORKLOADS = {'reduce': w_reduce, 'scan': w_scan, 'sgemm': w_sgemm, 'conv2d': w_conv2d, 'histogram': w_histogram,
             'nbody': w_nbody, 'bitonic': w_bitonic, 'sha256': w_sha256}


# ---------------------------------------------------------------------------------------------------------------

def cmd_build(a=None):
    BUILD.mkdir(parents=True, exist_ok=True)
    glslang = shutil.which('glslangValidator')
    if not glslang:
        sys.exit('glslangValidator not found (glslang-tools / glslang package)')
    for src in sorted(SHADERS.glob('*.comp')):
        out = BUILD / (src.stem + '.spv')
        subprocess.run([glslang, '-V', '--target-env', 'vulkan1.1', '-o', str(out), str(src)], check=True, stdout=subprocess.DEVNULL)
        if shutil.which('spirv-val'):
            subprocess.run(['spirv-val', '--target-env', 'vulkan1.1', str(out)], check=True)
    inc = os.environ.get('VULKAN_INCLUDE')
    cmd = ['cc', '-std=c11', '-O2', '-Wall', str(ROOT / 'tests/vk_run.c'), '-ldl', '-o', str(VK_RUN)] + (['-I', inc] if inc else [])
    subprocess.run(cmd, check=True)
    print(json.dumps({'built': sorted(p.name for p in BUILD.glob('*.spv')), 'runner': str(VK_RUN)}))


def run_one(name, size, repeat, out_root, export=True, seed=1):
    rng = np.random.default_rng([seed, list(WORKLOADS).index(name)])
    w = WORKLOADS[name](size, rng)
    out = out_root / f'{name}-{size}'
    out.mkdir(parents=True, exist_ok=True)
    spec = [f'shader {BUILD / (w.shader + ".spv")}']
    for binding, (init, readback) in sorted(w.buffers.items()):
        if isinstance(init, int):
            nbytes, initp = init, '-'
        else:
            initp = out / f'in{binding}.bin'; initp.write_bytes(init.tobytes()); nbytes = init.nbytes
        spec.append(f'buffer {binding} {nbytes} {initp} {out / f"out{binding}.bin" if readback else "-"}')
    for gx, gy, gz, pc in w.dispatches:
        spec.append(f'dispatch {gx} {gy} {gz} {pc.hex() or "-"}')
    spec.append(f'repeat {repeat}')
    if export:
        spec += [f'export {out / "shader.bin"}', f'props {out / "props.json"}']
    (out / 'spec.txt').write_text('\n'.join(spec) + '\n')
    r = subprocess.run([str(VK_RUN), str(out / 'spec.txt')], capture_output=True, text=True, timeout=600)
    lines = [json.loads(l) for l in r.stdout.splitlines() if l.startswith('{')]
    (out / 'vk_run.jsonl').write_text(r.stdout)
    if r.returncode:
        raise RuntimeError(f'{name}: vk_run failed: {r.stdout[-400:]} {r.stderr[-400:]}')
    outs = {b: np.fromfile(out / f'out{b}.bin', dtype=np.uint8) for b, (_, rb) in w.buffers.items() if rb}
    ok, detail = w.check(outs)
    reps = [l for l in lines if 'rep' in l]
    dev = next(l for l in lines if 'device' in l)
    res = {'workload': name, 'size': size, 'device': dev, 'functional_ok': bool(ok), 'functional': detail,
           'dispatches': [{'groups': [gx, gy, gz], 'push': pc.hex()} for gx, gy, gz, pc in w.dispatches], 'wg_threads': w.wg,
           'work': w.work, 'repeat': repeat, 'compile_ns': next(l for l in lines if 'functional' in l)['compile_ns']}
    if reps and 'total_ns' in reps[0]:
        tot = [l['total_ns'] for l in reps]
        med = statistics.median(tot)
        res['gpu_ns'] = {'median': med, 'min': min(tot), 'max': max(tot)}
        res['dispatch_ns_median'] = [statistics.median(l['dispatch_ns'][k] for l in reps) for k in range(len(w.dispatches))]
        if 'flops' in w.work: res['gflops'] = w.work['flops'] / med
        if 'bytes' in w.work: res['gbytes_per_s'] = w.work['bytes'] / med
    res['host_ns_median'] = statistics.median(l['host_ns'] for l in reps) if reps else None
    (out / 'result.json').write_text(json.dumps(res, indent=1) + '\n')
    return res


def cmd_run(a):
    if not VK_RUN.exists() or not list(BUILD.glob('*.spv')):
        cmd_build()
    names = list(WORKLOADS) if a.workload == 'all' else a.workload.split(',')
    bad = 0
    for name in names:
        try:
            r = run_one(name, a.size, a.repeat, a.out, not a.no_export)
        except Exception as ex:
            bad += 1; print(json.dumps({'workload': name, 'error': str(ex)[:500]})); continue
        bad += not r['functional_ok']
        print(json.dumps({k: r.get(k) for k in ('workload', 'size', 'functional_ok', 'functional', 'gpu_ns', 'gflops', 'gbytes_per_s')}), flush=True)
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------------------------------
# The driver's binary (Mesa honeykrisp, VK_EXT_shader_object): code segments for the simulator

def asm():
    exe = ROOT / 'rust/target/release/agx_asm'
    if not exe.exists():
        sys.exit('agx_asm not built: make rust')
    return exe


def decode_program(p, code, off):
    """decode from off to the first `stop`: [(offset, bytes, name, text)] or None if anything is undecodable"""
    out, pos = [], off
    while pos < len(code):
        p.stdin.write('?!' + code[pos:pos + 16].hex() + '\n'); p.stdin.flush()
        r = p.stdout.readline().rstrip('\n')
        if r.startswith('ERROR'):
            return None
        nm, size, dis, _ = r.split('\t')
        size = int(size)
        if pos + size > len(code) or nm.startswith('TODO') or nm in ('jmp_incomplete', 'trap'):
            return None
        out.append((pos, code[pos:pos + size], nm, dis))
        pos += size
        if nm == 'stop':
            return out
    return None


def cmd_extract(a):
    """MesaVkShaderBin wrapper (scripts/patch_agx_shader.py: Mesa 25.3.6 keeps the (code, data) length pair at
    0x8b4 and the code region at 0x8bc). The code region holds a constant pool and the programs at aligned offsets
    (the stored control: pool at 0, main at 0x80, preamble at 0x100). Every aligned offset that decodes cleanly to a
    `stop` is a program; one with uniform_store is labelled preamble (it loads the root table into uniforms)."""
    b = Path(a.blob).read_bytes()
    if b[:16] != b'MesaVkShaderBin\0':
        sys.exit('not a MesaVkShaderBin blob')
    cands = []
    for start in range(72, len(b) - 8, 4):
        cs, ds = struct.unpack_from('<II', b, start - 8)
        if 16 <= cs <= len(b) - start and cs % 2 == 0 and start + cs + ds <= len(b):
            cands.append((start != 0x8bc, start, cs))
    out = Path(a.out or Path(a.blob).with_suffix(''))
    p = subprocess.Popen([str(asm())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for _, start, cs in sorted(cands):
        code = b[start:start + cs]
        progs = []
        for align in (0x80, 0x10):
            off = 0
            while off < cs:
                if any(code[off:off + 4]) and not any(s <= off < e for s, e, _ in progs):
                    prog = decode_program(p, code, off)
                    if prog:
                        progs.append((off, prog[-1][0] + len(prog[-1][1]), prog))
                off += align
            if progs:
                break
        if not progs:
            continue
        out.mkdir(parents=True, exist_ok=True)
        segs = []
        for k, (s0, e0, prog) in enumerate(sorted(progs)):
            label = 'preamble' if any(nm == 'uniform_store' for _, _, nm, _ in prog) else 'main'
            name = f'{label}.bin' if label not in [x['label'] for x in segs] else f'{label}{k}.bin'
            (out / name).write_bytes(code[s0:e0])
            (out / name.replace('.bin', '.txt')).write_text(''.join(f'{o:#06x}  {bb.hex():<24} {dis}\n' for o, bb, _, dis in prog))
            segs.append({'label': label, 'file': name, 'offset': s0, 'bytes': e0 - s0, 'instructions': len(prog)})
        (out / 'code_region.bin').write_bytes(code)
        info = {'blob': a.blob, 'code_start': start, 'code_size': cs, 'layout_as_mesa_25_3_6': start == 0x8bc, 'programs': segs}
        (out / 'segments.json').write_text(json.dumps(info, indent=1) + '\n')
        print(json.dumps(info))
        return 0
    sys.exit('no decodable program found; inspect the blob by hand (see tasks/spirv-workloads.md)')


def root_value(kind, ctx):
    """value of one root-table / descriptor field (see spirv/root_layout.example.json)"""
    k, _, arg = kind.partition(':')
    if k == 'const': return int(arg, 0)
    if k in ('groups', 'grid', 'wg', 'base'):
        axis = 'xyz'.index(arg)
        return {'groups': ctx['groups'], 'grid': [g * w for g, w in zip(ctx['groups'], ctx['wg'])], 'wg': ctx['wg'], 'base': [0, 0, 0]}[k][axis]
    if k == 'set': return ctx['set_va']
    if k == 'buffer': return ctx['va'][int(arg)]
    if k == 'buffer_size': return ctx['size'][int(arg)]
    raise ValueError(f'unknown root field kind {kind!r}')


def put_field(mem, word, value, nbytes=4):
    mem[4 * word:4 * word + nbytes] = int(value & ((1 << 8 * nbytes) - 1)).to_bytes(nbytes, 'little')


def cmd_sim(a):
    """Functional simulation of a workload with the driver's compiled code (extract) on agx_run, buffer state carried
    across dispatches; outputs checked with the workload's reference and compared word for word with a hardware run."""
    layout = json.loads(Path(a.layout).read_text())
    seg = Path(a.code_dir)
    main_bin, pre_bin = seg / 'main.bin', seg / 'preamble.bin'
    rng = np.random.default_rng([1, list(WORKLOADS).index(a.workload)])
    w = WORKLOADS[a.workload](a.size, rng)
    work = Path(a.out or seg / f'sim-{a.workload}-{a.size}')
    work.mkdir(parents=True, exist_ok=True)
    va, size, files = {}, {}, {}
    nxt = 0x10000000
    for binding, (init, _) in sorted(w.buffers.items()):
        data = bytes(init) if isinstance(init, int) else init.tobytes()
        va[binding], size[binding] = nxt, len(data)
        files[binding] = work / f'buf{binding}.bin'; files[binding].write_bytes(data)
        nxt += (len(data) + 0xffff) // 0x10000 * 0x10000 + 0x10000
    set_va, root_va = nxt, nxt + 0x10000
    d = layout['descriptor']
    setmem = bytearray(d['stride'] * (max(w.buffers) + 1))
    for binding in w.buffers:
        for fld in d['fields']:
            put_field(setmem, binding * d['stride'] // 4 + fld['word'], root_value(fld['value'].replace(':B', f':{binding}'), {'va': va, 'size': size, 'set_va': set_va}), fld.get('bytes', 4))
    (work / 'set.bin').write_bytes(setmem)
    exe = ROOT / 'rust/target/release/agx_run'
    wg = [w.wg, 1, 1] if a.workload not in ('sgemm', 'conv2d') else [16, 16, 1]
    total = 0
    for k, (gx, gy, gz, pc) in enumerate(w.dispatches):
        ctx = {'groups': [gx, gy, gz], 'wg': wg, 'va': va, 'size': size, 'set_va': set_va}
        root = bytearray(layout['root_bytes'])
        for fld in layout['root']:
            if fld['value'] == 'push':
                root[4 * fld['word']:4 * fld['word'] + len(pc)] = pc
            else:
                put_field(root, fld['word'], root_value(fld['value'], ctx), fld.get('bytes', 4))
        (work / 'root.bin').write_bytes(root)
        cmd = [str(exe), str(main_bin), '--grid', ','.join(str(g * x) for g, x in zip(ctx['groups'], wg)), '--wg', ','.join(map(str, wg)),
               '--shared', str(a.shared), '--map', f'{root_va:#x}=' + str(work / 'root.bin'), '--map', f'{set_va:#x}=' + str(work / 'set.bin')]
        if pre_bin.exists():
            cmd += ['--preamble', str(pre_bin)]
        for u, kind in layout['uniforms'].items():
            v = root_va if kind == 'root' else set_va if kind == 'set' else int(kind, 0)
            cmd += ['--uniform', f'{int(u)}={v & 0xffffffff:#x}', '--uniform', f'{int(u) + 1}={v >> 32:#x}']
        for b in w.buffers:
            cmd += ['--map', f'{va[b]:#x}={files[b]}', '--dump', f'{va[b]:#x}={size[b]}:{files[b]}']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'dispatch {k}: {r.stdout.strip()} {r.stderr.strip()}')
        total += json.loads(r.stdout)['instructions']
    outs = {b: np.fromfile(files[b], dtype=np.uint8) for b, (_, rb) in w.buffers.items() if rb}
    ok, detail = w.check(outs)
    res = {'workload': a.workload, 'size': a.size, 'simulated_ok': bool(ok), 'functional': detail, 'simulated_instructions': total}
    if a.hardware:
        hw = Path(a.hardware)
        diff = {}
        for b in outs:
            h = np.fromfile(hw / f'out{b}.bin', dtype=np.uint32); s_ = outs[b].view(np.uint32)
            n = min(len(h), len(s_))
            diff[b] = int(np.sum(h[:n] != s_[:n]))
        res['words_differing_from_hardware'] = diff
    (work / 'sim.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res))
    return 0 if ok else 1


def cmd_predict(a):
    rng = np.random.default_rng([1, list(WORKLOADS).index(a.workload)])
    w = WORKLOADS[a.workload](a.size, rng)
    exe = ROOT / 'rust/target/release/agx_dispatch'
    total = 0.0
    rows = []
    for gx, gy, gz, pc in w.dispatches:
        wg = [w.wg, 1, 1] if a.workload not in ('sgemm', 'conv2d') else [16, 16, 1]
        grid = [gx * wg[0], gy * wg[1], gz * wg[2]]
        cmd = [str(exe), a.code, '--grid', ','.join(map(str, grid)), '--wg', ','.join(map(str, wg))]
        if a.regs: cmd += ['--regs', str(a.regs)]
        if a.tg_bytes: cmd += ['--tg-bytes', str(a.tg_bytes)]
        for u in a.uniform or []: cmd += ['--uniform', u]
        if a.no_mem: cmd.append('--no-mem')
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'agx_dispatch failed: {r.stderr.strip()}')
        j = json.loads(r.stdout); rows.append(j)
        total += j.get('us', 0.0)
    print(json.dumps({'workload': a.workload, 'size': a.size, 'predicted_us': total, 'dispatches': rows}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('build'); sub.add_parser('list')
    p = sub.add_parser('run'); p.add_argument('workload'); p.add_argument('--size', choices=['small', 'large'], default='large')
    p.add_argument('--repeat', type=int, default=10); p.add_argument('--out', type=Path, default=ROOT / 'spirv/out')
    p.add_argument('--no-export', action='store_true')
    p = sub.add_parser('extract'); p.add_argument('blob'); p.add_argument('--out')
    p = sub.add_parser('sim'); p.add_argument('workload'); p.add_argument('--code-dir', required=True, help='extract output (main.bin, preamble.bin)')
    p.add_argument('--layout', default=str(ROOT / 'spirv/root_layout.json')); p.add_argument('--size', choices=['small', 'large'], default='small')
    p.add_argument('--shared', type=int, default=32768); p.add_argument('--hardware', help='run output directory to compare with word for word')
    p.add_argument('--out')
    p = sub.add_parser('predict'); p.add_argument('workload'); p.add_argument('--code', required=True)
    p.add_argument('--size', choices=['small', 'large'], default='large'); p.add_argument('--regs', type=int)
    p.add_argument('--tg-bytes', type=int); p.add_argument('--uniform', action='append'); p.add_argument('--no-mem', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'list':
        for k, f in WORKLOADS.items():
            print(k, '-', (SHADERS / f'{k}.comp').read_text().splitlines()[1].lstrip('/ '))
        return
    sys.exit({'build': cmd_build, 'run': cmd_run, 'extract': cmd_extract, 'sim': cmd_sim, 'predict': cmd_predict}[a.cmd](a) or 0)


if __name__ == '__main__':
    main()
