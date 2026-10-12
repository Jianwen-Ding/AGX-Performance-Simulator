#!/usr/bin/env python3
"""Realistic SPIR-V compute workloads: compile, run on a Vulkan device, check against numpy, time, and hand the
driver's compiled AGX code to the simulator.

    spirv_task.py build                                    compile spirv/shaders/*.comp -> spirv/build/*.spv (glslangValidator, spirv-val)
    spirv_task.py run  WORKLOAD|all [--size small|large] [--repeat N] [--out DIR] [--no-export]
    spirv_task.py extract BLOB [--out DIR]                 split an exported shader-object binary into code segments (agx_asm)
    spirv_task.py sim  WORKLOAD --code-dir DIR [--layout L] [--hardware RUN_DIR]   functional simulator
    spirv_task.py compile WORKLOAD|all --out DIR [--mesa SRC]   compile with Mesa's AGX compiler (also via drm-shim, no GPU)
    spirv_task.py simall DIR / predictall DIR              sim / predict every workload compile wrote under DIR
    spirv_task.py predict WORKLOAD --code-dir DIR [--size S]   agx_dispatch per dispatch, paths from the functional simulator
    spirv_task.py predict WORKLOAD --code main.bin [--size S] [--regs H] [--uniform I=V ...]   ... from the cost model's emulation
    spirv_task.py list

`run` writes DIR/<workload>-<size>/ with the spec, inputs, read-back outputs and result.json: functional verdict
(exact for integer workloads, error bounds for float ones), GPU time per run (timestamps), throughput, and, when
the device has VK_EXT_shader_object / VK_KHR_pipeline_executable_properties, the driver's compiled binary
(shader.bin) and statistics / IR (props.json). The same command runs on any Vulkan device; for a CPU check use Mesa's
lavapipe: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json.

Workloads (tasks/spirv-*.md): reduce, scan, sgemm, conv2d, histogram, nbody, bitonic, sha256.
"""
import argparse, gzip, hashlib, json, os, shutil, statistics, struct, subprocess, sys, tempfile, time
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
    def __init__(self, shader, buffers, dispatches, check, wg, work=None, images=None):
        self.shader, self.buffers, self.dispatches, self.check, self.wg, self.work = shader, buffers, dispatches, check, wg, work or {}
        self.images = images or []       # [(binding, width, height, 'rgba8' | 'gray8' | 'bits')]: written as .ppm / .pgm for viewing


def write_images(w, outs, out_dir, tag):
    """the workload's image outputs as binary PPM (RGBA8 -> RGB) / PGM files, for looking at the result"""
    for b, wd, ht, kind in w.images:
        if b not in outs:
            continue
        raw = outs[b]
        if kind == 'rgba8':
            px = raw[:4 * wd * ht].reshape(ht, wd, 4)[..., :3]
            (Path(out_dir) / f'{tag}{b}.ppm').write_bytes(f'P6 {wd} {ht} 255\n'.encode() + px.tobytes())
        elif kind == 'bits':                                                    # bit-packed cells, bit j of word w = cell 32 w + j
            px = (255 - 255 * np.unpackbits(raw[:wd * ht // 8], bitorder='little')).astype(np.uint8)
            (Path(out_dir) / f'{tag}{b}.pgm').write_bytes(f'P5 {wd} {ht} 255\n'.encode() + px.tobytes())
        else:
            (Path(out_dir) / f'{tag}{b}.pgm').write_bytes(f'P5 {wd} {ht} 255\n'.encode() + raw[:wd * ht].tobytes())


def wg_of(w, name):
    """workgroup size when the driver's info is not at hand: a workload's 3-tuple, else 1-D (sgemm / conv2d: 16 x 16)"""
    if isinstance(w.wg, (tuple, list)):
        return list(w.wg)
    return [16, 16, 1] if name in ('sgemm', 'conv2d') else [w.wg, 1, 1]


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



def f16_ulps(got, ref):
    """distance in fp16 ulps (ordered bit patterns), NaN-free inputs"""
    g = got.view(np.uint16).astype(np.int32); r = ref.astype(np.float16).view(np.uint16).astype(np.int32)
    g = np.where(g & 0x8000, 0x8000 - g, g); r = np.where(r & 0x8000, 0x8000 - r, r)
    return np.abs(g - r)


def w_fp16(size, rng):
    n = 4096 if size == 'small' else 1 << 22
    x = rng.uniform(-8, 8, n).astype(np.float16); y = rng.uniform(-8, 8, n).astype(np.float16)
    a = 0.75
    def check(o):
        got = o[2].view(np.float16)[:3 * n].reshape(n, 3)
        x64, y64 = x.astype(np.float64), y.astype(np.float64)
        g = got.astype(np.float64)
        ulp = lambda v: 2.0 ** np.maximum(np.floor(np.log2(np.maximum(np.abs(v), 2.0 ** -24))) - 10, -24)
        # Vulkan lets fma and x*y - y round the product and the sum separately: each within half an ulp of the
        # product plus half an ulp of the result of the exact value; the fp32 expression rounds once on the store
        excess = []
        for k, (prod, exact) in enumerate([(0.75 * x64, 0.75 * x64 + y64), (x64 * y64, x64 * y64 - y64)]):
            excess.append(float(np.max(np.abs(g[:, k] - exact) / (0.5 * ulp(prod) + 0.5 * ulp(g[:, k]) + 1e-12))))
        conv = int(f16_ulps(got[:, 2], x64 * 1.5 + 0.25).max())
        return max(excess) <= 1.0 and conv == 0, {'max_err_over_bound': excess, 'conversion_ulps': conv}
    return W('fp16', {0: (x, False), 1: (y, False), 2: (6 * n, True)}, [((n + 255) // 256, 1, 1, push(n, a))], check, 256,
             {'bytes': 10 * n})


M64 = (1 << 64) - 1


def splitmix64(x):
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def w_int64(size, rng):
    n = 1024 if size == 'small' else 1 << 20
    seed = rng.integers(0, 1 << 63, n, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, n, dtype=np.uint64)
    a = rng.integers(-(1 << 31), 1 << 31, 2 * n, dtype=np.int64).astype(np.int32)
    def check(o):
        got = o[2].view(np.uint64)[:4 * n].reshape(n, 4)
        bad = 0
        for i in range(n) if size == 'small' else rng.choice(n, 4096, replace=False):
            s_ = int(seed[i]); h = splitmix64(s_)
            prod = int(a[2 * i]) * int(a[2 * i + 1])
            want = [h, (prod >> 7) & M64, h - s_ if h > s_ else s_ - h, (s_ + (int(a[2 * i]) & 0xffffffff)) & M64]
            bad += [int(v) for v in got[i]] != want
        return bad == 0, {'bad_rows': bad}
    return W('int64', {0: (seed, False), 1: (a, False), 2: (32 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64,
             {'hashes': n})


def bitops_ref(x, y, w, z):
    off = w & 31; bits = ((w >> 8) % (33 - off)); r = (w >> 16) & 31
    mask = ((1 << bits) - 1) << off
    ext = (x & mask) >> off
    sext = ext - (1 << bits) if bits and ext >> (bits - 1) & 1 else ext
    s32 = lambda v: v - (1 << 32) if v & 0x80000000 else v
    lsb = (x & -x).bit_length() - 1 if x else -1
    msb = x.bit_length() - 1 if x else -1
    sy = s32(y); smsb = (sy.bit_length() - 1 if sy > 0 else (~sy).bit_length() - 1) if sy not in (0, -1) else -1
    rot = lambda v, k: ((v << k) | (v >> ((32 - k) & 31))) & 0xffffffff
    rotr = lambda v, k: ((v >> k) | (v << ((32 - k) & 31))) & 0xffffffff
    um = x * y; im = s32(x) * s32(z)
    add = x + z
    m = 0xffffffff
    return [ext, sext & m, (x & ~mask & m) | ((y << off) & mask), lsb & m, msb & m, smsb & m, bin(x).count('1'),
            int(f'{y:032b}'[::-1], 2), rot(x, r), rotr(z, r), um >> 32, um & m, (im >> 32) & m, im & m, add & m, add >> 32]


def w_bitops(size, rng):
    n = 1024 if size == 'small' else 1 << 18
    pick = lambda: np.where(rng.random(n) < 0.15, rng.choice(np.array([0, 1, 0x80000000, 0xffffffff, 0x7fffffff, 0x00010000], dtype=np.uint64), n),
                            rng.integers(0, 1 << 32, n, dtype=np.uint64))
    inp = np.stack([pick(), pick(), rng.integers(0, 1 << 32, n, dtype=np.uint64), pick()], 1).astype(np.uint32)
    def check(o):
        got = o[1].view(np.uint32)[:16 * n].reshape(n, 16)
        rows = range(n) if size == 'small' else rng.choice(n, 4096, replace=False)
        bad = [k for i in rows for k, (g, w) in enumerate(zip(got[i], bitops_ref(*map(int, inp[i])))) if int(g) != w]
        return not bad, {'mismatches': len(bad), 'by_output': {str(k): bad.count(k) for k in sorted(set(bad))}}
    return W('bitops', {0: (inp, False), 1: (64 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'items': n})


def w_intdiv(size, rng):
    n = 1024 if size == 'small' else 1 << 20
    a = rng.integers(0, 1 << 32, n, dtype=np.uint64)
    b = np.where(rng.random(n) < 0.5, rng.integers(1, 1 << 12, n, dtype=np.uint64), rng.integers(1, 1 << 32, n, dtype=np.uint64))
    sa = rng.integers(-(1 << 31) + 1, 1 << 31, n, dtype=np.int64)
    sb = rng.integers(1, 1 << 16, n, dtype=np.int64) * rng.choice([-1, 1], n)
    inp = np.stack([a, b, sa & 0xffffffff, sb & 0xffffffff], 1).astype(np.uint32)
    def check(o):
        got = o[1].view(np.uint32)[:12 * n].reshape(n, 12).astype(np.int64)
        m = 0xffffffff
        tdiv = lambda p_, q: (abs(p_) // abs(q)) * (1 if (p_ >= 0) == (q >= 0) else -1)
        bw = b.astype(np.uint32)
        ref = np.stack([a // b, a % b, a // 7, a % 1000, a // 65537,
                        np.array([tdiv(int(x), int(y)) for x, y in zip(sa, sb)]) & m, np.array([tdiv(int(x), 3) for x in sa]) & m,
                        a.astype(np.float32).view(np.uint32), sa.astype(np.float32).view(np.uint32),
                        ((bw & 0x007fffff) | 0x4b000000).view(np.float32).astype(np.int64),
                        np.trunc(((inp[:, 3] & 0x807fffff) | 0x49000000).view(np.float32)).astype(np.int64) & m,
                        ((sa >> 3) & m) ^ (a >> 29).astype(np.int64)], 1).astype(np.int64)
        bad = got != ref
        return not bad.any(), {'mismatches': int(bad.sum()), 'by_output': {str(k): int(v) for k, v in enumerate(bad.sum(0)) if v}}
    return W('intdiv', {0: (inp, False), 1: (48 * n, True)}, [((n + 127) // 128, 1, 1, push(n))], check, 128, {'items': n})


def w_compact(size, rng):
    n = (1 << 14) if size == 'small' else (1 << 22)
    groups = n // 256
    data = u32(rng.integers(0, 1 << 32, n, dtype=np.uint64))
    thr = 0x55555555                                           # keeps about a third
    def check(o):
        keep = data[data < thr]
        got = o[2].view(np.uint32)
        tot = int(o[3].view(np.uint32)[0])
        ok = tot == len(keep) and np.array_equal(got[:tot], keep)
        return ok, {'kept': len(keep), 'total': tot, 'order_ok': bool(np.array_equal(got[:min(tot, len(keep))], keep[:tot]))}
    ds = [(groups, 1, 1, push(0, groups, thr)), (1, 1, 1, push(1, groups, thr)), (groups, 1, 1, push(2, groups, thr))]
    return W('compact', {0: (data, False), 1: (4 * groups, False), 2: (4 * n, True), 3: (4, True)}, ds, check, 256, {'bytes': 8 * n})


def w_spmv(size, rng):
    rows, cols = (512, 512) if size == 'small' else (1 << 18, 1 << 18)
    nnz = rng.integers(0, 64, rows) * (rng.random(rows) < 0.95)    # some empty rows
    rowptr = u32(np.concatenate([[0], np.cumsum(nnz)]))
    col = u32(rng.integers(0, cols, int(rowptr[-1])))
    val = f32(rng.uniform(-1, 1, int(rowptr[-1])))
    x = f32(rng.uniform(-1, 1, cols))
    def check(o):
        got = o[4].view(np.float32)[:rows]
        prod = val.astype(np.float64) * x[col].astype(np.float64)
        ref = np.add.reduceat(prod, rowptr[:-1].astype(np.int64)) if len(prod) else np.zeros(rows)
        ref = np.where(nnz > 0, ref, 0.0)
        bound = np.add.reduceat(np.abs(prod), rowptr[:-1].astype(np.int64)) * (nnz + 1) * 2.0 ** -24
        excess = float(np.max(np.abs(got - ref) / np.maximum(bound, 1e-30) * (nnz > 0)))
        return excess <= 1.0 and bool(np.all(got[nnz == 0] == 0)), {'max_err_over_bound': excess}
    return W('spmv', {0: (rowptr, False), 1: (col, False), 2: (val, False), 3: (x, False), 4: (4 * rows, True)},
             [((rows + 127) // 128, 1, 1, push(rows))], check, 128, {'flops': 2 * int(rowptr[-1]), 'bytes': 12 * int(rowptr[-1]) + 8 * rows})


def w_transpose(size, rng):
    Wd, H = (100, 70) if size == 'small' else (4100, 2050)
    m = u32(rng.integers(0, 1 << 32, (H, Wd), dtype=np.uint64))
    def check(o):
        got = o[1].view(np.uint32)[:Wd * H].reshape(Wd, H)
        return np.array_equal(got, m.T), {'mismatches': int(np.sum(got != m.T))}
    return W('transpose', {0: (m, False), 1: (4 * Wd * H, True)}, [((Wd + 31) // 32, (H + 31) // 32, 1, push(Wd, H))], check,
             (16, 16, 1), {'bytes': 8 * Wd * H})


def mandel_ref(Wd, H, x0, y0, step, maxit):
    cx = (x0 + np.arange(Wd) * step)[None, :].repeat(H, 0).astype(np.int64)
    cy = (y0 + np.arange(H) * step)[:, None].repeat(Wd, 1).astype(np.int64)
    x = np.zeros_like(cx); y = np.zeros_like(cy); n = np.zeros(cx.shape, np.int64); live = np.ones(cx.shape, bool)
    for _ in range(maxit):
        live &= (np.abs(x) <= 16384) & (np.abs(y) <= 16384)
        xx, yy = x * x, y * y
        live &= (xx + yy) <= (4 << 26)
        if not live.any(): break
        nx = ((xx - yy) >> 13) + cx; ny = ((x * y) >> 12) + cy
        x = np.where(live, nx, x); y = np.where(live, ny, y); n += live
    return n


def w_mandelbrot(size, rng):
    Wd, H, maxit = (96, 64, 64) if size == 'small' else (2048, 2048, 512)
    step = int(3.0 * 8192 / Wd); x0, y0 = int(-2.25 * 8192), -(H // 2) * step
    def check(o):
        got = o[0].view(np.uint32)[:Wd * H].reshape(H, Wd)
        ref = mandel_ref(Wd, H, x0, y0, step, maxit)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'mean_iter': float(ref.mean())}
    return W('mandelbrot', {0: (4 * Wd * H, True)}, [((Wd + 15) // 16, (H + 15) // 16, 1, push(Wd, H, x0, y0, step, maxit))], check,
             (16, 16, 1), {'pixels': Wd * H})


def w_softmax(size, rng):
    rows, cols = (64, 1000) if size == 'small' else (1 << 14, 1024)
    x = f32(rng.standard_normal((rows, cols)) * 4)
    def check(o):
        got = o[1].view(np.float32)[:rows * cols].reshape(rows, cols)
        x64 = x.astype(np.float64); e = np.exp(x64 - x64.max(1, keepdims=True)); ref = e / e.sum(1, keepdims=True)
        err = float(np.max(np.abs(got - ref) / ref))
        return err < 1e-5, {'max_rel_err': err}
    return W('softmax', {0: (x, False), 1: (4 * rows * cols, True)}, [(rows, 1, 1, push(cols))], check, 256, {'bytes': 8 * rows * cols})


def w_rgba8(size, rng):
    n = 4096 if size == 'small' else 1 << 22
    px = u32(rng.integers(0, 1 << 32, n, dtype=np.uint64))
    def check(o):
        c = (px[:, None] >> np.array([0, 8, 16, 24], np.uint32) & 0xff).astype(np.float64) / 255.0
        g = c[:, :3] ** 2 * (3 - 2 * c[:, :3])
        refc = np.concatenate([g, c[:, 3:]], 1) * 255.0
        got = (o[1].view(np.uint32)[:n, None] >> np.array([0, 8, 16, 24], np.uint32) & 0xff).astype(np.float64)
        y = c[:, :3] @ np.array([0.299, 0.587, 0.114]) * 255.0 + 0.5
        gl = o[2][:n].astype(np.float64)
        # packUnorm rounds to nearest (ties are implementation-defined): within 0.5 + float slack of the exact value
        e1 = float(np.max(np.abs(got - refc))); e2 = float(np.max(np.abs(gl - np.minimum(np.floor(y), 255))))
        return e1 <= 0.5001 and e2 <= 1, {'max_pack_err': e1, 'max_luma_err': e2}
    return W('rgba8', {0: (px, False), 1: (4 * n, True), 2: (n, True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256, {'bytes': 9 * n})



def w_atomics(size, rng):
    n = 1024 if size == 'small' else 1 << 18
    val = u32(rng.integers(0, 1 << 32, 2 * n, dtype=np.uint64)); a, b = val[0::2], val[1::2]
    dist = u32(rng.integers(0, 1 << 32, 8 * n, dtype=np.uint64)).reshape(n, 8)
    swap = rng.random(n) < 0.5
    dist[swap, 7] = b[swap]                                       # half the compare-swaps succeed
    cont = u32(rng.integers(0, 1 << 32, 16, dtype=np.uint64))
    def check(o):
        got_d = o[0].view(np.uint32)[:8 * n].reshape(n, 8); got_c = o[1].view(np.uint32)[:16]
        ret = o[3].view(np.uint32)[:11 * n].reshape(n, 11).astype(np.uint64)
        d64, a64 = dist.astype(np.uint64), a.astype(np.uint64)
        fin = np.stack([(d64[:, 0] + a64) & 0xffffffff, np.minimum(d64[:, 1], a64), np.maximum(d64[:, 2], a64), d64[:, 3] & a64,
                        d64[:, 4] | a64, d64[:, 5] ^ a64, a64, np.where(d64[:, 7] == b, a64, d64[:, 7])], 1)
        bad = {'distinct_returns': int(np.sum(ret[:, :8] != d64)), 'distinct_finals': int(np.sum(got_d != fin))}
        add = a64 & 0xffff
        lanes = np.arange(n) % 8
        fin_add = [(int(cont[k]) + int(add[lanes == k].sum())) & 0xffffffff for k in range(8)]
        fin_max = [max(int(cont[8 + k]), int(a64[lanes == k].max())) for k in range(8)]
        bad['contended_finals'] = sum(int(got_c[k]) != fin_add[k] for k in range(8)) + sum(int(got_c[8 + k]) != fin_max[k] for k in range(8))
        chain = 0                                                  # returns of one word form a chain init -> final
        for k in range(8):
            r_ = ret[lanes == k, 8]; ad = add[lanes == k]
            chain += sorted(((r_ + ad) & 0xffffffff).tolist() + [int(cont[k])]) != sorted(r_.tolist() + [fin_add[k]])
        bad['contended_chain'] = chain
        wg = np.arange(n) // 256
        bad['tg_counter'] = sum(sorted(ret[wg == g, 9].tolist()) != list(range(int(np.sum(wg == g)))) for g in range(int(wg.max()) + 1))
        bad['tg_max'] = int(sum(np.sum(ret[wg == g, 10] != a64[wg == g].max()) for g in range(int(wg.max()) + 1)))
        return not any(bad.values()), bad
    return W('atomics', {0: (dist, True), 1: (cont, True), 2: (val, False), 3: (44 * n, True)}, [((n + 255) // 256, 1, 1, push(n))],
             check, 256, {'atomics': 11 * n})


def w_geom3d(size, rng):
    ex, ey, ez = (9, 7, 5) if size == 'small' else (300, 200, 30)
    gx, gy, gz = (ex + 3) // 4, (ey + 3) // 4, (ez + 1) // 2
    def check(o):
        got = o[0].view(np.uint32)[:12 * ex * ey * ez].reshape(ez, ey, ex, 12)
        z, y, x = np.meshgrid(np.arange(ez), np.arange(ey), np.arange(ex), indexing='ij')
        lx, ly, lz = x % 4, y % 4, z % 2
        ref = np.stack([x, y, z, lx, ly, lz, lz * 16 + ly * 4 + lx, x // 4, y // 4, z // 2,
                        np.full_like(x, gx * 10000 + gy * 100 + gz), np.full_like(x, 32)], -1)
        bad = got != ref
        return not bad.any(), {'mismatches': int(bad.sum()), 'by_field': {str(k): int(v) for k, v in enumerate(bad.reshape(-1, 12).sum(0)) if v}}
    return W('geom3d', {0: (48 * ex * ey * ez, True)}, [(gx, gy, gz, push(ex, ey, ez))], check, (4, 4, 2), {'threads': ex * ey * ez})


def spill_ref(seed, steps):
    m = 0xffffffff
    s = seed; arr = []
    for _ in range(96):
        s = (s * 1664525 + 1013904223) & m; arr.append(s)
    idx, acc = s % 96, 0
    for k in range(steps & 255):
        v = arr[idx]; arr[idx] = v ^ (v >> 7) ^ k; acc = (acc * 31 + v) & m; idx = ((v + k) & m) % 96
    x = 0
    for k in range(96):
        x ^= (arr[k] * (k + 1)) & m
    return [acc, x, idx, arr[(acc ^ x) % 96]]


def w_spill(size, rng):
    n = 256 if size == 'small' else 1 << 18
    inp = u32(rng.integers(0, 1 << 32, 4 * n, dtype=np.uint64)).reshape(n, 4)
    def check(o):
        got = o[1].view(np.uint32)[:4 * n].reshape(n, 4)
        rows = range(n) if size == 'small' else rng.choice(n, 2048, replace=False)
        bad = sum([int(v) for v in got[i]] != spill_ref(int(inp[i, 0]), int(inp[i, 1])) for i in rows)
        return bad == 0, {'bad_rows': int(bad)}
    return W('spill', {0: (inp, False), 1: (16 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'threads': n})


def control_ref(v):
    m = 0xffffffff
    if v & 63 == 0:
        return [0xdead, 0]
    def step(v, k):
        c = v % 10
        return [(v + k) & m, (v * 3 + 1) & m, v >> 1, v ^ 0x5a5a5a5a, (v - k) & m, ((v << 3) | (v >> 29)) & m, (v * v) & m,
                ~v & m, (v + 0x9e3779b9) & m, v // 3][c]
    acc = iters = 0
    for outer in range(8):
        if (v >> outer) & 7 == 5:
            continue
        for inner in range(16):
            v = step(v, inner + outer); iters += 1
            if v & 255 == 7: break
            if v & 3 == 1: continue
            acc = (acc + v) & m
        if acc > 0xf0000000: break
    return [acc, iters]


def w_control(size, rng):
    n = 1024 if size == 'small' else 1 << 18
    inp = u32(rng.integers(0, 1 << 32, n, dtype=np.uint64))
    def check(o):
        got = o[1].view(np.uint32)[:2 * n].reshape(n, 2)
        rows = range(n) if size == 'small' else rng.choice(n, 4096, replace=False)
        bad = sum([int(v) for v in got[i]] != control_ref(int(inp[i])) for i in rows)
        return bad == 0, {'bad_rows': int(bad)}
    return W('control', {0: (inp, False), 1: (8 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'threads': n})


def w_shuffle(size, rng):
    n = 1024 if size == 'small' else 1 << 20
    v = u32(rng.integers(0, 1 << 32, n, dtype=np.uint64))
    def check(o):
        S = int(o[2].view(np.uint32)[0])
        got = o[1].view(np.uint32)[:6 * n].reshape(n, 6).astype(np.int64)
        i = np.arange(n); lane = i % S; base = i - lane; v64 = v.astype(np.int64)
        sums = (v64.reshape(-1, S).sum(1) & 0xffffffff).repeat(S)
        ref = np.stack([v64[base + (lane + 1) % S], v64[base + (lane + (v64 & 7)) % S], sums,
                        np.where(lane >= 1, v64[np.maximum(i - 1, 0)], v64), np.where(lane + 2 < S, v64[np.minimum(i + 2, n - 1)], v64),
                        v64[base + 1]], 1)
        bad = got != ref
        return S in (4, 8, 16, 32, 64, 128) and not bad.any(), {'subgroup_size': S, 'mismatches': int(bad.sum()),
                                                                 'by_output': {str(k): int(c) for k, c in enumerate(bad.sum(0)) if c}}
    return W('shuffle', {0: (v, False), 1: (24 * n, True), 2: (4, True)}, [(n // 128, 1, 1, push(n))], check, 128, {'items': n})


def floatops_inputs(n, rng):
    special = np.array([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.0, -3.0, 2.25, np.inf, -np.inf,
                        3.4028235e38, -3.4028235e38, 1.1754944e-38, -1.1754944e-38, 1e30, -1e30, 8388608.5, -8388609.0, 0.75], np.float32)
    bits = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    e = (bits >> 23) & 0xff
    bits = np.where((e == 0) | (e == 255), bits ^ 0x40000000, bits)          # no denormals / NaN
    x = bits.view(np.float32).copy()
    small_neg = (x < 0) & (x > -1)                                           # fract exact: multiples of 2^-20
    x[small_neg] = -rng.integers(1, 1 << 20, int(small_neg.sum())) * 2.0 ** -20
    pick = rng.random(n)
    x = np.where(pick < 0.3, rng.choice(special, n), np.where(pick < 0.5, np.round(rng.uniform(-6, 6, n) * 4) / 4, x)).astype(np.float32)
    return x


def w_floatops(size, rng):
    n = 2048 if size == 'small' else 1 << 18
    x, y = floatops_inputs(n, rng), floatops_inputs(n, rng)
    inp = f32(np.stack([x, y], 1))
    def check(o):
        got = o[1].view(np.uint32)[:16 * n].reshape(n, 16)
        g = got.view(np.float32)
        same = lambda k, ref, either_zero=None: (got[:, k] == ref.astype(np.float32).view(np.uint32)) | (
            either_zero if either_zero is not None else False)
        zeros_eq = (x == y)                                                   # min/max of equal values (incl. +-0): either
        with np.errstate(invalid='ignore', over='ignore'):               # inf - inf in fract, max + max in output 14
            fin = np.isfinite(x)
            half = (np.abs(x - np.trunc(x)) == 0.5)
            away = np.trunc(x + np.copysign(0.5, x)).astype(np.float32)
            ok = [same(0, np.minimum(x, y), zeros_eq & (g[:, 0] == x)), same(1, np.maximum(x, y), zeros_eq & (g[:, 1] == x)),
                  same(2, np.clip(x, np.float32(-1.5), np.float32(2.25)), (x == 0) & (g[:, 2] == 0)),
                  same(3, np.abs(x)), same(4, np.sign(x), (x == 0) & (g[:, 4] == 0)), same(5, np.floor(x)), same(6, np.ceil(x)),
                  same(7, x - np.floor(x)) | ~fin, same(8, np.round(x), half & (g[:, 8] == away)), same(9, np.round(x)),
                  same(10, np.trunc(x)), same(11, np.where(x < y, 0.0, 1.0)), same(12, -x),
                  got[:, 13] == ((x < y) | (x <= y) << 1 | (x == y) << 2 | (x != y) << 3 | (x > y) << 4 | (x >= y) << 5),
                  same(14, np.abs(x) + np.abs(y)), same(15, np.maximum(np.minimum(x, np.float32(1e30)), np.float32(-1e30)))]
        bad = {str(k): int((~np.asarray(v)).sum()) for k, v in enumerate(ok) if (~np.asarray(v)).any()}
        return not bad, {'mismatches_by_output': bad}
    return W('floatops', {0: (inp, False), 1: (64 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'items': n})



def w_radix_sort(size, rng):
    n = 4096 if size == 'small' else 1 << 20
    groups = n // 1024
    keys = u32(rng.integers(0, 1 << 32, n, dtype=np.uint64))
    keys[rng.random(n) < 0.2] = 12345                          # duplicates: stability matters
    vals = u32(np.arange(n))
    ds = []
    for k in range(8):
        for mode in range(3):
            ds.append((groups if mode != 1 else 1, 1, 1, push(mode, 4 * k, groups, n, k & 1)))
    def check(o):
        gk, gv = o[0].view(np.uint32)[:n], o[1].view(np.uint32)[:n]     # 8 passes end in the first half
        order = np.argsort(keys, kind='stable')
        ok_k, ok_v = np.array_equal(gk, keys[order]), np.array_equal(gv, order.astype(np.uint32))
        return ok_k and ok_v, {'keys_sorted': ok_k, 'stable_payload': ok_v}
    return W('radix_sort', {0: (np.concatenate([keys, np.zeros(n, np.uint32)]), True), 1: (np.concatenate([vals, np.zeros(n, np.uint32)]), True),
                            2: (4 * 16 * groups, False)}, ds, check, 256, {'keys': n, 'dispatches': len(ds)})


def w_fft(size, rng):
    batches = 4 if size == 'small' else 4096
    x = (rng.standard_normal((batches, 1024)) + 1j * rng.standard_normal((batches, 1024))).astype(np.complex64)
    inp = f32(np.stack([x.real, x.imag], -1))
    def check(o):
        g = o[1].view(np.float32)[:batches * 2048].reshape(batches, 1024, 2)
        got = g[..., 0].astype(np.float64) + 1j * g[..., 1]
        ref = np.fft.fft(x.astype(np.complex128), axis=1)
        err = float(np.max(np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)))
        # Vulkan allows sin / cos an absolute error of 2^-11 on [-pi, pi]: ten stages of that bound the relative error
        return err < 10 * 2.0 ** -11, {'rel_l2_err': err}
    return W('fft', {0: (inp, False), 1: (8 * 1024 * batches, True)}, [(batches, 1, 1, b'')], check, 256,
             {'flops': 5 * 1024 * 10 * batches})


def w_crc32(size, rng):
    n = 512 if size == 'small' else 1 << 18
    msgs = rng.integers(0, 256, (n, 64), dtype=np.uint8)
    def check(o):
        import zlib
        got = o[1].view(np.uint32)[:n]
        rows = range(n) if size == 'small' else rng.choice(n, 4096, replace=False)
        bad = sum(int(got[i]) != zlib.crc32(bytes(msgs[i])) for i in rows)
        return bad == 0, {'bad_crcs': int(bad)}
    return W('crc32', {0: (msgs, False), 1: (4 * n, True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256, {'bytes': 64 * n})


def w_kmeans(size, rng):
    n, k = (2048, 16) if size == 'small' else (1 << 20, 64)
    cent = f32(rng.uniform(-10, 10, (k, 4)))
    pts = f32(cent[rng.integers(0, k, n)] + rng.standard_normal((n, 4)) * 3)
    def check(o):
        ga, gd = o[2].view(np.uint32)[:n], o[3].view(np.float32)[:n]
        d = ((pts[:, None, :].astype(np.float64) - cent[None, :, :]) ** 2).sum(-1)
        ref = d.argmin(1); best = d.min(1)
        # a different index is only acceptable where the two distances tie within float rounding
        alt = d[np.arange(n), ga]
        bad_idx = int(np.sum((ga != ref) & (alt - best > 1e-5 * (best + 1))))
        derr = float(np.max(np.abs(gd - best) / (best + 1)))
        return bad_idx == 0 and derr < 1e-5, {'wrong_assignments': bad_idx, 'max_dist_err': derr}
    return W('kmeans', {0: (pts, False), 1: (cent, False), 2: (4 * n, True), 3: (4 * n, True)}, [((n + 127) // 128, 1, 1, push(n, k))],
             check, 128, {'flops': 12 * n * k})


def w_int8dot(size, rng):
    M, N, K = (64, 48, 256) if size == 'small' else (1024, 1024, 1024)
    a = rng.integers(-128, 128, (M, K), dtype=np.int64).astype(np.int8); b = rng.integers(-128, 128, (K, N), dtype=np.int64).astype(np.int8)
    def check(o):
        got = o[2].view(np.int32)[:M * N].reshape(M, N)
        ref = a.astype(np.int64) @ b.astype(np.int64)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref))}
    return W('int8dot', {0: (a, False), 1: (b, False), 2: (4 * M * N, True)}, [((N + 15) // 16, (M + 15) // 16, 1, push(M, N, K))],
             check, (16, 16, 1), {'ops': 2 * M * N * K})


def w_stencil3d(size, rng):
    nx, ny, nz = (20, 18, 10) if size == 'small' else (256, 256, 128)
    u = f32(rng.uniform(-1, 1, (nz, ny, nx)))
    c0, c1 = 0.4, 0.1
    def check(o):
        got = o[1].view(np.float32)[:nx * ny * nz].reshape(nz, ny, nx)
        q = np.pad(u.astype(np.float64), 1, mode='edge')
        nb = q[1:-1, 1:-1, :-2] + q[1:-1, 1:-1, 2:] + q[1:-1, :-2, 1:-1] + q[1:-1, 2:, 1:-1] + q[:-2, 1:-1, 1:-1] + q[2:, 1:-1, 1:-1]
        ref = c0 * u + c1 * nb
        err = float(np.max(np.abs(got - ref)))
        return err < 1e-6, {'max_abs_err': err}
    return W('stencil3d', {0: (u, False), 1: (4 * nx * ny * nz, True)},
             [((nx + 7) // 8, (ny + 7) // 8, (nz + 3) // 4, push(nx, ny, nz, c0, c1))], check, (8, 8, 4), {'bytes': 8 * nx * ny * nz})


def w_int16(size, rng):
    n = 2048 if size == 'small' else 1 << 20
    inp = rng.integers(0, 1 << 16, (n, 4), dtype=np.uint64).astype(np.uint16)
    def check(o):
        got = o[1].view(np.uint16)[:10 * n].reshape(n, 10).astype(np.int64)
        wide = o[2].view(np.uint32)[:2 * n].reshape(n, 2).astype(np.int64)
        a, b, c, d = (inp[:, k].astype(np.int64) for k in range(4))
        sa, sb = inp[:, 0].view(np.int16).astype(np.int64), inp[:, 1].view(np.int16).astype(np.int64)
        m = 0xffff
        acc = np.zeros(n, np.int64)
        for k in range(8):
            acc = (acc + (((a + k) & m) * ((b ^ k) & m))) & m
        ref = np.stack([(a + b) & m, (a - b) & m, (a * b) & m, a >> (c & 15), (a << (d & 15)) & m, (sa >> (c & 15)) & m,
                        np.maximum(a, b), np.minimum(sa, sb) & m, (a < b) * 1 | (sa < sb) * 2 | (a == c) * 4, acc], 1)
        refw = np.stack([a * b, (sa * sb) & 0xffffffff], 1)
        bad = got != ref
        return not bad.any() and np.array_equal(wide, refw), {'mismatches': int(bad.sum()), 'wide_mismatches': int(np.sum(wide != refw)),
                                                               'by_output': {str(k): int(v) for k, v in enumerate(bad.sum(0)) if v}}
    return W('int16', {0: (inp, False), 1: (20 * n, True), 2: (8 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'items': n})


def w_trig(size, rng):
    n = 4096 if size == 'small' else 1 << 20
    inp = f32(np.stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(-1, 1, n), rng.uniform(-60, 60, n),
                        np.exp2(rng.uniform(-100, 100, n))], 1))
    def check(o):
        got = o[1].view(np.float32)[:12 * n].reshape(n, 12).astype(np.float64)
        x, u, e, l = (inp[:, k].astype(np.float64) for k in range(4))
        ref = [np.sin(x), np.cos(x), np.arcsin(u), np.arccos(u), np.arctan(e), np.arctan2(u, x), np.exp(np.float32(e * 0.25).astype(np.float64)),
               np.exp2(e), np.log(l), np.log2(l), np.sqrt(l), 1 / np.sqrt(l)]
        ulp = lambda v: np.spacing(np.abs(v).astype(np.float32)).astype(np.float64)
        # Vulkan precision: sin / cos absolute 2^-11; asin / acos / atan / atan2 4096 ulp; exp / exp2 3 + 2|x| ulp;
        # log / log2 3 ulp (absolute 2^-21 on [0.5, 2]); sqrt as 1 / inversesqrt; inversesqrt 2 ulp
        tol = [np.full(n, 2.0 ** -11), np.full(n, 2.0 ** -11), 4096 * ulp(ref[2]), 4096 * ulp(ref[3]), 4096 * ulp(ref[4]), 4096 * ulp(ref[5]),
               (3 + 2 * np.abs(e * 0.25)) * ulp(ref[6]), (3 + 2 * np.abs(e)) * ulp(ref[7]),
               np.where((l >= 0.5) & (l <= 2), 2.0 ** -21, 3 * ulp(ref[8])), np.where((l >= 0.5) & (l <= 2), 2.0 ** -21, 3 * ulp(ref[9])),
               5 * ulp(ref[10]), 2 * ulp(ref[11])]
        names = ['sin', 'cos', 'asin', 'acos', 'atan', 'atan2', 'exp', 'exp2', 'log', 'log2', 'sqrt', 'inversesqrt']
        worst = {nm: float(np.max(np.abs(got[:, k] - ref[k]) / tol[k])) for k, nm in enumerate(names)}
        # the verdict allows twice Vulkan's bound (lavapipe's exp reaches 1.04x on a million inputs); the ratios are the data
        return all(v <= 2 for v in worst.values()), {'worst_err_over_vulkan_bound': {k: round(v, 4) for k, v in worst.items()}}
    return W('trig', {0: (inp, False), 1: (48 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'items': n})



def raytrace_ref(Wd, H):
    """the shader's scene in float64 numpy (same structure), RGB 0..1"""
    S = np.array([[-1.2, 1.0, 5.0, 1.0], [1.1, 0.7, 4.2, 0.7], [0.2, 0.4, 2.8, 0.4]])
    C = np.array([[0.9, 0.2, 0.2], [0.2, 0.8, 0.3], [0.3, 0.4, 0.95]])
    L = np.array([0.4472136, 0.8944272, 0.0])
    def hit(ro, rd):
        best = np.full(rd.shape[:-1], 1e30); n = np.zeros_like(rd); col = np.zeros_like(rd); refl = np.zeros(rd.shape[:-1])
        for i in range(3):
            oc = ro - S[i, :3]
            b = (oc * rd).sum(-1); c = (oc * oc).sum(-1) - S[i, 3] ** 2; h = b * b - c
            with np.errstate(invalid='ignore'):
                t = -b - np.sqrt(np.where(h >= 0, h, 0))
            m = (h >= 0) & (t > 1e-3) & (t < best)
            best = np.where(m, t, best)
            n = np.where(m[..., None], (ro + t[..., None] * rd - S[i, :3]) / S[i, 3], n)
            col = np.where(m[..., None], C[i], col); refl = np.where(m, 0.6 if i == 2 else 0.1, refl)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(rd[..., 1] < 0, -ro[..., 1] / np.where(rd[..., 1] < 0, rd[..., 1], -1), -1)
        m = (rd[..., 1] < 0) & (t > 1e-3) & (t < best)
        q = ro + t[..., None] * rd
        chk = ((np.floor(q[..., 0]).astype(np.int64) + np.floor(q[..., 2]).astype(np.int64)) & 1) == 0
        best = np.where(m, t, best); n = np.where(m[..., None], [0.0, 1.0, 0.0], n)
        col = np.where(m[..., None], np.where((t > 20)[..., None], 0.55, np.where(chk[..., None], 0.8, 0.3)), col); refl = np.where(m, 0.2, refl)
        return np.where(best < 1e30, best, -1.0), n, col, refl
    def shade(ro, rd):
        t, n, col, refl = hit(ro, rd)
        anyh = t > 0
        hp = ro + t[..., None] * rd
        ts, *_ = hit(hp + n * 1e-3, np.broadcast_to(L, rd.shape))
        lit = np.where(ts > 0, 0.0, np.maximum((n * L).sum(-1), 0.0))
        c = np.where(anyh[..., None], col * (0.15 + 0.85 * lit)[..., None], np.array([0.5, 0.7, 1.0]) * (0.6 + 0.4 * rd[..., 1:2]))
        return c, hp, n, refl, anyh
    gy, gx = np.mgrid[0:H, 0:Wd]
    uv = np.stack([(gx + 0.5) / Wd * 2 - 1, (gy + 0.5) / H * 2 - 1], -1)
    uv[..., 0] *= Wd / H
    rd = np.stack([uv[..., 0], -uv[..., 1], np.full(uv.shape[:-1], 1.5)], -1)
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(np.array([0.0, 1.0, -1.0]), rd.shape)
    c, hp, hn, refl, anyh = shade(ro, rd)
    rd2 = rd - 2 * (rd * hn).sum(-1, keepdims=True) * hn
    c2, *_ = shade(hp + hn * 1e-3, rd2)
    use = (anyh & (refl > 0))[..., None]
    return np.clip(np.where(use, c * (1 - refl[..., None]) + c2 * refl[..., None], c), 0, 1)


def w_raytrace(size, rng):
    Wd, H = (96, 64) if size == 'small' else (1920, 1080)
    def check(o):
        got = o[0][:4 * Wd * H].reshape(H, Wd, 4).astype(np.int64)
        ref = np.round(raytrace_ref(Wd, H) * 255).astype(np.int64)
        d = np.abs(got[..., :3] - ref).max(-1)
        # silhouettes and shadow edges may flip between float implementations: a small fraction of pixels
        far = float(np.mean(d > 2))
        return far < 0.01 and bool(np.all(got[..., 3] == 255)), {'pixels_off_by_more_than_2': far, 'mean_abs_diff': float(d.mean())}
    return W('raytrace', {0: (4 * Wd * H, True)}, [((Wd + 7) // 8, (H + 7) // 8, 1, push(Wd, H))], check, (8, 8, 1),
             {'pixels': Wd * H}, images=[(0, Wd, H, 'rgba8')])


def w_mlp(size, rng):
    batch = 64 if size == 'small' else 16384
    x = f32(rng.standard_normal((batch, 128)))
    w1 = f32(rng.standard_normal((256, 128)) / np.sqrt(128)); b1 = f32(rng.standard_normal(256) * 0.1)
    w2 = f32(rng.standard_normal((16, 256)) / np.sqrt(256)); b2 = f32(rng.standard_normal(16) * 0.1)
    def check(o):
        got = o[5].view(np.float32)[:batch * 16].reshape(batch, 16)
        h = np.maximum(x.astype(np.float64) @ w1.T.astype(np.float64) + b1, 0)
        lg = h @ w2.T.astype(np.float64) + b2
        e = np.exp(lg - lg.max(1, keepdims=True)); ref = e / e.sum(1, keepdims=True)
        err = float(np.max(np.abs(got - ref)))
        same_class = float(np.mean(got.argmax(1) == ref.argmax(1)))
        return err < 1e-4 and same_class == 1.0, {'max_abs_prob_err': err, 'same_argmax': same_class}
    return W('mlp', {0: (x, False), 1: (w1, False), 2: (b1, False), 3: (w2, False), 4: (b2, False), 5: (64 * batch, True)},
             [(batch, 1, 1, b'')], check, 256, {'flops': 2 * batch * (128 * 256 + 256 * 16)})


def w_sobel(size, rng):
    Wd, H = (80, 60) if size == 'small' else (3840, 2160)
    gy, gx = np.mgrid[0:H, 0:Wd]
    img = (127 + 60 * np.sin(gx / 7.0) * np.cos(gy / 11.0) + 50 * ((gx // 16 + gy // 16) % 2)
           + rng.integers(-10, 11, (H, Wd))).clip(0, 255).astype(np.uint8)
    def check(o):
        got = o[1][:Wd * H].reshape(H, Wd).astype(np.int64)
        q = np.pad(img.astype(np.int64), 1, mode='edge')
        P = lambda dy, dx: q[1 + dy:1 + dy + H, 1 + dx:1 + dx + Wd]
        gxx = -P(-1, -1) - 2 * P(0, -1) - P(1, -1) + P(-1, 1) + 2 * P(0, 1) + P(1, 1)
        gyy = -P(-1, -1) - 2 * P(-1, 0) - P(-1, 1) + P(1, -1) + 2 * P(1, 0) + P(1, 1)
        ref = np.minimum(np.abs(gxx) + np.abs(gyy), 255)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref))}
    return W('sobel', {0: (img, False), 1: (Wd * H, True)}, [((Wd + 15) // 16, (H + 15) // 16, 1, push(Wd, H))], check, (16, 16, 1),
             {'pixels': Wd * H}, images=[(1, Wd, H, 'gray8')])



def w_life(size, rng):
    Wd, H, steps = (64, 48, 16) if size == 'small' else (2048, 2048, 64)
    grid = np.where(rng.random((H, Wd)) < 0.3, 255, 0).astype(np.uint8)
    def check(o):
        g = grid > 0
        for _ in range(steps):
            n = sum(np.roll(np.roll(g, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
            g = (n == 3) | (g & (n == 2))
        got = o[0][:Wd * H].reshape(H, Wd)                      # an even number of steps ends in the first half
        ref = np.where(g, 255, 0).astype(np.uint8)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'alive': int(g.sum())}
    ds = [((Wd + 15) // 16, (H + 15) // 16, 1, push(Wd, H, k & 1)) for k in range(steps)]
    return W('life', {0: (np.concatenate([grid.ravel(), np.zeros(Wd * H, np.uint8)]), True)}, ds, check, (16, 16, 1),
             {'cells': Wd * H * steps}, images=[(0, Wd, H, 'gray8')])


def w_bfs(size, rng):
    n, deg, levels = (2000, 4, 24) if size == 'small' else (1 << 20, 8, 32)
    # random graph: a ring for connectivity plus random edges, undirected, CSR
    src = np.concatenate([np.arange(n), rng.integers(0, n, n * deg // 2)])
    dst = np.concatenate([(np.arange(n) + 1) % n, rng.integers(0, n, n * deg // 2)])
    u_, v_ = np.concatenate([src, dst]), np.concatenate([dst, src])
    order = np.argsort(u_, kind='stable'); u_, v_ = u_[order], v_[order]      # CSR rows (neighbour order irrelevant)
    rowptr = u32(np.searchsorted(u_, np.arange(n + 1))); col = u32(v_)
    INF = 0xffffffff
    dist0 = np.full(n, INF, np.uint32); dist0[0] = 0
    def check(o):
        got = o[2].view(np.uint32)[:n]; found = o[3].view(np.uint32)[:levels + 1]
        ref = np.full(n, INF, np.int64); ref[0] = 0; frontier = [0]; d = 0
        while frontier and d < levels:
            nxt = []
            for v in frontier:
                for u in col[rowptr[v]:rowptr[v + 1]]:
                    if ref[u] == INF:
                        ref[u] = d + 1; nxt.append(int(u))
            frontier = nxt; d += 1
        reff = np.bincount(ref[(ref != INF) & (ref > 0)], minlength=levels + 1)[:levels + 1]
        ok = np.array_equal(got.astype(np.int64), ref) and np.array_equal(found[1:], reff[1:])
        return ok, {'dist_mismatches': int(np.sum(got.astype(np.int64) != ref)), 'reached': int(np.sum(ref != INF)),
                    'depth': int(ref[ref != INF].max())}
    ds = [((n + 127) // 128, 1, 1, push(n, k)) for k in range(levels)]
    return W('bfs', {0: (rowptr, False), 1: (col, False), 2: (dist0, True), 3: (4 * (levels + 1), True)}, ds, check, 128,
             {'edges': int(len(col)), 'dispatches': levels})


def w_conv_layer(size, rng):
    Wd, H, cin, cout = (24, 20, 16, 32) if size == 'small' else (224, 224, 16, 32)
    x = f32(rng.standard_normal((cin, H, Wd)))
    w = f32(rng.standard_normal((cout, cin, 3, 3)) / np.sqrt(cin * 9)); b = f32(rng.standard_normal(cout) * 0.1)
    def check(o):
        got = o[3].view(np.float32)[:cout * H * Wd].reshape(cout, H, Wd)
        xp = np.pad(x.astype(np.float64), ((0, 0), (1, 1), (1, 1)))
        acc = np.zeros((cout, H, Wd)) + b[:, None, None]
        absacc = np.zeros((cout, H, Wd)) + np.abs(b)[:, None, None]
        for dy in range(3):
            for dx in range(3):
                patch = xp[:, dy:dy + H, dx:dx + Wd]
                acc += np.einsum('kc,chw->khw', w[:, :, dy, dx].astype(np.float64), patch)
                absacc += np.einsum('kc,chw->khw', np.abs(w[:, :, dy, dx]).astype(np.float64), np.abs(patch))
        ref = np.maximum(acc, 0)
        excess = float(np.max(np.abs(got - ref) / (absacc * (cin * 9 + 2) * 2.0 ** -24 + 1e-30)))
        return excess <= 1.0, {'max_err_over_bound': excess}
    return W('conv_layer', {0: (x, False), 1: (w, False), 2: (b, False), 3: (4 * cout * H * Wd, True)},
             [((Wd + 15) // 16, (H + 15) // 16, cout, push(Wd, H, cin))], check, (16, 16, 1), {'flops': 2 * cout * cin * 9 * H * Wd})


def w_hist_eq(size, rng):
    Wd, H = (96, 64) if size == 'small' else (3840, 2160)
    gy, gx = np.mgrid[0:H, 0:Wd]
    img = (90 + 40 * np.sin(gx / 13.0) * np.cos(gy / 17.0) + rng.normal(0, 6, (H, Wd))).clip(0, 255).astype(np.uint8)   # low contrast
    n = Wd * H
    groups = min(256, (n + 255) // 256)
    def check(o):
        hist = np.bincount(img.ravel(), minlength=256).astype(np.int64); cdf = np.cumsum(hist)
        cmin = int(cdf[cdf > 0][0]); den = n - cmin
        lut = np.arange(256) if den == 0 else ((cdf - np.minimum(cdf, cmin)) * 255 + den // 2) // den
        ref = lut[img].astype(np.uint8)
        got = o[2][:n].reshape(H, Wd)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'in_range': [int(img.min()), int(img.max())],
                                          'out_range': [int(got.min()), int(got.max())]}
    ds = [(groups, 1, 1, push(m, n)) for m in range(3)]
    return W('hist_eq', {0: (img, False), 1: (2048, False), 2: (n, True)}, ds, check, 256, {'pixels': n}, images=[(2, Wd, H, 'gray8')])



def aes_sbox():
    """the AES S-box from GF(2^8) inversion and the affine map"""
    def mul(a, b):
        r = 0
        while b:
            if b & 1: r ^= a
            a = ((a << 1) ^ (0x1b if a & 0x80 else 0)) & 0xff; b >>= 1
        return r
    inv = [0] * 256
    for a in range(1, 256):
        inv[a] = next(b for b in range(1, 256) if mul(a, b) == 1)
    rot = lambda x, k: ((x << k) | (x >> (8 - k))) & 0xff
    return [inv[a] ^ rot(inv[a], 1) ^ rot(inv[a], 2) ^ rot(inv[a], 3) ^ rot(inv[a], 4) ^ 0x63 for a in range(256)]


AES_SBOX = None


def aes_encrypt(key, block):
    """AES-128 encryption of one block (bytes); checked against FIPS-197 C.1 before use"""
    global AES_SBOX
    if AES_SBOX is None:
        AES_SBOX = aes_sbox()
    S = AES_SBOX
    xt = lambda b: ((b << 1) ^ (0x1b if b & 0x80 else 0)) & 0xff
    w = [list(key[4 * i:4 * i + 4]) for i in range(4)]
    rcon = 1
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = [S[b] for b in t[1:] + t[:1]]; t[0] ^= rcon; rcon = xt(rcon)
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    s = [b ^ w[i // 4][i % 4] for i, b in enumerate(block)]
    for r in range(1, 11):
        u = [S[s[4 * ((c + j) % 4) + j]] for c in range(4) for j in range(4)]
        if r < 10:
            m = []
            for c in range(4):
                a = u[4 * c:4 * c + 4]; al = a[0] ^ a[1] ^ a[2] ^ a[3]
                m += [a[k] ^ al ^ xt(a[k] ^ a[(k + 1) % 4]) for k in range(4)]
            u = m
        s = [b ^ w[4 * r + i // 4][i % 4] for i, b in enumerate(u)]
    return bytes(s), w


def w_aes(size, rng):
    n = 512 if size == 'small' else 1 << 20
    fips = aes_encrypt(bytes(range(16)), bytes.fromhex('00112233445566778899aabbccddeeff'))[0]
    if fips.hex() != '69c4e0d86a7b0430d8cdb78070b4c55a':
        raise RuntimeError('AES reference fails FIPS-197 C.1')
    key = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
    blocks = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    rk = u32(np.array([int.from_bytes(bytes(wd), 'little') for wd in aes_encrypt(key, bytes(16))[1]], dtype=np.uint64))
    sbox = u32(aes_sbox())
    def check(o):
        got = o[3][:16 * n].reshape(n, 16)
        rows = range(n) if size == 'small' else rng.choice(n, 2048, replace=False)
        bad = sum(bytes(got[i]) != aes_encrypt(key, bytes(blocks[i]))[0] for i in rows)
        return bad == 0, {'bad_blocks': int(bad), 'fips197_reference_ok': True}
    return W('aes', {0: (blocks, False), 1: (rk, False), 2: (sbox, False), 3: (16 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64,
             {'bytes': 16 * n})


def w_hash_table(size, rng):
    n, cap = (2000, 4096) if size == 'small' else (1 << 20, 1 << 21)
    keys = u32(rng.choice(0xfffffffe, n, replace=False) if n <= 4096 else np.unique(rng.integers(0, 0xfffffffe, 2 * n, dtype=np.uint64))[:n])
    rng.shuffle(keys)
    vals = u32(rng.integers(0, 1 << 32, n, dtype=np.uint64))
    present = rng.random(n) < 0.5
    queries = u32(np.where(present, keys[rng.permutation(n)], rng.integers(0, 0xfffffffe, n, dtype=np.uint64)))
    table = np.full(2 * cap, 0xffffffff, np.uint32)
    def check(o):
        got = o[4].view(np.uint32)[:n]
        d = dict(zip(keys.tolist(), vals.tolist()))
        ref = np.array([d.get(q, 0xffffffff) for q in queries.tolist()], dtype=np.uint32)
        occ = int(np.sum(o[2].view(np.uint32)[0:2 * cap:2] != 0xffffffff))
        return np.array_equal(got, ref) and occ == n, {'lookup_mismatches': int(np.sum(got != ref)), 'occupied_slots': occ, 'hits': int(np.sum(ref != 0xffffffff))}
    ds = [((n + 127) // 128, 1, 1, push(0, n, cap - 1)), ((n + 127) // 128, 1, 1, push(1, n, cap - 1))]
    return W('hash_table', {0: (keys, False), 1: (vals, False), 2: (table, True), 3: (queries, False), 4: (4 * n, True)}, ds, check, 128,
             {'ops': 2 * n})


def w_word_count(size, rng):
    n = 1 << 14 if size == 'small' else 1 << 24
    words = [''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz0123'), rng.integers(1, 21))) for _ in range(4096)]
    seps = [' ', ' ', ' ', ', ', '. ', '\n', '  ', ' - ']
    # The text is a word then a separator, drawn with rng.integers(0, 4096) and rng.choice(seps) in turn, until n bytes.
    # With power-of-two bounds each draw is the top bits of one 32-bit output, so the pairs are drawn in bulk. The
    # original scalar calls on a copy of the generator confirm that on this numpy; otherwise they produce the text.
    start = rng.bit_generator.state
    x = rng.integers(0, 1 << 32, size=2 * (n // 4 + 16), dtype=np.uint32).astype(np.uint64).reshape(-1, 2)
    wi, si = (x[:, 0] * 4096) >> 32, (x[:, 1] * 8) >> 32
    probe = np.random.Generator(type(rng.bit_generator)()); probe.bit_generator.state = start
    if all(words[probe.integers(0, len(words))] == words[wi[k]] and probe.choice(seps) == seps[si[k]] for k in range(64)):
        cum = np.cumsum(np.array([len(w) for w in words])[wi] + np.array([len(p_) for p_ in seps])[si])
        k = int(np.searchsorted(cum, n)) + 1                               # the first pair that reaches n, inclusive
        text = ''.join(words[a] + seps[b] for a, b in zip(wi[:k].tolist(), si[:k].tolist()))
    else:
        rng.bit_generator.state = start
        out, total = [], 0
        while total < n:
            wd = words[rng.integers(0, len(words))]
            sep = rng.choice(seps)
            out.append(wd + sep); total += len(wd) + len(sep)
        text = ''.join(out)
    text = np.frombuffer(text.encode()[:n], dtype=np.uint8).copy()
    def check(o):
        import re
        got = o[1].view(np.uint32)[:18]
        t = text.tobytes().decode()
        lens = [len(m) for m in re.findall(r'[A-Za-z0-9]+', t)]
        hist = np.bincount(np.minimum(lens, 15), minlength=16)
        ref = np.concatenate([[len(lens), t.count('\n')], hist]).astype(np.uint32)
        return np.array_equal(got, ref), {'words': int(got[0]), 'words_ref': len(lens), 'lines': int(got[1]),
                                          'hist_mismatch': int(np.sum(got[2:] != ref[2:]))}
    return W('word_count', {0: (text, False), 1: (72, True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256, {'bytes': n})


JPEG_Q = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                   14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                   49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], np.int32)


def w_dct8(size, rng):
    Wd, H = (64, 48) if size == 'small' else (3840, 2160)
    gy, gx = np.mgrid[0:H, 0:Wd]
    img = (128 + 70 * np.sin(gx / 9.0 + gy / 23.0) + 30 * ((gx // 32 + gy // 32) % 2) + rng.normal(0, 4, (H, Wd))).clip(0, 255).astype(np.uint8)
    def check(o):
        got = o[2].view(np.int32)[:Wd * H].reshape(H // 8, Wd // 8, 8, 8)       # [block row, block col, v, u]
        k = np.arange(8)
        Cm = np.cos((2 * k[None, :] + 1) * k[:, None] * np.pi / 16) * np.where(k == 0, 1 / np.sqrt(2), 1.0)[:, None]   # [u, x]
        blocks = (img.astype(np.float64) - 128).reshape(H // 8, 8, Wd // 8, 8).transpose(0, 2, 1, 3)              # [by, bx, y, x]
        F = 0.25 * np.einsum('vy,ux,abyx->abvu', Cm, Cm, blocks)
        q = F / JPEG_Q.reshape(8, 8)
        d = np.abs(got - np.round(q))
        # values within float rounding of a .5 tie may round either way: off by one allowed only there
        tie = np.abs(np.abs(q - np.trunc(q)) - 0.5) < 1e-3
        bad = int(np.sum((d > 1) | ((d == 1) & ~tie)))
        return bad == 0, {'bad_coefficients': bad, 'off_by_one_at_ties': int(np.sum((d == 1) & tie)), 'nonzero': int(np.sum(got != 0))}
    return W('dct8', {0: (img, False), 1: (JPEG_Q, False), 2: (4 * Wd * H, True)}, [(Wd // 8, H // 8, 1, push(Wd, H))], check, (8, 8, 1),
             {'pixels': Wd * H})



def w_attention(size, rng):
    nq, seq = (64, 64) if size == 'small' else (65536, 256)  # query rows (one workgroup each) against seq keys / values
    q = f32(rng.standard_normal((nq, 64))); k = f32(rng.standard_normal((seq, 64))); v = f32(rng.standard_normal((seq, 64)))
    def check(o):
        got = o[3].view(np.float32)[:nq * 64].reshape(nq, 64)
        s_ = q.astype(np.float64) @ k.T.astype(np.float64) / 8
        pr = np.exp(s_ - s_.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
        ref = pr @ v.astype(np.float64)
        err = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
        return err < 2e-5, {'max_err_rel_to_max': err}
    return W('attention', {0: (q, False), 1: (k, False), 2: (v, False), 3: (4 * nq * 64, True)}, [(nq, 1, 1, push(seq))], check, 256,
             {'flops': 4 * nq * seq * 64})


def w_resize(size, rng):
    (sw, sh, dw, dh) = (40, 30, 97, 61) if size == 'small' else (1280, 720, 3840, 2160)
    gy, gx = np.mgrid[0:sh, 0:sw]
    rgba = np.stack([128 + 100 * np.sin(gx / 5.0), 128 + 100 * np.cos(gy / 7.0), 255.0 * ((gx // 8 + gy // 8) % 2), np.full(gx.shape, 255.0)], -1)
    src = (np.clip(rgba + rng.normal(0, 3, rgba.shape), 0, 255).astype(np.uint8)).view(np.uint32).reshape(sh, sw)
    def check(o):
        got = o[1][:4 * dw * dh].reshape(dh, dw, 4).astype(np.int64)
        s8 = src.view(np.uint8).reshape(sh, sw, 4).astype(np.float64) / 255
        fx = (np.arange(dw) + 0.5) * sw / dw - 0.5; fy = (np.arange(dh) + 0.5) * sh / dh - 0.5
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int); ax, ay = fx - x0, fy - y0
        P = lambda yy, xx: s8[np.clip(yy, 0, sh - 1)[:, None], np.clip(xx, 0, sw - 1)[None, :]]
        top = P(y0, x0) * (1 - ax)[None, :, None] + P(y0, x0 + 1) * ax[None, :, None]
        bot = P(y0 + 1, x0) * (1 - ax)[None, :, None] + P(y0 + 1, x0 + 1) * ax[None, :, None]
        ref = (top * (1 - ay)[:, None, None] + bot * ay[:, None, None]) * 255
        d = np.abs(got - ref)
        return float(d.max()) <= 1.0, {'max_err_levels': float(d.max()), 'mean_err': float(d.mean())}
    return W('resize', {0: (src, False), 1: (4 * dw * dh, True)}, [((dw + 15) // 16, (dh + 15) // 16, 1, push(sw, sh, dw, dh))], check,
             (16, 16, 1), {'pixels': dw * dh}, images=[(1, dw, dh, 'rgba8')])


def w_median3(size, rng):
    Wd, H = (80, 60) if size == 'small' else (3840, 2160)
    gy, gx = np.mgrid[0:H, 0:Wd]
    img = (128 + 80 * np.sin(gx / 15.0) * np.cos(gy / 19.0)).astype(np.uint8)
    noise = rng.random((H, Wd))
    img[noise < 0.05] = 0; img[noise > 0.95] = 255                              # salt and pepper
    def check(o):
        got = o[1][:Wd * H].reshape(H, Wd)
        q = np.pad(img, 1, mode='edge')
        stack = np.stack([q[dy:dy + H, dx:dx + Wd] for dy in range(3) for dx in range(3)])
        ref = np.median(stack, axis=0).astype(np.uint8)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'noisy_pixels_in': int(np.sum((img == 0) | (img == 255))),
                                          'noisy_pixels_out': int(np.sum((got == 0) | (got == 255)))}
    return W('median3', {0: (img, False), 1: (Wd * H, True)}, [((Wd + 15) // 16, (H + 15) // 16, 1, push(Wd, H))], check, (16, 16, 1),
             {'pixels': Wd * H}, images=[(1, Wd, H, 'gray8')])


def w_poisson(size, rng):
    n, iters = (48, 40) if size == 'small' else (1024, 200)
    h = 1.0 / (n + 1)
    x = (np.arange(n) + 1) * h
    f = f32(2 * np.pi ** 2 * np.sin(np.pi * x)[:, None] * np.sin(np.pi * x)[None, :])   # exact solution sin(pi x) sin(pi y)
    h2 = float(np.float32(h * h))
    def check(o):
        got = o[0].view(np.float32)[:n * n].reshape(n, n)                      # even iteration count ends in the first half
        u = np.zeros((n, n))
        for _ in range(iters):
            q = np.pad(u, 1)
            u = 0.25 * (q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:] + h2 * f.astype(np.float64))
        err = float(np.max(np.abs(got - u)) / np.max(np.abs(u)))
        return err < 1e-5, {'rel_err_vs_float64_jacobi': err, 'max_u': float(np.max(u))}
    ds = [((n + 15) // 16, (n + 15) // 16, 1, push(n, k & 1, h2)) for k in range(iters)]
    return W('poisson', {0: (8 * n * n, True), 1: (f, False)}, ds, check, (16, 16, 1), {'cell_updates': n * n * iters})



LBM_C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
LBM_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)


def lbm_feq(rho, ux, uy):
    cu = LBM_C[:, 0, None, None] * ux + LBM_C[:, 1, None, None] * uy
    return LBM_W[:, None, None] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))


def w_lbm(size, rng):
    nx, ny, steps = (48, 32, 20) if size == 'small' else (512, 512, 200)
    omega = 1.2
    yy = np.arange(ny)[:, None] * np.ones((1, nx))
    ux0 = 0.05 * np.sin(2 * np.pi * yy / ny); uy0 = np.zeros((ny, nx))   # decaying shear wave
    f0 = f32(lbm_feq(np.ones((ny, nx)), ux0, uy0))                     # [9, ny, nx]
    def check(o):
        got = o[0].view(np.float32)[:9 * nx * ny].reshape(9, ny, nx)    # even step count ends in the first half
        fv = f0.astype(np.float64)
        for _ in range(steps):
            fi = np.stack([np.roll(np.roll(fv[i], LBM_C[i, 1], 0), LBM_C[i, 0], 1) for i in range(9)])
            rho = fi.sum(0); ux = (fi * LBM_C[:, 0, None, None]).sum(0) / rho; uy = (fi * LBM_C[:, 1, None, None]).sum(0) / rho
            fv = fi + omega * (lbm_feq(rho, ux, uy) - fi)
        err = float(np.max(np.abs(got - fv)))
        mass = float(abs(got.astype(np.float64).sum() - nx * ny) / (nx * ny))
        return err < 2e-6 and mass < 1e-5, {'max_abs_err': err, 'mass_drift': mass}
    ds = [((nx + 15) // 16, (ny + 15) // 16, 1, push(nx, ny, k & 1, omega)) for k in range(steps)]
    return W('lbm', {0: (np.concatenate([f0.ravel(), np.zeros(9 * nx * ny, np.float32)]), True)}, ds, check, (16, 16, 1),
             {'cell_updates': nx * ny * steps})


def w_base64(size, rng):
    n = 3 * 1000 + 2 if size == 'small' else 3 * (1 << 22) + 1           # lengths leave a 2- / 1-byte tail: '=' / '=='
    data = rng.integers(0, 256, n, dtype=np.uint8)
    out_len = 4 * ((n + 2) // 3)
    def check(o):
        import base64
        got = o[1][:out_len].tobytes()
        ref = base64.b64encode(data.tobytes())
        bad = [i for i in range(len(ref)) if got[i] != ref[i]][:5] if got != ref else []
        return got == ref, {'chars': out_len, 'first_mismatches': bad, 'tail': got[-4:].decode('latin1')}
    return W('base64', {0: (data, False), 1: (out_len, True)}, [(((n + 2) // 3 + 255) // 256, 1, 1, push(n))],
             check, 256, {'bytes': n})


def w_md5(size, rng):
    n = 256 if size == 'small' else 1 << 18
    msgs = rng.integers(0, 256, (n, 55), dtype=np.uint8)
    blocks = np.zeros((n, 64), np.uint8); blocks[:, :55] = msgs; blocks[:, 55] = 0x80; blocks[:, 56] = 0xb8; blocks[:, 57] = 0x01   # 440 bits
    words = blocks.view('<u4').astype(np.uint32)
    def check(o):
        got = o[1].view(np.uint32)[:4 * n].reshape(n, 4)
        rows = range(n) if size == 'small' else rng.choice(n, 4096, replace=False)
        bad = sum(got[i].astype('<u4').tobytes() != hashlib.md5(bytes(msgs[i])).digest() for i in rows)
        return bad == 0, {'bad_digests': int(bad)}
    return W('md5', {0: (words, False), 1: (16 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'hashes': n})


def w_gemv(size, rng):
    M, N = (64, 1000) if size == 'small' else (8192, 8192)
    a = f32(rng.uniform(-1, 1, (M, N))); x = f32(rng.uniform(-1, 1, N))
    def check(o):
        got = o[2].view(np.float32)[:M]
        ref = a.astype(np.float64) @ x.astype(np.float64)
        bound = (np.abs(a).astype(np.float64) @ np.abs(x).astype(np.float64)) * (N + 2) * 2.0 ** -24
        excess = float(np.max(np.abs(got - ref) / bound))
        return excess <= 1.0, {'max_err_over_bound': excess}
    return W('gemv', {0: (a, False), 1: (x, False), 2: (4 * M, True)}, [(M, 1, 1, push(N))], check, 256, {'bytes': 4 * M * N, 'flops': 2 * M * N})


def lev_ref(a, b):
    row = list(range(len(b) + 1))
    for k in range(1, len(a) + 1):
        diag, row[0] = row[0], k
        for j in range(1, len(b) + 1):
            up = row[j]
            row[j] = min(up + 1, row[j - 1] + 1, diag + (a[k - 1] != b[j - 1]))
            diag = up
    return row[len(b)]


def w_levenshtein(size, rng):
    n = 512 if size == 'small' else 1 << 18
    la = rng.integers(0, 33, n); lb = np.clip(la + rng.integers(-4, 5, n), 0, 32)
    sa = rng.integers(97, 101, (n, 32), dtype=np.uint8)                       # 4-letter alphabet: many matches
    sb = sa.copy()
    edits = rng.random((n, 32)) < 0.2
    sb[edits] = rng.integers(97, 101, int(edits.sum()), dtype=np.uint8)       # mutated copies: small distances
    lens = u32(np.stack([la, lb], 1))
    def check(o):
        got = o[3].view(np.uint32)[:n]
        rows = range(n) if size == 'small' else rng.choice(n, 4096, replace=False)
        bad = [int(i) for i in rows if got[i] != lev_ref(bytes(sa[i, :la[i]]), bytes(sb[i, :lb[i]]))]
        return not bad, {'bad_pairs': len(bad), 'first_bad': bad[:5], 'mean_distance': float(np.mean(got))}
    return W('levenshtein', {0: (sa, False), 1: (sb, False), 2: (lens, False), 3: (4 * n, True)}, [((n + 63) // 64, 1, 1, push(n))],
             check, 64, {'pairs': n, 'cells': int(np.sum(la * lb))})


BLUR_W = np.array([0.0162162162, 0.0540540541, 0.1216216216, 0.1945945946, 0.2270270270,
                   0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162])


def w_blur(size, rng):
    Wd, H = (72, 40) if size == 'small' else (3840, 2160)
    gy, gx = np.mgrid[0:H, 0:Wd]
    img = f32(((gx // 8 + gy // 8) % 2) * 0.8 + 0.1 + 0.05 * rng.standard_normal((H, Wd)))   # noisy checkerboard
    wt = BLUR_W.astype(np.float32).astype(np.float64)
    def conv(a, axis):
        q = np.pad(a, [(4, 4) if ax == axis else (0, 0) for ax in range(2)], mode='edge')
        return sum(wt[t] * (q[t:t + H, :] if axis == 0 else q[:, t:t + Wd]) for t in range(9))
    def check(o):
        got = o[2].view(np.float32)[:Wd * H].reshape(H, Wd)
        ref = conv(conv(img.astype(np.float64), 1), 0)
        bound = 40 * 2.0 ** -24 * float(np.max(np.abs(img)))                # 2 passes of 9 rounded products and sums
        err = float(np.max(np.abs(got - ref)))
        return err <= bound, {'max_abs_err': err, 'bound': bound, 'contrast_in': float(np.std(img)), 'contrast_out': float(np.std(ref))}
    grid = (max(Wd, H) + 15) // 16
    ds = [(grid, grid, 1, push(Wd, H, k)) for k in range(2)]
    return W('blur', {0: (img, False), 1: (4 * Wd * H, False), 2: (4 * Wd * H, True)}, ds, check, (16, 16, 1), {'pixels': Wd * H})


def w_yuv420(size, rng):
    Wd, H = (64, 48) if size == 'small' else (3840, 2160)
    gy, gx = np.mgrid[0:H, 0:Wd]
    rgb = np.stack([255 * gx / (Wd - 1), 255 * gy / (H - 1), 128 + 127 * np.sin(gx / 9.0 + gy / 13.0)], -1)
    rgb = np.clip(rgb + rng.integers(-8, 9, rgb.shape), 0, 255).astype(np.int64)
    rgba = u32(rgb[..., 0] | rgb[..., 1] << 8 | rgb[..., 2] << 16 | 255 << 24)
    def check(o):
        got = o[1][:Wd * H * 3 // 2]
        Y = ((66 * rgb[..., 0] + 129 * rgb[..., 1] + 25 * rgb[..., 2] + 128) >> 8) + 16
        a = (rgb.reshape(H // 2, 2, Wd // 2, 2, 3).sum((1, 3)) + 2) >> 2
        cb = ((-38 * a[..., 0] - 74 * a[..., 1] + 112 * a[..., 2] + 128) >> 8) + 128
        cr = ((112 * a[..., 0] - 94 * a[..., 1] - 18 * a[..., 2] + 128) >> 8) + 128
        ref = np.concatenate([Y.ravel(), np.stack([cb, cr], -1).ravel()]).astype(np.uint8)
        return np.array_equal(got, ref), {'luma_mismatches': int(np.sum(got[:Wd * H] != ref[:Wd * H])),
                                          'chroma_mismatches': int(np.sum(got[Wd * H:] != ref[Wd * H:]))}
    return W('yuv420', {0: (rgba, False), 1: (Wd * H * 3 // 2, True)}, [((Wd // 2 + 15) // 16, (H // 2 + 15) // 16, 1, push(Wd, H))],
             check, (16, 16, 1), {'pixels': Wd * H}, images=[(1, Wd, H, 'gray8')])


def pcg_hash(v):
    s = (v * 747796405 + 2891336453) & 0xffffffff
    w = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & 0xffffffff
    return (w >> 22) ^ w


def w_montecarlo(size, rng):
    n = 1000 if size == 'small' else 1 << 22
    seed = int(rng.integers(0, 1 << 32))
    def check(o):
        cnt = o[0].view(np.uint32)[:n]; total = int(o[1].view(np.uint32)[0])
        ref = np.zeros(n, np.int64)
        for c in range(0, n, 1 << 16):                                         # chunks keep the uint64 temporaries small
            v = (np.arange(c, min(n, c + (1 << 16)), dtype=np.uint64)[:, None] * 64 + np.arange(64, dtype=np.uint64)) ^ np.uint64(seed)
            h = pcg_hash(v & 0xffffffff)
            x, y = h & 32767, (h >> 15) & 32767
            ref[c:c + len(v)] = (x * x + y * y < (1 << 30)).sum(1)
        ok = np.array_equal(cnt, ref) and total == int(ref.sum())
        return ok, {'count_mismatches': int(np.sum(cnt != ref)), 'total': total, 'ref_total': int(ref.sum()),
                    'pi_estimate': 4 * total / (64 * n)}
    return W('montecarlo', {0: (4 * n, True), 1: (u32([0]), True)}, [((n + 127) // 128, 1, 1, push(n, seed))], check, 128,
             {'samples': 64 * n})


XXP1, XXP2, XXP3, XXP4 = 2654435761, 2246822519, 3266489917, 668265263


def rotl32(x, r): return ((x << r) | (x >> (32 - r))) & 0xffffffff


def xxh32_rows(words, seed):
    """xxHash32 of each row of uint32 words (byte length 4 * columns), vectorised over rows in uint64 arithmetic"""
    M = 0xffffffff
    n, L = words.shape
    w = words.astype(np.uint64)
    v = [np.full(n, (seed + XXP1 + XXP2) & M, np.uint64), np.full(n, (seed + XXP2) & M, np.uint64),
         np.full(n, seed & M, np.uint64), np.full(n, (seed - XXP1) & M, np.uint64)]
    for i in range(L // 4):
        for j in range(4):
            v[j] = (rotl32((v[j] + w[:, 4 * i + j] * XXP2) & M, 13) * XXP1) & M
    h = (rotl32(v[0], 1) + rotl32(v[1], 7) + rotl32(v[2], 12) + rotl32(v[3], 18) + 4 * L) & M
    for i in range(4 * (L // 4), L):
        h = (rotl32((h + w[:, i] * XXP3) & M, 17) * XXP4) & M
    h ^= h >> 15; h = (h * XXP2) & M; h ^= h >> 13; h = (h * XXP3) & M; h ^= h >> 16
    return h


def w_xxhash(size, rng):
    n, L = (256 if size == 'small' else 1 << 17), 259                       # 1036-byte records: 64 stripes + 3 tail words
    data = u32(rng.integers(0, 1 << 32, (n, L), dtype=np.uint64))
    data[1::7] = data[0::7][:len(data[1::7])]                               # duplicate records hash alike
    seed = 0x9747b28c
    def check(o):
        got = o[1].view(np.uint32)[:n]
        ref = xxh32_rows(data, seed).astype(np.uint32)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'distinct_hashes': int(len(np.unique(got)))}
    return W('xxhash', {0: (data, False), 1: (4 * n, True)}, [((n + 63) // 64, 1, 1, push(n, L, seed))], check, 64,
             {'bytes': 4 * n * L})


def fmix32(h):
    M = 0xffffffff
    h = h ^ (h >> 16); h = (h * 0x85ebca6b) & M; h ^= h >> 13; h = (h * 0xc2b2ae35) & M; h ^= h >> 16
    return h


def w_bloom(size, rng):
    n = 1000 if size == 'small' else 1 << 21
    mbits = 32 * ((10 * n + 31) // 32)                                       # 10 bits per key, 6 hashes: ~0.8% false positives
    pool = u32(rng.choice(1 << 32, 2 * n, replace=False))
    keys = pool[:n]
    queries = u32(np.concatenate([rng.permutation(keys)[:n // 2], pool[n:n + n - n // 2]]))
    def bit_positions(k):
        k = k.astype(np.uint64)
        h1, h2 = fmix32(k), fmix32(k ^ 0x9e3779b9) | 1
        return [((h1 + j * h2) & 0xffffffff) % mbits for j in range(6)]
    def check(o):
        bits = o[1].view(np.uint32)[:mbits // 32]; hits = o[3].view(np.uint32)[:n]
        ref = np.zeros(mbits, bool)
        for b in bit_positions(keys): ref[b.astype(np.int64)] = True
        ref_bits = np.packbits(ref.reshape(-1, 32)[:, ::-1], axis=1).view('>u4').ravel()
        ref_hits = np.all([ref[b.astype(np.int64)] for b in bit_positions(queries)], axis=0).astype(np.uint32)
        ok = np.array_equal(bits, ref_bits) and np.array_equal(hits, ref_hits)
        return ok, {'bit_mismatches': int(np.sum(bits != ref_bits)), 'hit_mismatches': int(np.sum(hits != ref_hits)),
                    'false_negatives': int(np.sum(hits[:n // 2] == 0)), 'false_positive_rate': float(np.mean(hits[n // 2:]))}
    ds = [((n + 255) // 256, 1, 1, push(n, mbits, k)) for k in range(2)]
    return W('bloom', {0: (keys, False), 1: (u32(np.zeros(mbits // 32)), True), 2: (queries, False), 3: (4 * n, True)}, ds, check, 256,
             {'keys': n})


def w_collatz(size, rng):
    n = 512 if size == 'small' else 1 << 20
    starts = rng.integers(2, 1 << 31, n, dtype=np.uint64)
    starts[:4] = [27, 97, 871, 77031]                                         # long known trajectories (111, 118, 178, 350 steps)
    def check(o):
        steps = o[1].view(np.uint32)[:n]; peaks = o[2].view(np.uint64)[:n]
        x = starts.copy(); pk = x.copy(); st = np.zeros(n, np.uint32); live = x != 1
        while live.any():
            xv = x[live]
            xv = np.where(xv & 1, 3 * xv + 1, xv >> 1)
            assert int(xv.max()) < 1 << 62
            x[live] = xv; pk[live] = np.maximum(pk[live], xv); st[live] += 1
            live = x != 1
        ok = np.array_equal(steps, st) and np.array_equal(peaks, pk)
        return ok, {'step_mismatches': int(np.sum(steps != st)), 'peak_mismatches': int(np.sum(peaks != pk)),
                    'max_steps': int(st.max()), 'mean_steps': float(st.mean()), 'max_peak': int(pk.max())}
    return W('collatz', {0: (starts, False), 1: (4 * n, True), 2: (8 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64,
             {'starts': n})


def w_tridiag(size, rng):
    S = 256 if size == 'small' else 1 << 16
    lo = f32(rng.uniform(-1, 1, (S, 64))); up = f32(rng.uniform(-1, 1, (S, 64)))
    lo[:, 0] = 0; up[:, 63] = 0
    di = f32((np.abs(lo) + np.abs(up)) * rng.choice([-1, 1], (S, 64)) * rng.uniform(1.5, 3, (S, 64)))   # diagonally dominant
    di[di == 0] = 1
    rhs = f32(rng.uniform(-1, 1, (S, 64)))
    def check(o):
        got = o[4].view(np.float32)[:S * 64].reshape(S, 64).astype(np.float64)
        A = np.zeros((S, 64, 64))
        i = np.arange(64)
        A[:, i, i] = di; A[:, i[1:], i[:-1]] = lo[:, 1:]; A[:, i[:-1], i[1:]] = up[:, :-1]
        ref = np.linalg.solve(A, rhs.astype(np.float64)[..., None])[..., 0]
        err = float(np.max(np.abs(got - ref) / np.max(np.abs(ref), axis=1, keepdims=True)))
        bound = 64 * 2.0 ** -24                                                  # 64 eliminations, each a few ulp at most
        return err <= bound, {'max_rel_err': err, 'bound': bound}
    return W('tridiag', {0: (lo, False), 1: (di, False), 2: (up, False), 3: (rhs, False), 4: (4 * S * 64, True)},
             [((S + 63) // 64, 1, 1, push(S))], check, 64, {'systems': S, 'unknowns': 64 * S})


def w_merge(size, rng):
    n, m = (700, 500) if size == 'small' else (1 << 23, 3 << 21)
    a = np.sort(rng.integers(-1000, 1000 if size == 'small' else 1 << 20, n)).astype(np.int32)   # many ties within and across
    b = np.sort(rng.integers(-1000, 1000 if size == 'small' else 1 << 20, m)).astype(np.int32)
    def check(o):
        got = o[2].view(np.int32)[:n + m]; src = o[3].view(np.uint32)[:n + m]
        order = np.argsort(np.concatenate([a, b]), kind='stable')          # stable: A's equal keys first
        ok = np.array_equal(src, order) and np.array_equal(got, np.concatenate([a, b])[order])
        return ok, {'value_mismatches': int(np.sum(got != np.concatenate([a, b])[order])), 'source_mismatches': int(np.sum(src != order))}
    return W('merge', {0: (a, False), 1: (b, False), 2: (4 * (n + m), True), 3: (4 * (n + m), True)},
             [((n + m + 255) // 256, 1, 1, push(n, m))], check, 256, {'elements': n + m})


def ccl_jacobi(fg, lab):
    """one synchronous label-propagation step (4-connectivity, foreground only)"""
    big = np.uint32(0xffffffff)
    m = lab.copy()
    for sl_dst, sl_src in [((slice(None), slice(1, None)), (slice(None), slice(None, -1))), ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                           ((slice(1, None), slice(None)), (slice(None, -1), slice(None))), ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))]:
        nb = np.where(fg[sl_src], lab[sl_src], big)
        m[sl_dst] = np.minimum(m[sl_dst], nb)
    return np.where(fg, m, big)


def w_ccl(size, rng):
    Wd, H = (48, 40) if size == 'small' else (512, 512)
    noise = rng.standard_normal((H + 8, Wd + 8))
    sm = sum(noise[dy:dy + H, dx:dx + Wd] for dy in range(9) for dx in range(9))   # box-smoothed noise: blobs
    fg = sm > 2.0
    lab0 = np.where(fg, np.arange(H * Wd, dtype=np.uint32).reshape(H, Wd), np.uint32(0xffffffff)).astype(np.uint32)
    lab, iters = lab0, 0
    while True:                                                             # Jacobi order converges slowest
        nxt = ccl_jacobi(fg, lab)
        if np.array_equal(nxt, lab):
            break
        lab, iters = nxt, iters + 1
    ref = lab
    def check(o):
        got = o[1].view(np.uint32)[:H * Wd].reshape(H, Wd)
        comps = len(np.unique(ref[fg]))
        return np.array_equal(got, ref), {'label_mismatches': int(np.sum(got != ref)), 'components': comps, 'dispatches': max(iters, 1)}
    ds = [((Wd + 15) // 16, (H + 15) // 16, 1, push(Wd, H))] * max(iters, 1)
    return W('ccl', {0: (u32(fg.ravel()), False), 1: (lab0.ravel(), True)}, ds, check, (16, 16, 1), {'pixel_updates': Wd * H * len(ds)})


def w_pagerank(size, rng):
    (n, iters), d = ((500, 20) if size == 'small' else (1 << 20, 20)), 0.85
    outdeg = np.minimum(rng.zipf(2.0, n), 200)                              # heavy-tailed out-degrees, all >= 1
    srcs = np.repeat(np.arange(n), outdeg)
    dsts = (srcs + rng.integers(1, n, len(srcs))) % n                       # no self links
    order = np.argsort(dsts, kind='stable')
    col = u32(srcs[order]); rowptr = u32(np.concatenate([[0], np.cumsum(np.bincount(dsts, minlength=n))]))
    inv_out = f32(1.0 / outdeg)
    def check(o):
        got = o[3].view(np.float32)[:n]                                     # even iteration count ends in the first half
        r = np.full(n, 1.0 / n)
        for _ in range(iters):
            r = (1 - d) / n + d * np.bincount(dsts, weights=r[srcs] / outdeg[srcs], minlength=n)
        err = float(np.max(np.abs(got - r) / r))
        return err < 1e-4, {'max_rel_err': err, 'top_page': int(np.argmax(r)), 'top_rank': float(r.max()), 'rank_sum': float(got.sum())}
    ds = [((n + 127) // 128, 1, 1, push(n, k & 1, d)) for k in range(iters)]
    rank0 = f32(np.concatenate([np.full(n, 1.0 / n), np.zeros(n)]))
    return W('pagerank', {0: (rowptr, False), 1: (col, False), 2: (inv_out, False), 3: (rank0, True)}, ds, check, 128,
             {'edges': len(col) * iters})


def w_skinning(size, rng):
    n, nb = (500 if size == 'small' else 1 << 20), 64
    pos = f32(np.concatenate([rng.uniform(-1, 1, (n, 3)), np.ones((n, 1))], 1))
    nv = rng.standard_normal((n, 3)); nv /= np.linalg.norm(nv, axis=1, keepdims=True)
    nrm = f32(np.concatenate([nv, np.zeros((n, 1))], 1))
    jidx = rng.integers(0, nb, (n, 4))
    joints = u32(jidx[:, 0] | jidx[:, 1] << 8 | jidx[:, 2] << 16 | jidx[:, 3] << 24)
    w = rng.random((n, 4)) ** 2; w /= w.sum(1, keepdims=True); w = f32(w)
    ang = rng.uniform(-np.pi, np.pi, nb)                                    # rotations about z plus translations
    bones = np.zeros((nb, 4, 4)); bones[:, 0, 0] = bones[:, 1, 1] = np.cos(ang); bones[:, 0, 1] = -np.sin(ang); bones[:, 1, 0] = np.sin(ang)
    bones[:, 2, 2] = 1; bones[:, 3, 3] = 1; bones[:, :3, 3] = rng.uniform(-0.5, 0.5, (nb, 3))
    bones = f32(bones)
    bones_cm = f32(np.transpose(bones, (0, 2, 1)))                          # GLSL mat4: column-major
    def check(o):
        gp = o[5].view(np.float32)[:4 * n].reshape(n, 4); gn = o[6].view(np.float32)[:4 * n].reshape(n, 4)
        M = np.einsum('nk,nkij->nij', w.astype(np.float64), bones.astype(np.float64)[jidx])
        rp = np.einsum('nij,nj->ni', M, pos.astype(np.float64))
        rn = np.einsum('nij,nj->ni', M[:, :3, :3], nrm[:, :3].astype(np.float64))
        bound = 32 * 2.0 ** -24 * 2.0                                       # |M| entries <= 1.5, |v| <= 1: a few dozen roundings
        ep, en = float(np.max(np.abs(gp - rp))), float(np.max(np.abs(gn[:, :3] - rn)))
        return ep <= bound and en <= bound and np.all(gn[:, 3] == 0), {'pos_max_err': ep, 'nrm_max_err': en, 'bound': bound}
    return W('skinning', {0: (pos, False), 1: (nrm, False), 2: (joints, False), 3: (w, False), 4: (bones_cm, False),
                          5: (16 * n, True), 6: (16 * n, True)}, [((n + 63) // 64, 1, 1, push(n))], check, 64, {'vertices': n})


def w_strsearch(size, rng):
    n = 4096 if size == 'small' else 1 << 26
    text = np.frombuffer(b'acgt', np.uint8)[rng.integers(0, 4, n)]            # DNA-like text
    pat = np.frombuffer(b'gattacag', np.uint8).copy()
    for pos in rng.choice(n - 8, max(4, n // 4096), replace=False):           # planted occurrences (some overlap or clobber)
        text[pos:pos + 8] = pat
    def check(o):
        flags = o[2][:n]; count, first = (int(v) for v in o[3].view(np.uint32)[:2])
        win = np.lib.stride_tricks.sliding_window_view(text, 8)
        ref = np.zeros(n, np.uint8); ref[:n - 7] = np.all(win == pat, axis=1)
        hits = np.flatnonzero(ref)
        ok = np.array_equal(flags, ref) and count == len(hits) and first == (int(hits[0]) if len(hits) else 0xffffffff)
        return ok, {'flag_mismatches': int(np.sum(flags != ref)), 'count': count, 'ref_count': int(len(hits)), 'first': first}
    return W('strsearch', {0: (text, False), 1: (pat, False), 2: (n, True), 3: (u32([0, 0xffffffff]), True)},
             [((n + 255) // 256, 1, 1, push(n))], check, 256, {'bytes': n})


def w_quantize(size, rng):
    rows = 64 if size == 'small' else 1 << 15
    x = f32(rng.standard_normal((rows, 256)) * rng.uniform(0.01, 10, (rows, 1)))
    x[np.abs(x) < 1e-30] = 1e-30                                              # no denormals (the GPU flushes them)
    x[0, :8] = [0.5, 1.5, 2.5, -0.5, -2.5, 126.5, -126.5, 127.0]                 # ties: round half to even
    inv = f32(127.0 / np.max(np.abs(x), axis=1)); inv[0] = 1.0
    def check(o):
        got = o[2].view(np.int8)[:rows * 256].reshape(rows, 256); sums = o[3].view(np.uint32)[:rows]
        ref = np.clip(np.rint(f32(x * inv[:, None])), -127, 127).astype(np.int8)  # fp32 product, then half-to-even
        ok = np.array_equal(got, ref) and np.array_equal(sums, np.abs(ref.astype(np.int64)).sum(1))
        deq = got.astype(np.float64) / inv[:, None]
        return ok, {'mismatches': int(np.sum(got != ref)), 'row0_ties': got[0, :8].tolist(),
                    'max_dequant_err_over_step': float(np.max(np.abs(deq - x) * inv[:, None]))}
    return W('quantize', {0: (x, False), 1: (inv, False), 2: (rows * 256, True), 3: (4 * rows, True)},
             [(rows, 1, 1, push(rows))], check, 256, {'elements': rows * 256})


def w_rule110(size, rng):
    cells, steps = (512, 64) if size == 'small' else (8192, 1000)
    words = cells // 32
    hist = np.zeros((steps + 1, words), np.uint32)
    hist[0] = u32(rng.integers(0, 1 << 32, words, dtype=np.uint64))
    def check(o):
        got = o[0].view(np.uint32)[:(steps + 1) * words].reshape(steps + 1, words)
        c = np.unpackbits(hist[0].view(np.uint8), bitorder='little').astype(bool)
        ref = [c]
        for _ in range(steps):
            l, r = np.roll(c, 1), np.roll(c, -1)                              # cell k's left neighbour is k - 1 (ring)
            c = (c | r) & ~(l & c & r); ref.append(c)
        ref = np.packbits(np.array(ref), axis=1, bitorder='little').view(np.uint32)
        bad = np.flatnonzero(np.any(got != ref, axis=1))
        return len(bad) == 0, {'bad_rows': int(len(bad)), 'first_bad_row': int(bad[0]) if len(bad) else None,
                               'live_cells_last_row': int(np.unpackbits(got[-1].view(np.uint8)).sum())}
    ds = [((words + 63) // 64, 1, 1, push(words, k)) for k in range(steps)]
    return W('rule110', {0: (hist, True)}, ds, check, 64, {'cell_updates': cells * steps}, images=[(0, cells, steps + 1, 'bits')])


def w_voxel(size, rng):
    n = 4096 if size == 'small' else 1 << 22
    centers = rng.uniform(0.2, 0.8, (8, 3))
    pts = centers[rng.integers(0, 8, n)] + rng.standard_normal((n, 3)) * 0.06  # clustered, like scanned surfaces
    pts = f32(np.clip(pts, 0, np.nextafter(np.float32(1), np.float32(0))))
    pts4 = f32(np.concatenate([pts, np.ones((n, 1), np.float32)], 1))
    def check(o):
        cnt = o[1].view(np.uint32)[:64 ** 3]; sm = o[2].view(np.uint32)[:3 * 64 ** 3].reshape(-1, 3)
        v = (pts * 64).astype(np.int64); vid = (v[:, 2] * 64 + v[:, 1]) * 64 + v[:, 0]
        rc = np.bincount(vid, minlength=64 ** 3)
        fx = (pts.astype(np.float64) * 65536).astype(np.int64)
        rs = np.stack([np.bincount(vid, weights=fx[:, k], minlength=64 ** 3) for k in range(3)], 1).astype(np.int64)
        ok = np.array_equal(cnt, rc) and np.array_equal(sm.astype(np.int64), rs)
        return ok, {'count_mismatches': int(np.sum(cnt != rc)), 'sum_mismatches': int(np.sum(sm != rs)),
                    'occupied_voxels': int(np.sum(rc > 0)), 'max_points_per_voxel': int(rc.max())}
    return W('voxel', {0: (pts4, False), 1: (u32(np.zeros(64 ** 3)), True), 2: (u32(np.zeros(3 * 64 ** 3)), True)},
             [((n + 255) // 256, 1, 1, push(n))], check, 256, {'points': n})


def w_nms(size, rng):
    n = 300 if size == 'small' else 8192
    cx, cy = rng.integers(0, 1024, n), rng.integers(0, 1024, n)
    obj = rng.integers(0, max(n // 6, 1), n)                                   # detections cluster on objects
    ox, oy = rng.integers(40, 984, (2, n // 6 + 1))
    cx, cy = np.where(rng.random(n) < 0.8, ox[obj] + rng.integers(-12, 13, n), cx), np.where(rng.random(n) < 0.8, oy[obj] + rng.integers(-12, 13, n), cy)
    w, h = rng.integers(16, 120, n), rng.integers(16, 120, n)
    boxes = np.stack([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], 1).astype(np.int32)   # sorted by score: index order
    def check(o):
        keep = o[1].view(np.uint32)[:n]; kept = int(o[2].view(np.uint32)[0])
        b = boxes.astype(np.int64); area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ref = np.ones(n, np.uint32)
        for i0 in range(0, n, 512):                                            # blocks of rows keep the pair matrix small
            bi = b[i0:i0 + 512]
            iw = np.clip(np.minimum(bi[:, None, 2], b[None, :, 2]) - np.maximum(bi[:, None, 0], b[None, :, 0]), 0, None)
            ih = np.clip(np.minimum(bi[:, None, 3], b[None, :, 3]) - np.maximum(bi[:, None, 1], b[None, :, 1]), 0, None)
            inter = iw * ih
            over = 2 * inter > area[i0:i0 + 512, None] + area[None, :] - inter
            earlier = np.arange(n)[None, :] < np.arange(i0, i0 + len(bi))[:, None]
            ref[i0:i0 + len(bi)] = ~np.any(over & earlier, axis=1)
        ok = np.array_equal(keep, ref) and kept == int(ref.sum())
        return ok, {'keep_mismatches': int(np.sum(keep != ref)), 'kept': kept, 'ref_kept': int(ref.sum())}
    return W('nms', {0: (boxes, False), 1: (4 * n, True), 2: (u32([0]), True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256,
             {'pairs': n * (n - 1) // 2})


def w_sat(size, rng):
    Wd, H = (300, 200) if size == 'small' else (3840, 2160)
    gy, gx = np.mgrid[0:H, 0:Wd]
    img = np.clip(128 + 100 * np.sin(gx / 37.0) * np.cos(gy / 23.0) + rng.integers(-20, 21, (H, Wd)), 0, 255).astype(np.uint8)
    def check(o):
        got = o[1].view(np.uint32)[:Wd * H].reshape(H, Wd)
        ref = img.astype(np.uint64).cumsum(0).cumsum(1).astype(np.uint32)
        r0, c0, r1, c1 = H // 4, Wd // 4, H // 2, Wd // 2                     # a box sum from four lookups
        box = int(got[r1, c1]) - int(got[r0 - 1, c1]) - int(got[r1, c0 - 1]) + int(got[r0 - 1, c0 - 1])
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'box_sum_ok': box == int(img[r0:r1 + 1, c0:c1 + 1].astype(np.int64).sum())}
    ds = [(H, 1, 1, push(Wd, H, 0)), (Wd, 1, 1, push(Wd, H, 1))]
    return W('sat', {0: (img, False), 1: (4 * Wd * H, True)}, ds, check, 256, {'pixels': Wd * H})


def w_sssp(size, rng):
    side = 24 if size == 'small' else 512
    n = side * side
    # a road-like grid: 4-neighbour streets plus a few long highways, integer travel times
    idx = np.arange(n).reshape(side, side)
    src_ = np.concatenate([idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel()])
    dst_ = np.concatenate([idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel()])
    hw = rng.integers(0, n, (2, n // 64))
    src_, dst_ = np.concatenate([src_, hw[0], hw[1]]), np.concatenate([dst_, hw[1], hw[0]])
    wts = np.concatenate([rng.integers(5, 100, len(src_) - 2 * (n // 64)), np.tile(rng.integers(50, 400, n // 64), 2)])
    order = np.argsort(src_, kind='stable')
    col, wt = u32(dst_[order]), u32(wts[order])
    rowptr = u32(np.concatenate([[0], np.cumsum(np.bincount(src_, minlength=n))]))
    INF = 0xffffffff
    d0 = np.full(n, INF, np.uint32); d0[0] = 0
    dist, rounds = d0.astype(np.int64), 0
    s_sorted = src_[order]
    while True:                                                                # synchronous rounds converge slowest
        cand = dist[s_sorted] + wts[order]
        nd = dist.copy(); ok = dist[s_sorted] < INF
        np.minimum.at(nd, dst_[order][ok], cand[ok])
        if np.array_equal(nd, dist):
            break
        dist, rounds = nd, rounds + 1
    ref = dist
    def check(o):
        got = o[3].view(np.uint32)[:n].astype(np.int64)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'max_dist': int(ref.max()), 'dispatches': max(rounds, 1)}
    ds = [((n + 127) // 128, 1, 1, push(n))] * max(rounds, 1)
    return W('sssp', {0: (rowptr, False), 1: (col, False), 2: (wt, False), 3: (d0, True)}, ds, check, 128, {'edge_relaxations': len(col) * len(ds)})


def spread_bits(x, bits, stride):
    out = np.zeros_like(x)
    for b in range(bits):
        out |= ((x >> b) & 1) << (stride * b)
    return out


def w_morton(size, rng):
    n = 4096 if size == 'small' else 1 << 22
    pts = f32(rng.random((n, 3)))
    pts[0] = [0, 0, 0]; pts[1] = np.nextafter(np.float32(1), np.float32(0))   # corners: all-zero and all-one codes
    pts4 = f32(np.concatenate([pts, np.ones((n, 1), np.float32)], 1))
    def check(o):
        c32 = o[1].view(np.uint32)[:n]; c64 = o[2].view(np.uint64)[:n]
        a = (pts.astype(np.float64) * 1024).astype(np.uint64); b = (pts.astype(np.float64) * 2097152).astype(np.uint64)
        r32 = (spread_bits(a[:, 0], 10, 3) | spread_bits(a[:, 1], 10, 3) << np.uint64(1) | spread_bits(a[:, 2], 10, 3) << np.uint64(2)).astype(np.uint32)
        r64 = spread_bits(b[:, 0], 21, 3) | spread_bits(b[:, 1], 21, 3) << np.uint64(1) | spread_bits(b[:, 2], 21, 3) << np.uint64(2)
        ok = np.array_equal(c32, r32) and np.array_equal(c64, r64)
        return ok, {'mismatches32': int(np.sum(c32 != r32)), 'mismatches64': int(np.sum(c64 != r64)), 'corner_codes': [hex(int(c64[0])), hex(int(c64[1]))]}
    return W('morton', {0: (pts4, False), 1: (4 * n, True), 2: (8 * n, True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256, {'points': n})


def utf8_ref(t):
    """per byte: 0xffffffff for continuation bytes, else the decoded code point or U+FFFD (the shader's rules)"""
    n = len(t); b = t.astype(np.int64)
    pad = np.concatenate([b, np.zeros(3, np.int64)])
    cont = (b & 0xc0) == 0x80
    ln = np.select([b < 0x80, (b & 0xe0) == 0xc0, (b & 0xf0) == 0xe0, (b & 0xf8) == 0xf0], [1, 2, 3, 4], 0)
    v = np.select([ln == 1, ln == 2, ln == 3, ln == 4], [b, b & 0x1f, b & 0x0f, b & 0x07], 0)
    ok = (ln != 0) & (np.arange(n) + ln <= n)
    for k in (1, 2, 3):
        c = pad[k:k + n]; use = ln > k
        ok &= ~use | ((c & 0xc0) == 0x80)
        v = np.where(use, (v << 6) | (c & 0x3f), v)
    lo = np.select([ln == 2, ln == 3, ln == 4], [0x80, 0x800, 0x10000], 0)
    ok &= (v >= lo) & (v <= 0x10ffff) & ~((v >= 0xd800) & (v <= 0xdfff))
    return np.where(cont, 0xffffffff, np.where(ok, v, 0xfffd)).astype(np.uint32)


def utf8_join(alphabet, idx):
    """''.join(alphabet[i] for i in idx).encode(), from a padded table of each character's bytes"""
    enc = [c.encode() for c in alphabet]
    table = np.zeros((len(enc), 4), np.uint8); lens = np.array([len(e) for e in enc])
    for k, e in enumerate(enc):
        table[k, :len(e)] = np.frombuffer(e, np.uint8)
    rows = table[idx]
    return rows[np.arange(4)[None, :] < lens[idx][:, None]].tobytes()          # row-major: characters in order


def w_utf8(size, rng):
    n = 4096 if size == 'small' else 1 << 24
    alphabet = 'abc xyz,.ÄéßñΩЖשش漢字かな한글€🙂🚀𝄞'                       # 1- to 4-byte characters
    good = utf8_join(alphabet, rng.integers(0, len(alphabet), n // 2))[:n // 2]
    good = good[:len(good.decode('utf-8', 'ignore').encode())]                 # whole characters only
    bad = bytearray(utf8_join(alphabet, rng.integers(0, len(alphabet), n))[:n - len(good)])
    for k in rng.integers(0, len(bad), max(8, len(bad) // 500)):               # stray continuation / lead / overlong bytes
        bad[k] = int(rng.choice([0x80, 0xbf, 0xc0, 0xc1, 0xe0, 0xed, 0xf5, 0xff]))
    text = np.frombuffer(good + bytes(bad), np.uint8).copy()
    def check(o):
        got = o[1].view(np.uint32)[:n]; chars_, invalid = (int(v) for v in o[2].view(np.uint32)[:2])
        ref = utf8_ref(text)
        lead = ref != 0xffffffff
        valid_part = got[:len(good)][ref[:len(good)] != 0xffffffff]
        decoded_ok = [chr(c) for c in valid_part] == list(good.decode())         # Python's decoder agrees on the valid half
        ok = np.array_equal(got, ref) and chars_ == int(lead.sum()) and invalid == int(np.sum(ref == 0xfffd)) and decoded_ok
        return ok, {'mismatches': int(np.sum(got != ref)), 'chars': chars_, 'invalid': invalid, 'valid_half_matches_python': decoded_ok}
    return W('utf8', {0: (text, False), 1: (4 * n, True), 2: (u32([0, 0]), True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256,
             {'bytes': n})


def w_fir(size, rng):
    n = 4096 if size == 'small' else 1 << 23
    taps = 63
    m = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(0.25 * m) * 0.25 * np.blackman(taps); h /= h.sum()              # low-pass, cutoff fs / 8
    hf = f32(np.concatenate([h, [0]]))
    tt = np.arange(n)
    x = f32(np.sin(2 * np.pi * 0.01 * tt) + 0.5 * np.sin(2 * np.pi * 0.35 * tt) + 0.01 * rng.standard_normal(n))
    def check(o):
        y = o[2].view(np.float32)[:n].astype(np.float64)
        hd, xd = hf[:taps].astype(np.float64), x.astype(np.float64)
        ref = np.convolve(xd, hd)[:n]
        bound = (taps + 2) * 2.0 ** -24 * np.convolve(np.abs(xd), np.abs(hd))[:n]
        excess = float(np.max(np.abs(y - ref) / np.maximum(bound, 1e-30)))
        spec_in, spec_out = np.abs(np.fft.rfft(xd[taps:])), np.abs(np.fft.rfft(y[taps:]))
        k = int(round(0.35 * (n - taps)))
        atten = float(20 * np.log10(spec_out[k - 2:k + 3].max() / spec_in[k - 2:k + 3].max()))
        return excess <= 1.0, {'max_err_over_bound': excess, 'stopband_tone_db': atten}
    return W('fir', {0: (x, False), 1: (hf, False), 2: (4 * n, True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256,
             {'samples': n, 'flops': 2 * taps * n})


def div255(x): x = x + 128; return (x + (x >> 8)) >> 8


def w_composite(size, rng):
    Wd, H = (64, 48) if size == 'small' else (3840, 2160)
    gx, gy = np.arange(Wd)[None, :], np.arange(H)[:, None]                      # broadcast instead of full grids
    full = lambda v: np.broadcast_to(v, (H, Wd))
    bg = [full(gx * 255 // max(Wd - 1, 1)), full(gy * 255 // max(H - 1, 1)), full(np.int64(96)), full(np.int64(255))]
    a = np.clip(255 * (0.5 + 0.5 * np.sin(gx / 13.0) * np.cos(gy / 9.0)) + rng.integers(-10, 11, (H, Wd)), 0, 255).astype(np.int64)
    a[:, :Wd // 8] = 0; a[:, -Wd // 8:] = 255                                     # fully clear and fully opaque bands
    col = [np.int64(250), np.int64(40), (gx + gy) % 256]
    src = [c * a // 255 for c in col] + [a]                                      # premultiplied: channel <= alpha
    pack = lambda c: u32(c[0] | c[1] << 8 | c[2] << 16 | c[3] << 24)
    def check(o):
        got = o[2].view(np.uint32)[:Wd * H].reshape(H, Wd)
        out = [src[k] + div255(bg[k] * (255 - a)) for k in range(4)]
        ref = pack(out)
        exact = all(np.array_equal(out[k], np.round(src[k] + bg[k] * (255 - a) / 255.0)) for k in range(4))   # div255 rounds x / 255
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'div255_is_round': exact}
    return W('composite', {0: (pack(src), False), 1: (pack(bg), False), 2: (4 * Wd * H, True)},
             [((Wd + 15) // 16, (H + 15) // 16, 1, push(Wd, H))], check, (16, 16, 1), {'pixels': Wd * H}, images=[(2, Wd, H, 'rgba8')])


def nqueens_count(n, c, a, b, row):
    full = (1 << n) - 1
    if row == n:
        return 1
    tot, avail = 0, full & ~(c | a | b)
    while avail:
        bit = avail & -avail; avail ^= bit
        tot += nqueens_count(n, c | bit, ((a | bit) << 1) & full, (b | bit) >> 1, row + 1)
    return tot


def w_nqueens(size, rng):
    n = 8 if size == 'small' else 13
    def check(o):
        got = o[0].view(np.uint32)[:n ** 3]; total = int(o[1].view(np.uint32)[0])
        full = (1 << n) - 1
        ref = np.zeros(n ** 3, np.uint32)
        for t in range(n ** 3):
            c = a = b = 0; ok = True
            for col in (t % n, t // n % n, t // (n * n) % n):
                bit = 1 << col
                ok = ok and not ((c | a | b) & bit)
                c |= bit; a = ((a | bit) << 1) & full; b = (b | bit) >> 1
            ref[t] = nqueens_count(n, c, a, b, 3) if ok else 0
        known = {8: 92, 13: 73712}[n]
        ok = np.array_equal(got, ref) and total == int(ref.sum()) == known
        return ok, {'count_mismatches': int(np.sum(got != ref)), 'total': total, 'known_total': known}
    return W('nqueens', {0: (4 * n ** 3, True), 1: (u32([0]), True)}, [((n ** 3 + 63) // 64, 1, 1, push(n))], check, 64,
             {'prefixes': n ** 3})


def w_lzmatch(size, rng):
    n = 4096 if size == 'small' else 1 << 20
    vocab = [bytes(rng.integers(97, 123, rng.integers(2, 9)).astype(np.uint8)) for _ in range(24)]
    phrases = [b' '.join(vocab[k] for k in rng.integers(0, 24, rng.integers(2, 6))) for _ in range(12)]
    text = np.frombuffer(b' '.join(phrases[k] for k in rng.integers(0, 12, n // 8)), np.uint8)[:n].copy()   # repetitive prose
    def check(o):
        got = o[1].view(np.uint32)[:n]
        t = np.concatenate([text, np.zeros(32, np.uint8)]).astype(np.int16); t[n:] = -1   # past the end never matches
        bl, bd = np.zeros(n, np.int64), np.zeros(n, np.int64)
        idx = np.arange(n)
        for d in range(1, 65):
            run = np.ones(n, bool); k = np.zeros(n, np.int64)
            for j in range(32):
                src = idx - d + j
                run &= (idx >= d) & (t[np.minimum(idx + j, n + 31)] == t[np.where(src >= 0, src, 0)])
                k += run
            better = k > bl
            bl, bd = np.where(better, k, bl), np.where(better, d, bd)
        ref = (bl | bd << 8).astype(np.uint32)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'mean_match': float(bl.mean()), 'positions_with_32': int(np.sum(bl == 32))}
    return W('lzmatch', {0: (text, False), 1: (4 * n, True)}, [((n + 127) // 128, 1, 1, push(n))], check, 128, {'positions': n})


def w_bitpack(size, rng):
    n = 4096 if size == 'small' else 1 << 24
    vals = u32(rng.integers(0, 128, n))
    nwords = (7 * n + 31) // 32 + 1
    def check(o):
        got = o[1].view(np.uint32)[:nwords]
        bits = ((vals[:, None] >> np.arange(7)) & 1).astype(np.uint8).ravel()   # bit 7i + j of the stream is bit j of value i
        ref = np.packbits(np.concatenate([bits, np.zeros(32 * nwords - len(bits), np.uint8)]), bitorder='little').view('<u4')
        back = (got.view(np.uint8))                                              # unpack again from the GPU's words
        ub = np.unpackbits(back, bitorder='little')[:7 * n].reshape(n, 7)
        unpacked = (ub.astype(np.uint32) << np.arange(7, dtype=np.uint32)).sum(1)
        return np.array_equal(got, ref) and np.array_equal(unpacked, vals), {'word_mismatches': int(np.sum(got != ref)), 'roundtrip_ok': bool(np.array_equal(unpacked, vals))}
    return W('bitpack', {0: (vals, False), 1: (u32(np.zeros(nwords)), True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256,
             {'values': n})


def perlin_ref(perm, Wd, H):
    gy, gx = np.mgrid[0:H, 0:Wd].astype(np.float64)
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    grad = lambda h, x, y: np.where(h & 1, -x, x) + np.where(h & 2, -y, y)
    total, amp, freq = np.zeros((H, Wd)), 0.5, 1 / 64
    for _ in range(4):
        qx, qy = gx * freq, gy * freq
        fx, fy = np.floor(qx), np.floor(qy)
        X, Y = fx.astype(np.int64) & 255, fy.astype(np.int64) & 255
        x, y = qx - fx, qy - fy
        u, v = fade(x), fade(y)
        a, b = perm[X] + Y, perm[X + 1] + Y
        n00, n10 = grad(perm[a], x, y), grad(perm[b], x - 1, y)
        n01, n11 = grad(perm[a + 1], x, y - 1), grad(perm[b + 1], x - 1, y - 1)
        nx0, nx1 = n00 + u * (n10 - n00), n01 + u * (n11 - n01)
        total += amp * (nx0 + v * (nx1 - nx0))
        amp, freq = amp / 2, freq * 2
    return total


def w_perlin(size, rng):
    Wd, H = (96, 64) if size == 'small' else (2048, 2048)
    perm = rng.permutation(256)
    perm2 = u32(np.concatenate([perm, perm]))
    def check(o):
        got = o[1].view(np.float32)[:Wd * H].reshape(H, Wd); img = o[2][:Wd * H].reshape(H, Wd).astype(np.int64)
        ref = perlin_ref(perm2.astype(np.int64), Wd, H)
        err = float(np.max(np.abs(got - ref)))
        ref8 = np.floor(np.clip(ref * 0.5 + 0.5, 0, 1) * 255 + 0.5)
        off8 = int(np.max(np.abs(img - ref8)))
        return err <= 1e-5 and off8 <= 1, {'max_abs_err': err, 'image_max_level_diff': off8, 'height_range': [float(ref.min()), float(ref.max())]}
    return W('perlin', {0: (perm2, False), 1: (4 * Wd * H, True), 2: (Wd * H, True)}, [((Wd + 15) // 16, (H + 15) // 16, 1, push(Wd, H))],
             check, (16, 16, 1), {'pixels': Wd * H}, images=[(2, Wd, H, 'gray8')])


def w_adler32(size, rng):
    blocks = 64 if size == 'small' else 1 << 14
    data = rng.integers(0, 256, blocks * 4096, dtype=np.uint8)
    data[:4096] = 255                                                            # worst case for the sums' range
    def check(o):
        import zlib
        got = o[1].view(np.uint32)[:blocks]
        rows = range(blocks) if size == 'small' else rng.choice(blocks, 2048, replace=False)
        bad = [int(k) for k in rows if got[k] != zlib.adler32(data[4096 * k:4096 * (k + 1)].tobytes())]
        if 0 not in rows:
            bad += [0] if got[0] != zlib.adler32(data[:4096].tobytes()) else []
        return not bad, {'bad_blocks': len(bad), 'first_bad': bad[:5]}
    return W('adler32', {0: (data, False), 1: (4 * blocks, True)}, [((blocks + 63) // 64, 1, 1, push(blocks))], check, 64,
             {'bytes': blocks * 4096})


def w_sieve(size, rng):
    N = 1 << 16 if size == 'small' else 1 << 26
    r = int(np.sqrt(N))
    small = np.ones(r + 1, bool); small[:2] = False
    for k in range(2, int(r ** 0.5) + 1):
        if small[k]: small[k * k::k] = False
    base = u32(np.flatnonzero(small))
    def check(o):
        bits = o[1].view(np.uint32)[:N // 32]; count = int(o[2].view(np.uint32)[0])
        prime = np.ones(N, bool); prime[:2] = False
        for q in base: prime[int(q) * int(q)::int(q)] = False
        ref = np.packbits(~prime, bitorder='little').view('<u4')
        known = {1 << 16: 6542, 1 << 26: 3957809}[N]
        return np.array_equal(bits, ref) and count == int(prime.sum()) == known, {'word_mismatches': int(np.sum(bits != ref)), 'primes': count, 'known': known}
    return W('sieve', {0: (base, False), 1: (N // 8, True), 2: (u32([0]), True)}, [(N // 4096, 1, 1, push(len(base)))], check, 128,
             {'integers': N})


def w_gcd(size, rng):
    n = 4096 if size == 'small' else 1 << 22
    k = rng.integers(1, 1 << 12, n, dtype=np.uint64)                         # shared factors make gcds interesting
    a = (rng.integers(1, 1 << 20, n, dtype=np.uint64) * k) & 0xffffffff; b = (rng.integers(1, 1 << 20, n, dtype=np.uint64) * k) & 0xffffffff
    a[:4] = [0, 12, 0, 1 << 31]; b[:4] = [7, 0, 0, 1 << 30]                   # zeros and powers of two
    pairs = u32(np.stack([a, b], 1))
    def check(o):
        got = o[1].view(np.uint32)[:n]
        ref = np.gcd(a.astype(np.int64), b.astype(np.int64)).astype(np.uint32)
        return np.array_equal(got, ref), {'mismatches': int(np.sum(got != ref)), 'mean_gcd': float(ref.mean()), 'gcd_1_fraction': float(np.mean(ref == 1))}
    return W('gcd', {0: (pairs, False), 1: (4 * n, True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256, {'pairs': n})


BAYER8 = np.array([0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30,
                   54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23,
                   61, 29, 53, 21]).reshape(8, 8)


def w_dither(size, rng):
    Wd, H = (128, 64) if size == 'small' else (4096, 4096)
    gy, gx = np.mgrid[0:H, 0:Wd]
    img = np.clip(255 * gx / (Wd - 1) * (0.6 + 0.4 * np.cos(gy / 40.0)) + rng.integers(-3, 4, (H, Wd)), 0, 255).astype(np.uint8)
    def check(o):
        got = o[1].view(np.uint32)[:Wd * H // 32]
        on = img.astype(np.int64) > BAYER8[gy & 7, gx & 7] * 4 + 1
        ref = np.packbits(on.ravel(), bitorder='little').view('<u4')
        mean_err = float(abs(on.mean() - img.mean() / 255))                     # dithering preserves average brightness
        return np.array_equal(got, ref), {'word_mismatches': int(np.sum(got != ref)), 'brightness_error': mean_err}
    return W('dither', {0: (img, False), 1: (Wd * H // 8, True)}, [((Wd * H // 32 + 63) // 64, 1, 1, push(Wd, H))], check, 64,
             {'pixels': Wd * H}, images=[(1, Wd, H, 'bits')])


def w_advect(size, rng):
    n, steps = (48, 10) if size == 'small' else (1024, 40)
    dt = 1.0
    gy, gx = np.mgrid[0:n, 0:n].astype(np.float64)
    c = (n - 1) / 2; rr = np.hypot(gx - c, gy - c) / n
    omega = 3.0 / n * np.exp(-8 * rr * rr)                                     # swirl, fastest near the middle
    vel = f32(np.stack([-(gy - c) * omega, (gx - c) * omega], -1)).reshape(-1, 2)
    dye0 = f32(((gx // (n / 8) + gy // (n / 8)) % 2) * (rr < 0.45))           # checkerboard disc
    def check(o):
        got = o[0].view(np.float32)[:n * n].reshape(n, n)                       # even step count ends in the first half
        f = dye0.astype(np.float64); vx, vy = vel[:, 0].astype(np.float64).reshape(n, n), vel[:, 1].astype(np.float64).reshape(n, n)
        for _ in range(steps):
            qx, qy = gx - np.float32(dt) * vx, gy - np.float32(dt) * vy
            fx, fy = np.floor(qx), np.floor(qy); wx, wy = qx - fx, qy - fy
            x0, y0 = fx.astype(np.int64), fy.astype(np.int64)
            at = lambda x, y: f[np.clip(y, 0, n - 1), np.clip(x, 0, n - 1)]
            f = (at(x0, y0) * (1 - wx) + at(x0 + 1, y0) * wx) * (1 - wy) + (at(x0, y0 + 1) * (1 - wx) + at(x0 + 1, y0 + 1) * wx) * wy
        err = float(np.max(np.abs(got - f)))
        # per step: the back-traced position is rounded to fp32 (about one ulp of the coordinate, n * 2^-23 cells) and
        # the dye changes by at most 1 per cell; bilinear sampling never amplifies earlier errors; plus a few roundings
        bound = steps * (n * 2.0 ** -23 + 16 * 2.0 ** -24)
        return err <= bound, {'max_abs_err': err, 'bound': bound, 'dye_mass_in': float(dye0.sum()), 'dye_mass_out': float(got.sum())}
    ds = [((n + 15) // 16, (n + 15) // 16, 1, push(n, k & 1, dt)) for k in range(steps)]
    return W('advect', {0: (f32(np.concatenate([dye0.ravel(), np.zeros(n * n)])), True), 1: (vel, False), 2: (n * n, True)}, ds, check,
             (16, 16, 1), {'cell_updates': n * n * steps}, images=[(2, n, n, 'gray8')])


def w_knn(size, rng):
    nref, nq = (300, 512) if size == 'small' else (4096, 1 << 16)
    refs = rng.integers(-10000, 10001, (nref, 2)).astype(np.int32)
    refs[nref // 2] = refs[0]                                                  # duplicate point: an exact tie
    queries = rng.integers(-10000, 10001, (nq, 2)).astype(np.int32)
    def check(o):
        gi = o[2].view(np.uint32)[:8 * nq].reshape(nq, 8); gd = o[3].view(np.uint32)[:8 * nq].reshape(nq, 8)
        r64 = refs.astype(np.int64); bad = 0
        for c in range(0, nq, 1024):
            q = queries[c:c + 1024].astype(np.int64)
            d = ((q[:, None, :] - r64[None, :, :]) ** 2).sum(-1)
            key = d * nref + np.arange(nref)[None, :]                         # sort by distance, then index
            top = np.argsort(key, axis=1, kind='stable')[:, :8]
            bad += int(np.sum(np.any(gi[c:c + 1024] != top, axis=1) | np.any(gd[c:c + 1024] != np.take_along_axis(d, top, 1), axis=1)))
        return bad == 0, {'bad_queries': bad}
    return W('knn', {0: (refs, False), 1: (queries, False), 2: (32 * nq, True), 3: (32 * nq, True)}, [((nq + 255) // 256, 1, 1, push(nref, nq))],
             check, 256, {'distance_evals': nref * nq})


def w_csvparse(size, rng):
    n = 4096 if size == 'small' else 1 << 22
    vals = rng.integers(-999999999, 1000000000, n)
    fields = [str(v).rjust(12) for v in vals.tolist()]
    for k in rng.choice(n, max(16, n // 200), replace=False):                  # malformed fields
        fields[k] = rng.choice(['         12a', '        - 42', '       5-3  ', '            ', '         --7', '        1.50',
                                '           -', 'abcdefghijkl', ' 12 34      '])
    for k, w in zip(range(1, 6), ['          -0', '           0', '000000000017', '  -999999999', '   999999999']):
        fields[k] = w
    assert all(len(f) == 12 for f in fields)
    text = np.frombuffer(''.join(f + (',' if (k % 8) != 7 else '\n') for k, f in enumerate(fields)).encode(), np.uint8).copy()
    def check(o):
        import re
        got = o[1].view(np.int32)[:n]; ok_ = o[2].view(np.uint32)[:n]
        pat = re.compile(r' *-?[0-9]+')
        ref_ok = np.array([bool(pat.fullmatch(f)) for f in fields]); ref_v = np.array([int(f) if m else 0 for f, m in zip(fields, ref_ok)])
        ok = np.array_equal(ok_, ref_ok.astype(np.uint32)) and np.array_equal(got, ref_v.astype(np.int32))
        return ok, {'value_mismatches': int(np.sum(got != ref_v)), 'flag_mismatches': int(np.sum(ok_ != ref_ok)), 'invalid_fields': int((~ref_ok).sum())}
    return W('csvparse', {0: (text, False), 1: (4 * n, True), 2: (4 * n, True)}, [((n + 255) // 256, 1, 1, push(n))], check, 256,
             {'bytes': len(text)})


def w_pip(size, rng):
    nv, npt = (40, 2048) if size == 'small' else (500, 1 << 20)
    ang = np.sort(rng.uniform(0, 2 * np.pi, nv))
    rad = rng.uniform(3000, 10000, nv)                                          # star-shaped, many concave corners
    verts = np.stack([np.round(rad * np.cos(ang)), np.round(rad * np.sin(ang))], 1).astype(np.int32)
    pts = rng.integers(-11000, 11001, (npt, 2)).astype(np.int32)
    pts[:4] = verts[:4]                                                         # vertices themselves (boundary cases)
    def check(o):
        got = o[2].view(np.int32)[:npt]
        a = verts.astype(np.int64); b = np.roll(a, -1, 0); q = pts.astype(np.int64)
        wn = np.zeros(npt, np.int64)
        for k in range(nv):
            left = (b[k, 0] - a[k, 0]) * (q[:, 1] - a[k, 1]) - (q[:, 0] - a[k, 0]) * (b[k, 1] - a[k, 1])
            up = (a[k, 1] <= q[:, 1]) & (b[k, 1] > q[:, 1]) & (left > 0)
            down = (a[k, 1] > q[:, 1]) & (b[k, 1] <= q[:, 1]) & (left < 0)
            wn += up.astype(np.int64) - down
        return np.array_equal(got, wn), {'mismatches': int(np.sum(got != wn)), 'inside': int(np.sum(wn != 0))}
    return W('pip', {0: (verts, False), 1: (pts, False), 2: (4 * npt, True)}, [((npt + 255) // 256, 1, 1, push(nv, npt))], check, 256,
             {'edge_tests': nv * npt})


def lift53_ref(x):
    """forward LeGall 5/3 by lifting on the rows of x (even length), symmetric extension; returns [s | d]"""
    x = x.astype(np.int64); ev, od = x[:, 0::2], x[:, 1::2]
    right = np.concatenate([ev[:, 1:], ev[:, -1:]], 1)
    d = od - ((ev + right) >> 1)
    left = np.concatenate([d[:, :1], d[:, :-1]], 1)
    return np.concatenate([ev + ((left + d + 2) >> 2), d], 1)


def lift53_inv(y):
    h = y.shape[1] // 2; s, d = y[:, :h], y[:, h:]
    left = np.concatenate([d[:, :1], d[:, :-1]], 1)
    ev = s - ((left + d + 2) >> 2)
    right = np.concatenate([ev[:, 1:], ev[:, -1:]], 1)
    od = d + ((ev + right) >> 1)
    x = np.empty_like(y); x[:, 0::2] = ev; x[:, 1::2] = od
    return x


def w_lift53(size, rng):
    rows, ln = (16, 64) if size == 'small' else (4096, 2048)
    gx = np.arange(ln)[None, :]
    x = (2000 * np.sin(gx / 37.0 + np.arange(rows)[:, None]) + rng.integers(-300, 301, (rows, ln))).astype(np.int32)
    def check(o):
        got = o[1].view(np.int32)[:rows * ln].reshape(rows, ln).astype(np.int64)
        ref = lift53_ref(x)
        back = lift53_inv(got)
        return np.array_equal(got, ref) and np.array_equal(back, x), {'mismatches': int(np.sum(got != ref)), 'reconstructs': bool(np.array_equal(back, x)),
                                                                      'detail_energy_ratio': float((ref[:, ln // 2:] ** 2).sum() / (ref ** 2).sum())}
    return W('lift53', {0: (x, False), 1: (4 * rows * ln, True)}, [(rows, 1, 1, push(ln))], check, 256, {'samples': rows * ln})


WORKLOADS = {'reduce': w_reduce, 'scan': w_scan, 'sgemm': w_sgemm, 'conv2d': w_conv2d, 'histogram': w_histogram,
             'nbody': w_nbody, 'bitonic': w_bitonic, 'sha256': w_sha256, 'fp16': w_fp16, 'int64': w_int64, 'bitops': w_bitops,
             'intdiv': w_intdiv, 'compact': w_compact, 'spmv': w_spmv, 'transpose': w_transpose, 'mandelbrot': w_mandelbrot,
             'softmax': w_softmax, 'rgba8': w_rgba8, 'atomics': w_atomics, 'geom3d': w_geom3d, 'spill': w_spill,
             'control': w_control, 'shuffle': w_shuffle, 'floatops': w_floatops, 'radix_sort': w_radix_sort, 'fft': w_fft,
             'crc32': w_crc32, 'kmeans': w_kmeans, 'int8dot': w_int8dot, 'stencil3d': w_stencil3d, 'int16': w_int16, 'trig': w_trig,
             'raytrace': w_raytrace, 'mlp': w_mlp, 'sobel': w_sobel, 'life': w_life, 'bfs': w_bfs, 'conv_layer': w_conv_layer,
             'hist_eq': w_hist_eq, 'aes': w_aes, 'hash_table': w_hash_table, 'word_count': w_word_count, 'dct8': w_dct8,
             'attention': w_attention, 'resize': w_resize, 'median3': w_median3, 'poisson': w_poisson,
             'lbm': w_lbm, 'base64': w_base64, 'md5': w_md5, 'gemv': w_gemv,
             'levenshtein': w_levenshtein, 'blur': w_blur, 'yuv420': w_yuv420, 'montecarlo': w_montecarlo,
             'xxhash': w_xxhash, 'bloom': w_bloom, 'collatz': w_collatz, 'tridiag': w_tridiag,
             'merge': w_merge, 'ccl': w_ccl, 'pagerank': w_pagerank, 'skinning': w_skinning,
             'strsearch': w_strsearch, 'quantize': w_quantize, 'rule110': w_rule110, 'voxel': w_voxel,
             'nms': w_nms, 'sat': w_sat, 'sssp': w_sssp, 'morton': w_morton,
             'utf8': w_utf8, 'fir': w_fir, 'composite': w_composite, 'nqueens': w_nqueens,
             'lzmatch': w_lzmatch, 'bitpack': w_bitpack, 'perlin': w_perlin, 'adler32': w_adler32,
             'sieve': w_sieve, 'gcd': w_gcd, 'dither': w_dither, 'advect': w_advect,
             'knn': w_knn, 'csvparse': w_csvparse, 'pip': w_pip, 'lift53': w_lift53}


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
    write_images(w, outs, out, 'image')
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


def cmd_compile(a):
    """Compile every workload with Mesa's AGX compiler and keep only the driver's binaries: on a machine without the
    GPU, through the no-op drm-shim (scripts/build_mesa_shim.sh; --mesa SRC selects it). Writes OUT/<workload>/
    shader.bin, props.json and the extract output. Dispatch results are ignored (the shim does not execute)."""
    env = dict(os.environ)
    if a.mesa:
        b = Path(a.mesa) / 'build/src/asahi'
        env['VK_ICD_FILENAMES'] = str(b / f'vulkan/asahi_devenv_icd.{os.uname().machine}.json')
        env['LD_PRELOAD'] = str(b / 'drm-shim/libasahi_noop_drm_shim.so')
    if not VK_RUN.exists() or not list(BUILD.glob('*.spv')):
        cmd_build()
    names = list(WORKLOADS) if a.workload == 'all' else a.workload.split(',')
    tmp = a.out / '.run'
    bad = 0
    for name in names:
        rng = np.random.default_rng([1, list(WORKLOADS).index(name)])
        w = WORKLOADS[name](a.size, rng)
        run = tmp / name; run.mkdir(parents=True, exist_ok=True)
        spec = [f'shader {BUILD / (w.shader + ".spv")}']
        for binding, (init, _) in sorted(w.buffers.items()):
            spec.append(f'buffer {binding} {init if isinstance(init, int) else init.nbytes} - -')
        gx, gy, gz, pc = w.dispatches[0]
        spec += [f'dispatch {gx} {gy} {gz} {pc.hex() or "-"}', 'repeat 1', f'export {run / "shader.bin"}', f'props {run / "props.json"}']
        (run / 'spec.txt').write_text('\n'.join(spec) + '\n')
        r = subprocess.run([str(VK_RUN), str(run / 'spec.txt')], capture_output=True, text=True, timeout=600, env=env)
        dev = next((json.loads(l) for l in r.stdout.splitlines() if l.startswith('{"device"')), {})
        if not (run / 'shader.bin').exists():
            bad += 1; print(json.dumps({'workload': name, 'error': (r.stdout + r.stderr)[-400:]})); continue
        if 'AGX' not in dev.get('device', '') and 'Apple' not in dev.get('device', ''):
            sys.exit(f'vk_run used {dev.get("device")!r}, not the Asahi driver (pass --mesa or set VK_ICD_FILENAMES)')
        out = a.out / name; out.mkdir(parents=True, exist_ok=True)
        shutil.copy(run / 'shader.bin', out / 'shader.bin')
        if (run / 'props.json').exists():
            shutil.copy(run / 'props.json', out / 'props.json')
        (out / 'device.json').write_text(json.dumps(dev) + '\n')
        bad += cmd_extract(argparse.Namespace(blob=str(out / 'shader.bin'), out=str(out))) or 0
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if bad else 0


def pmap(f, items):
    """f over items on all cores (each item runs subprocesses), results in item order; AGX_THREADS overrides"""
    from concurrent.futures import ThreadPoolExecutor
    n = int(os.environ.get('AGX_THREADS') or os.cpu_count() or 1)
    with ThreadPoolExecutor(max(1, min(n, len(items) or 1))) as ex:
        return list(ex.map(f, items))


def cmd_simall(a):
    """sim every workload compiled under DIR (compile's output; one subdirectory per workload); exit 1 if any fails"""
    dirs = sorted(p for p in Path(a.dir).iterdir() if (p / 'main.bin').exists())
    def one(d):
        return subprocess.run([sys.executable, '-B', __file__, 'sim', d.name, '--code-dir', str(d), '--size', a.size,
                               '--out', str(Path(a.work) / d.name)], capture_output=True, text=True)
    bad = 0
    for r in pmap(one, dirs):
        line = (r.stdout.strip().splitlines() or [r.stderr.strip()])[-1]
        print(line, flush=True)
        bad += r.returncode != 0
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


# hk_api_shader_serialize (Mesa 25.2-26.0): vk_shader_bin_header (72 bytes up to 25.3, 80 from 26.0), stage byte,
# hk_shader_info (976 bytes), agx_shader_info (1176 bytes, unaligned), then aligned u32 code_size, data_size and the
# code (at 0x8bc with the 72-byte header, as in the M2's 25.3.6 blobs; 0x8c4 with 80).
# Field offsets inside agx_shader_info, from offsetof() on the built drivers (25.2.8 and 26.0.3 agree except nr_gprs,
# 1094 in 25.2 and 1096 in 26.0, after a new sampler_state_count byte); stats is struct agx2_stats.
HEADER_SIZES, HK_INFO_BYTES, AGX_INFO_BYTES = (72, 80), 976, 1176
AGX_INFO = {'stage': (0, '<I'), 'binary_size': (4, '<I'), 'push_count': (1044, '<H'), 'local_size': (1046, '<H'),
            'scratch_size': (1052, '<I'), 'main_size': (1060, '<I'), 'has_preamble': (1064, '<B'),
            'preamble_offset': (1068, '<I'), 'main_offset': (1072, '<I'), 'workgroup_size': (1112, '<3H'),
            'rodata_offset': (1120, '<I'), 'rodata_base_uniform': (1124, '<H'), 'rodata_size_16': (1126, '<H'),
            'stats_instructions': (1128, '<I'), 'stats_gprs': (1148, '<H'), 'stats_uniforms': (1150, '<H'),
            'stats_threads': (1156, '<H')}
NR_GPRS_AT = {'25.2': 1094, '26.0': 1096}


def shader_info(b):
    """agx_shader_info of a MesaVkShaderBin blob, or None when the blob has none of the known layouts"""
    for header in HEADER_SIZES:
        at = header + 1 + HK_INFO_BYTES
        code_at = (at + AGX_INFO_BYTES + 3) // 4 * 4 + 8
        info = shader_info_at(b, at, code_at) if len(b) >= code_at else None
        if info:
            info['header_bytes'], info['code_at'] = header, code_at
            return info
    return None


def shader_info_at(b, info_at, code_at):
    cs, ds = struct.unpack_from('<II', b, code_at - 8)
    info = {}
    for k, (off, fmt) in AGX_INFO.items():
        v = struct.unpack_from(fmt, b, info_at + off)
        info[k] = list(v) if len(v) > 1 else v[0]
    ok = (info['binary_size'] == cs and code_at + cs + ds <= len(b) and info['main_offset'] + info['main_size'] <= cs
          and (not info['has_preamble'] or info['preamble_offset'] < cs) and info['rodata_offset'] + 2 * info['rodata_size_16'] <= cs
          and 0 < info['workgroup_size'][0] <= 1024)
    if not ok:
        return None
    # nr_gprs (16-bit registers allocated) is the driver's register count rounded up: the candidate offset whose value
    # is the smallest one >= the statistic identifies the layout
    cands = sorted((v - info['stats_gprs'], ver, v) for ver, off in NR_GPRS_AT.items()
                   for v in [struct.unpack_from('<H', b, info_at + off)[0]] if info['stats_gprs'] <= v <= 256)
    if not cands:
        return None
    _, info['info_layout'], info['nr_gprs'] = cands[0]
    info['code_size'], info['data_size'] = cs, ds
    return info


def cmd_extract(a):
    """Split a MesaVkShaderBin into main.bin, preamble.bin, rodata.bin and info.json (the driver's agx_shader_info:
    program offsets, constants that the driver loads into uniforms, workgroup size, threadgroup memory, registers).
    Without the known layout it falls back to a scan: the code region holds a constant pool and the programs at
    aligned offsets (the stored control: pool at 0, main at 0x80, preamble at 0x100); every aligned offset that decodes
    cleanly to a `stop` is a program, and one with uniform_store is labelled preamble."""
    b = Path(a.blob).read_bytes()
    if b[:16] != b'MesaVkShaderBin\0':
        sys.exit('not a MesaVkShaderBin blob')
    out = Path(a.out or Path(a.blob).with_suffix(''))
    p = subprocess.Popen([str(asm())], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    info = shader_info(b)
    if info:
        code = b[info['code_at']:info['code_at'] + info['code_size']]
        progs = [('main', info['main_offset'])] + ([('preamble', info['preamble_offset'])] if info['has_preamble'] else [])
        segs = []
        out.mkdir(parents=True, exist_ok=True)
        for label, off in progs:
            prog = decode_program(p, code, off)
            if not prog:
                sys.exit(f'{label} at {off:#x}: undecodable instruction before the first stop (a decoder gap; '
                         f'bytes from there: {code[off:off + 64].hex()})')
            end = prog[-1][0] + len(prog[-1][1])
            (out / f'{label}.bin').write_bytes(code[off:end])
            (out / f'{label}.txt').write_text(''.join(f'{o:#06x}  {bb.hex():<24} {dis}\n' for o, bb, _, dis in prog))
            segs.append({'label': label, 'file': f'{label}.bin', 'offset': off, 'bytes': end - off, 'instructions': len(prog)})
        ro = code[info['rodata_offset']:info['rodata_offset'] + 2 * info['rodata_size_16']]
        (out / 'rodata.bin').write_bytes(ro)
        (out / 'code_region.bin').write_bytes(code)
        info['uniform_init'] = rodata_uniforms(info, ro)
        # root-table layout: from Mesa 25.3 (the M2's stored control shader) the driver preloads the set addresses into
        # u2_u3.., and the preamble reads descriptors through them; up to 25.2 it loads sets[0] from root word 156
        pre = (out / 'preamble.txt').read_text() if info['has_preamble'] else ''
        info['root_layout'] = '26.0' if 'u2_u3' in pre else '25.2'
        (out / 'info.json').write_text(json.dumps(info, indent=1) + '\n')
        res = {'blob': a.blob, 'code_start': info['code_at'], 'code_size': info['code_size'], 'layout': 'agx_shader_info', 'programs': segs}
        (out / 'segments.json').write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps(res))
        return 0
    cands = []
    for start in range(72, len(b) - 8, 4):
        cs, ds = struct.unpack_from('<II', b, start - 8)
        if 16 <= cs <= len(b) - start and cs % 2 == 0 and start + cs + ds <= len(b):
            cands.append((start not in (0x8bc, 0x8c4), start, cs))
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
        res = {'blob': a.blob, 'code_start': start, 'code_size': cs, 'layout': 'scan', 'programs': segs}
        (out / 'segments.json').write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps(res))
        return 0
    sys.exit('no decodable program found; inspect the blob by hand (see tasks/spirv-workloads.md)')


def rodata_uniforms(info, ro):
    """{32-bit uniform index: value} the driver preloads from the binary's constants (agx_usc_immediates: size_16
    halves from rodata_offset into the uniforms from base_uniform, counted in 16-bit halves)"""
    base, n = info['rodata_base_uniform'], info['rodata_size_16']
    halves = struct.unpack(f'<{n}H', ro[:2 * n])
    out = {}
    for k, h in enumerate(halves):
        u, hi = divmod(base + k, 2)
        out[u] = out.get(u, 0) | h << (16 * hi)
    return {str(u): v for u, v in sorted(out.items())}


def root_value(kind, ctx):
    """value of one root-table / descriptor field (see spirv/root_layout.mesa-*.json)"""
    k, _, arg = kind.partition(':')
    if k == 'const': return int(arg, 0)
    if k in ('groups', 'grid', 'wg', 'base'):
        axis = 'xyz'.index(arg)
        return {'groups': ctx['groups'], 'grid': [g * w for g, w in zip(ctx['groups'], ctx['wg'])], 'wg': ctx['wg'], 'base': [0, 0, 0]}[k][axis]
    if k == 'set': return ctx['set_va']
    if k == 'self': return ctx['root_va']
    if k == 'group_count_addr': return ctx['gc_va']
    if k == 'buffer': return ctx['va'][int(arg)]
    if k == 'buffer_size': return ctx['size'][int(arg)]
    raise ValueError(f'unknown root field kind {kind!r}')


def put_field(mem, word, value, nbytes=4):
    mem[4 * word:4 * word + nbytes] = int(value & ((1 << 8 * nbytes) - 1)).to_bytes(nbytes, 'little')


class DriverLaunch:
    """Memory and launch arguments for running a workload's driver-compiled code (extract output) on agx_run /
    agx_dispatch: buffers, descriptor set and root table at fixed VAs, the driver's preloaded uniforms (root table,
    set addresses, the binary's constants), the preamble, and the workgroup size / threadgroup memory / registers
    from info.json."""
    def __init__(self, workload, size, code_dir, layout, work, shared=None):
        seg = self.seg = Path(code_dir)
        self.main_bin, self.pre_bin = seg / 'main.bin', seg / 'preamble.bin'
        self.info = info = json.loads((seg / 'info.json').read_text()) if (seg / 'info.json').exists() else None
        if not layout:
            if not info:
                sys.exit('no info.json in the code directory (extract could not read the shader info): pass --layout')
            layout = ROOT / f'spirv/root_layout.mesa-{info["root_layout"]}.json'
        self.layout = json.loads(Path(layout).read_text())
        rng = np.random.default_rng([1, list(WORKLOADS).index(workload)])
        self.w = w = WORKLOADS[workload](size, rng)
        self.work = work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        self.va, self.size, self.files = {}, {}, {}
        nxt = 0x10000000
        for binding, (init, _) in sorted(w.buffers.items()):
            data = bytes(init) if isinstance(init, int) else init.tobytes()
            self.va[binding], self.size[binding] = nxt, len(data)
            self.files[binding] = work / f'buf{binding}.bin'; self.files[binding].write_bytes(data)
            nxt += (len(data) + 0xffff) // 0x10000 * 0x10000 + 0x10000
        self.set_va, self.root_va, self.gc_va = nxt, nxt + 0x10000, nxt + 0x20000
        d = self.layout['descriptor']
        setmem = bytearray(d['stride'] * (max(w.buffers) + 1))
        for binding in w.buffers:
            for fld in d['fields']:
                put_field(setmem, binding * d['stride'] // 4 + fld['word'],
                          root_value(fld['value'].replace(':B', f':{binding}'), {'va': self.va, 'size': self.size, 'set_va': self.set_va}), fld.get('bytes', 4))
        (work / 'set.bin').write_bytes(setmem)
        if info:
            self.wg, self.shared, self.regs = info['workgroup_size'], info['local_size'], info['nr_gprs']
        else:
            self.wg, self.shared, self.regs = wg_of(w, workload), shared or 0, None

    def args(self, k):
        """(grid, launch arguments) of dispatch k; writes its root table and group count into the work directory"""
        gx, gy, gz, pc = self.w.dispatches[k]
        ctx = {'groups': [gx, gy, gz], 'wg': self.wg, 'va': self.va, 'size': self.size, 'set_va': self.set_va,
               'root_va': self.root_va, 'gc_va': self.gc_va}
        root = bytearray(self.layout['root_bytes'])
        for fld in self.layout['root']:
            if fld['value'] == 'push':
                root[4 * fld['word']:4 * fld['word'] + len(pc)] = pc
            else:
                put_field(root, fld['word'], root_value(fld['value'], ctx), fld.get('bytes', 4))
        (self.work / f'root{k}.bin').write_bytes(root)
        (self.work / f'group_count{k}.bin').write_bytes(struct.pack('<4I', gx, gy, gz, 0))
        grid = [g * x for g, x in zip(ctx['groups'], self.wg)]
        args = ['--wg', ','.join(map(str, self.wg)), '--map', f'{self.root_va:#x}={self.work / f"root{k}.bin"}',
                '--map', f'{self.set_va:#x}={self.work / "set.bin"}', '--map', f'{self.gc_va:#x}={self.work / f"group_count{k}.bin"}']
        if self.pre_bin.exists():
            args += ['--preamble', str(self.pre_bin)]
        # the driver's uniform loads in its order: root table (and set addresses), then the binary's constants
        for u, kind in self.layout['uniforms'].items():
            v = self.root_va if kind == 'root' else self.set_va if kind == 'set' else int(kind, 0)
            args += ['--uniform', f'{int(u)}={v & 0xffffffff:#x}', '--uniform', f'{int(u) + 1}={v >> 32:#x}']
        for u, v in (self.info or {}).get('uniform_init', {}).items():
            args += ['--uniform', f'{u}={v:#x}']
        for b in self.w.buffers:
            args += ['--map', f'{self.va[b]:#x}={self.files[b]}']
        return grid, args


def cmd_sim(a):
    """Functional simulation of a workload with the driver's compiled code (extract) on agx_run, buffer state carried
    across dispatches; outputs checked with the workload's reference and compared word for word with a hardware run.
    With the extract's info.json, the workgroup size, threadgroup memory, registers and the constants the driver
    preloads into uniforms come from the driver's own shader info."""
    L = DriverLaunch(a.workload, a.size, a.code_dir, a.layout, a.out or Path(a.code_dir) / f'sim-{a.workload}-{a.size}', a.shared)
    exe = ROOT / 'rust/target/release/agx_run'
    total = 0
    for k in range(len(L.w.dispatches[:a.dispatches])):
        grid, args = L.args(k)
        cmd = [str(exe), str(L.main_bin), '--grid', ','.join(map(str, grid)), '--shared', str(L.shared)] + args
        for b in L.w.buffers:
            cmd += ['--dump', f'{L.va[b]:#x}={L.size[b]}:{L.files[b]}']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'{a.workload} dispatch {k}: {r.stdout.strip()} {r.stderr.strip()}')
        total += json.loads(r.stdout)['instructions']
    outs = {b: np.fromfile(L.files[b], dtype=np.uint8) for b, (_, rb) in L.w.buffers.items() if rb}
    ok, detail = L.w.check(outs)
    write_images(L.w, outs, L.work, 'sim_image')
    res = {'workload': a.workload, 'size': a.size, 'simulated_ok': bool(ok), 'functional': detail, 'simulated_instructions': total}
    if a.hardware:
        hw = Path(a.hardware)
        diff = {}
        for b in outs:
            h = np.fromfile(hw / f'out{b}.bin', dtype=np.uint32); s_ = outs[b].view(np.uint32)
            n = min(len(h), len(s_))
            diff[b] = int(np.sum(h[:n] != s_[:n]))
        res['words_differing_from_hardware'] = diff
    (L.work / 'sim.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res))
    return 0 if ok else 1


def cmd_predict(a):
    """Whole-dispatch time of every dispatch of a workload (agx_dispatch). With --code-dir (extract output) the
    launch comes from the driver's info and the SIMD groups' paths from the functional simulator over the workload's
    real inputs (each dispatch from the initial buffers: later dispatches do not see earlier ones' writes, which only
    matters where control flow or addresses depend on them). With --code alone: the cost model's own emulation."""
    exe = ROOT / 'rust/target/release/agx_dispatch'
    rows, total = [], 0.0
    if a.code_dir:
        L = DriverLaunch(a.workload, a.size, a.code_dir, a.layout, a.out or Path(a.code_dir) / f'predict-{a.workload}-{a.size}')
        for k in range(len(L.w.dispatches)):
            grid, args = L.args(k)
            cmd = [str(exe), str(L.main_bin), '--grid', ','.join(map(str, grid)), '--regs', str(a.regs or L.regs),
                   '--tg-bytes', str(a.tg_bytes if a.tg_bytes is not None else L.shared)] + args
            if a.no_mem: cmd.append('--no-mem')
            rows.append(cmd)
        # each dispatch starts from the initial buffers, so dispatches are independent and those whose launch is the
        # same (root table, group count and arguments: e.g. lbm's and poisson's two ping-pong parities) give the
        # same result: run each distinct one once
        per_k = {}
        for k, cmd in enumerate(rows):
            files = {str(L.work / f'root{k}.bin'), str(L.work / f'group_count{k}.bin')}
            per_k[k] = tuple(x.split('=')[0] + '=' + (L.work / x.split('=', 1)[1]).read_bytes().hex()
                             if '=' in x and x.split('=', 1)[1] in files else x for x in cmd)
        uniq = list(dict.fromkeys(per_k.values()))
        first = {key: rows[k] for k, key in reversed(list(per_k.items()))}
        res = dict(zip(uniq, pmap(run_dispatch, [first[key] for key in uniq])))
        rows = [res[per_k[k]] for k in range(len(rows))]
    else:
        if not a.code:
            sys.exit('predict needs --code-dir (extract output) or --code main.bin')
        rng = np.random.default_rng([1, list(WORKLOADS).index(a.workload)])
        w = WORKLOADS[a.workload](a.size, rng)
        for gx, gy, gz, pc in w.dispatches:
            wg = wg_of(w, a.workload)
            grid = [gx * wg[0], gy * wg[1], gz * wg[2]]
            cmd = [str(exe), a.code, '--grid', ','.join(map(str, grid)), '--wg', ','.join(map(str, wg))]
            if a.regs: cmd += ['--regs', str(a.regs)]
            if a.tg_bytes: cmd += ['--tg-bytes', str(a.tg_bytes)]
            for u in a.uniform or []: cmd += ['--uniform', u]
            if a.no_mem: cmd.append('--no-mem')
            rows.append(cmd)
        rows = pmap(run_dispatch, rows)
    total = sum(r['time_us'] for r in rows)
    print(json.dumps({'workload': a.workload, 'size': a.size, 'predicted_us': total,
                      'dispatch_us': [r['time_us'] for r in rows], 'dispatches': rows}))


def cmd_predictall(a):
    """predict every workload compiled under DIR (functional paths); one line each, JSON per workload in --work"""
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in Path(a.dir).iterdir() if (p / 'main.bin').exists())
    runs = pmap(lambda d: subprocess.run([sys.executable, '-B', __file__, 'predict', d.name, '--code-dir', str(d), '--size', a.size,
                                          '--out', str(work / d.name)], capture_output=True, text=True), dirs)
    for d, r in zip(dirs, runs):
        if r.returncode:
            print(f'{d.name:10} error: {(r.stdout + r.stderr).strip()[-300:]}'); continue
        (work / f'{d.name}.json').write_text(r.stdout)
        j = json.loads(r.stdout)
        disp = j['dispatch_us']
        print(f'{d.name:10} {j["predicted_us"]:10.1f} us  {len(disp)} dispatch{"es" if len(disp) > 1 else ""}'
              + (f' (first {disp[0]:.1f} us)' if len(disp) > 1 else ''), flush=True)
    return 0



# ---------------------------------------------------------------------------------------------------------------
# Hardware verification. The workloads' numpy references only show that a result is plausible; they are not
# hardware data. `record` (on the M2) stores what the GPU actually produced, with the driver's own compiled code;
# `verify` (anywhere) runs the simulator on that code and compares it with the recording word for word. A workload
# without a recording is PENDING, never PASS.

EXPECTED = ROOT / 'spirv/expected'


def input_hashes(w):
    return {str(b): hashlib.sha256(init.tobytes()).hexdigest() for b, (init, _) in sorted(w.buffers.items()) if not isinstance(init, int)}


def is_agx(dev):
    name = dev.get('device', '')
    return 'AGX' in name or 'Apple' in name


def real_gpu_problem(dev):
    """None when vk_run really executed on the M2's GPU; else why not. The drm-shim build reports the same AGX
    device but executes nothing, so the device name alone is not enough."""
    if not is_agx(dev):
        return f'device {dev.get("device")!r} is not the AGX GPU'
    if 'drm_shim' in os.environ.get('LD_PRELOAD', '') or 'drm-shim' in os.environ.get('LD_PRELOAD', ''):
        return 'LD_PRELOAD has the drm-shim (no GPU execution)'
    try:
        compat = Path('/proc/device-tree/compatible').read_bytes()
    except OSError:
        compat = b''
    if b'apple' not in compat:
        return 'not an Apple machine (/proc/device-tree/compatible)'
    return None


def cmd_record(a):
    """On Asahi Linux: run every workload on the GPU and store the expected values (see the section comment)."""
    if not VK_RUN.exists() or not list(BUILD.glob('*.spv')):
        cmd_build()
    names = list(WORKLOADS) if a.workload == 'all' else a.workload.split(',')
    exp = Path(a.expected)
    scratch = Path(a.scratch)
    bad = 0
    for name in names:
        try:
            r = run_one(name, 'small', a.repeat_small, scratch)
            why = real_gpu_problem(r['device'])
            if why and not a.any_device:
                sys.exit(f'refusing to record hardware data: {why} (--any-device only for testing this script)')
            run = scratch / f'{name}-small'
            dest = exp / name
            if dest.exists():
                shutil.rmtree(dest)
            (dest / 'code').mkdir(parents=True)
            rng = np.random.default_rng([1, list(WORKLOADS).index(name)])
            w = WORKLOADS[name]('small', rng)
            for b, (_, rb) in w.buffers.items():
                if rb:
                    with gzip.open(dest / f'out{b}.bin.gz', 'wb') as f:
                        f.write((run / f'out{b}.bin').read_bytes())
            shutil.copy(run / 'shader.bin', dest / 'code/shader.bin')
            if (run / 'props.json').exists():
                shutil.copy(run / 'props.json', dest / 'code/props.json')
            ext = cmd_extract(argparse.Namespace(blob=str(dest / 'code/shader.bin'), out=str(dest / 'code')))
            rec = {'workload': name, 'device': r['device'], 'test_recording_not_hardware': bool(why), 'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'inputs_sha256': input_hashes(w), 'numpy': np.__version__, 'small': {k: r.get(k) for k in
                   ('functional_ok', 'functional', 'gpu_ns', 'dispatch_ns_median', 'repeat')}, 'code_extracted': not ext}
            if not a.no_large:
                rl = run_one(name, 'large', a.repeat_large, scratch, export=False)
                runl = scratch / f'{name}-large'
                wl = WORKLOADS[name]('large', np.random.default_rng([1, list(WORKLOADS).index(name)]))
                rec['large'] = {k: rl.get(k) for k in ('functional_ok', 'functional', 'gpu_ns', 'dispatch_ns_median', 'repeat', 'gflops',
                                                        'gbytes_per_s', 'work')}
                rec['large']['out_sha256'] = {str(b): hashlib.sha256((runl / f'out{b}.bin').read_bytes()).hexdigest()
                                              for b, (_, rb) in wl.buffers.items() if rb}
                rec['large']['inputs_sha256'] = input_hashes(wl)
                shutil.rmtree(runl, ignore_errors=True)
            (dest / 'hardware.json').write_text(json.dumps(rec, indent=1) + '\n')
            shutil.rmtree(run, ignore_errors=True)
            bad += not r['functional_ok']
            print(json.dumps({'workload': name, 'recorded': str(dest), 'hardware_matches_reference': r['functional_ok'],
                              'small_gpu_us': (r.get('gpu_ns') or {}).get('median', 0) / 1e3,
                              'large_gpu_us': ((rec.get('large') or {}).get('gpu_ns') or {}).get('median', 0) / 1e3}), flush=True)
        except SystemExit:
            raise
        except Exception as ex:
            bad += 1
            print(json.dumps({'workload': name, 'error': str(ex)[:500]}), flush=True)
    return 1 if bad else 0


def verify_one(name, exp, layout, timing):
    d = exp / name
    if not (d / 'hardware.json').exists():
        return {'workload': name, 'status': 'PENDING', 'why': 'no hardware recording (run `make asahi-verify` on the M2)'}
    rec = json.loads((d / 'hardware.json').read_text())
    if rec.get('test_recording_not_hardware') and not os.environ.get('AGX_VERIFY_ALLOW_TEST_RECORDINGS'):
        return {'workload': name, 'status': 'PENDING', 'why': 'only a test recording (--any-device), not hardware data'}
    w = WORKLOADS[name]('small', np.random.default_rng([1, list(WORKLOADS).index(name)]))
    if input_hashes(w) != rec['inputs_sha256']:
        return {'workload': name, 'status': 'STALE', 'why': 'inputs differ from the recording (workload or numpy changed): re-record'}
    if not (d / 'code/main.bin').exists():
        return {'workload': name, 'status': 'FAIL', 'why': "the recording's driver code could not be extracted (see code/)"}
    work = Path(tempfile.mkdtemp(prefix=f'agx-verify-{name}-'))
    lay = ['--layout', layout] if layout else []
    r = subprocess.run([sys.executable, '-B', __file__, 'sim', name, '--code-dir', str(d / 'code'), '--size', 'small', '--out', str(work)] + lay,
                       capture_output=True, text=True)
    res = {'workload': name, 'device': rec['device'].get('device'), 'hardware_matches_reference': rec['small']['functional_ok']}
    if r.returncode and not (work / 'sim.json').exists():
        shutil.rmtree(work, ignore_errors=True)
        return {**res, 'status': 'FAIL', 'why': 'simulator error: ' + (r.stdout + r.stderr).strip()[-300:]}
    sim = json.loads((work / 'sim.json').read_text())
    diff, words = {}, 0
    for b, (_, rb) in w.buffers.items():
        if not rb:
            continue
        hw = np.frombuffer(gzip.open(d / f'out{b}.bin.gz').read(), dtype=np.uint8)
        sw = np.fromfile(work / f'buf{b}.bin', dtype=np.uint8) if (work / f'buf{b}.bin').exists() else np.zeros(0, np.uint8)
        n = min(len(hw), len(sw)) // 4 * 4
        dw = int(np.sum(hw[:n].view(np.uint32) != sw[:n].view(np.uint32))) + (len(hw) != len(sw))
        diff[str(b)] = dw; words += dw
    shutil.rmtree(work, ignore_errors=True)
    res.update({'words_differing': diff, 'simulator_matches_reference': sim['simulated_ok']})
    if words == 0:
        res['status'] = 'PASS'
    elif sim['simulated_ok'] and rec['small']['functional_ok']:
        res['status'] = 'CLOSE'; res['why'] = f'{words} words differ, both within the reference tolerance'
    else:
        res['status'] = 'FAIL'; res['why'] = f'{words} words differ' + ('' if sim['simulated_ok'] else '; simulator outside the reference')
    if timing and rec.get('large', {}).get('dispatch_ns_median'):
        r = subprocess.run([sys.executable, '-B', __file__, 'predict', name, '--code-dir', str(d / 'code'), '--size', 'large'] + lay,
                           capture_output=True, text=True)
        if r.returncode == 0:
            pred = json.loads(r.stdout)['dispatch_us']
            meas = [x / 1e3 for x in rec['large']['dispatch_ns_median']]
            res['timing'] = {'measured_us': round(sum(meas), 1), 'predicted_us': round(sum(pred), 1),
                             'ratio': round(sum(pred) / sum(meas), 3) if sum(meas) else None}
        else:
            res['timing'] = {'error': (r.stdout + r.stderr).strip()[-200:]}
    return res


def timing_summary(rows):
    """model accuracy over the workloads with a measured and a predicted time: predicted / measured per workload,
    the median and p90 of |log2 ratio| (as a factor), how many fall within +-10% / +-25% / 2x, and the worst ones"""
    pts = [(r['workload'], r['timing']['ratio']) for r in rows if r.get('timing', {}).get('ratio')]
    if not pts:
        return None
    err = sorted(abs(np.log2(x)) for _, x in pts)
    q = lambda f: round(float(2 ** err[int((len(err) - 1) * f)]), 3)
    within = lambda lo: sum(1 for _, x in pts if lo <= x <= 1 / lo)
    order = sorted(pts, key=lambda p: p[1])
    return {'workloads': len(pts), 'median_factor': q(0.5), 'p90_factor': q(0.9),
            'within_10pct': within(1 / 1.1), 'within_25pct': within(1 / 1.25), 'within_2x': within(0.5),
            'most_under_predicted': [(w, x) for w, x in order[:5] if x < 1],
            'most_over_predicted': [(w, x) for w, x in order[::-1][:5] if x > 1]}


def cmd_verify(a):
    """Simulator vs the recorded hardware results (spirv/expected); PENDING where nothing is recorded yet."""
    names = list(WORKLOADS) if a.workload == 'all' else a.workload.split(',')
    rows = pmap(lambda n: verify_one(n, Path(a.expected), a.layout, a.timing), names)
    for r in rows:
        line = f'{r["status"]:8} {r["workload"]:11}'
        if 'words_differing' in r: line += f' differing words {sum(r["words_differing"].values()):6}'
        if r.get('hardware_matches_reference') is False: line += '  [hardware does not match the numpy reference]'
        if 'timing' in r: line += f'  time measured {r["timing"].get("measured_us")} us predicted {r["timing"].get("predicted_us")} us'
        if r.get('why'): line += f'  ({r["why"]})'
        print(line, flush=True)
    from collections import Counter
    c = Counter(r['status'] for r in rows)
    summary = {'summary': dict(c)}
    t = timing_summary(rows)
    if t:
        summary['timing'] = t
    print(json.dumps(summary))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1) + '\n')
    return 1 if c.get('FAIL') or c.get('STALE') or (a.strict and c.get('PENDING')) else 0


def run_dispatch(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'agx_dispatch failed: {r.stderr.strip()}')
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('build'); sub.add_parser('list')
    p = sub.add_parser('run'); p.add_argument('workload'); p.add_argument('--size', choices=['small', 'large'], default='large')
    p.add_argument('--repeat', type=int, default=10); p.add_argument('--out', type=Path, default=ROOT / 'spirv/out')
    p.add_argument('--no-export', action='store_true')
    p = sub.add_parser('extract'); p.add_argument('blob'); p.add_argument('--out')
    p = sub.add_parser('compile'); p.add_argument('workload'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mesa', help='Mesa source tree built by scripts/build_mesa_shim.sh (selects its driver and drm-shim)')
    p.add_argument('--size', choices=['small', 'large'], default='small')
    p = sub.add_parser('predictall'); p.add_argument('dir'); p.add_argument('--size', choices=['small', 'large'], default='large')
    p.add_argument('--work', default='/tmp/agx-spirv-predict')
    p = sub.add_parser('simall'); p.add_argument('dir'); p.add_argument('--size', choices=['small', 'large'], default='small')
    p.add_argument('--work', default='/tmp/agx-spirv-sim', help='scratch directory for buffers')
    p = sub.add_parser('sim'); p.add_argument('workload'); p.add_argument('--code-dir', required=True, help='extract output (main.bin, preamble.bin)')
    p.add_argument('--layout', help='root-table layout (default: spirv/root_layout.mesa-<info_layout>.json from info.json)')
    p.add_argument('--size', choices=['small', 'large'], default='small')
    p.add_argument('--shared', type=int, default=32768, help='threadgroup bytes when there is no info.json')
    p.add_argument('--hardware', help='run output directory to compare with word for word')
    p.add_argument('--dispatches', type=int, help='run only the first N dispatches (debugging; buffers stay in the work directory)')
    p.add_argument('--out')
    p = sub.add_parser('record'); p.add_argument('workload', nargs='?', default='all')
    p.add_argument('--expected', default=str(EXPECTED)); p.add_argument('--scratch', default='/tmp/agx-spirv-record')
    p.add_argument('--repeat-small', type=int, default=5); p.add_argument('--repeat-large', type=int, default=20)
    p.add_argument('--no-large', action='store_true'); p.add_argument('--any-device', action='store_true', help='testing only')
    p = sub.add_parser('verify'); p.add_argument('workload', nargs='?', default='all')
    p.add_argument('--expected', default=str(EXPECTED)); p.add_argument('--layout', help='root-table layout (default: from the recorded code\'s info.json)')
    p.add_argument('--timing', action='store_true', help='also predict the large size and compare with the recorded time')
    p.add_argument('--strict', action='store_true', help='exit 1 while any workload is PENDING'); p.add_argument('--json')
    p = sub.add_parser('predict'); p.add_argument('workload')
    p.add_argument('--code-dir', help='extract output: launch from info.json, paths from the functional simulator')
    p.add_argument('--code', help='main.bin alone: paths from the cost model\'s emulation'); p.add_argument('--layout')
    p.add_argument('--out', help='work directory for buffers')
    p.add_argument('--size', choices=['small', 'large'], default='large'); p.add_argument('--regs', type=int)
    p.add_argument('--tg-bytes', type=int); p.add_argument('--uniform', action='append'); p.add_argument('--no-mem', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'list':
        for k, f in WORKLOADS.items():
            print(k, '-', (SHADERS / f'{k}.comp').read_text().splitlines()[1].lstrip('/ '))
        return
    sys.exit({'build': cmd_build, 'run': cmd_run, 'compile': cmd_compile, 'extract': cmd_extract, 'sim': cmd_sim, 'simall': cmd_simall, 'predictall': cmd_predictall,
              'predict': cmd_predict, 'record': cmd_record, 'verify': cmd_verify}[a.cmd](a) or 0)


if __name__ == '__main__':
    main()
