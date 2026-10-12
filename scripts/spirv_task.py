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
    def __init__(self, shader, buffers, dispatches, check, wg, work=None):
        self.shader, self.buffers, self.dispatches, self.check, self.wg, self.work = shader, buffers, dispatches, check, wg, work or {}


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


WORKLOADS = {'reduce': w_reduce, 'scan': w_scan, 'sgemm': w_sgemm, 'conv2d': w_conv2d, 'histogram': w_histogram,
             'nbody': w_nbody, 'bitonic': w_bitonic, 'sha256': w_sha256, 'fp16': w_fp16, 'int64': w_int64, 'bitops': w_bitops,
             'intdiv': w_intdiv, 'compact': w_compact, 'spmv': w_spmv, 'transpose': w_transpose, 'mandelbrot': w_mandelbrot,
             'softmax': w_softmax, 'rgba8': w_rgba8, 'atomics': w_atomics, 'geom3d': w_geom3d, 'spill': w_spill,
             'control': w_control, 'shuffle': w_shuffle, 'floatops': w_floatops, 'radix_sort': w_radix_sort, 'fft': w_fft,
             'crc32': w_crc32, 'kmeans': w_kmeans, 'int8dot': w_int8dot, 'stencil3d': w_stencil3d, 'int16': w_int16, 'trig': w_trig}


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


def cmd_simall(a):
    """sim every workload compiled under DIR (compile's output; one subdirectory per workload); exit 1 if any fails"""
    bad = 0
    for d in sorted(p for p in Path(a.dir).iterdir() if (p / 'main.bin').exists()):
        r = subprocess.run([sys.executable, '-B', __file__, 'sim', d.name, '--code-dir', str(d), '--size', a.size,
                            '--out', str(Path(a.work) / d.name)], capture_output=True, text=True)
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
            rows.append(run_dispatch(cmd))
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
            rows.append(run_dispatch(cmd))
    total = sum(r['time_us'] for r in rows)
    print(json.dumps({'workload': a.workload, 'size': a.size, 'predicted_us': total,
                      'dispatch_us': [r['time_us'] for r in rows], 'dispatches': rows}))


def cmd_predictall(a):
    """predict every workload compiled under DIR (functional paths); one line each, JSON per workload in --work"""
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in Path(a.dir).iterdir() if (p / 'main.bin').exists()):
        r = subprocess.run([sys.executable, '-B', __file__, 'predict', d.name, '--code-dir', str(d), '--size', a.size,
                            '--out', str(work / d.name)], capture_output=True, text=True)
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


def cmd_verify(a):
    """Simulator vs the recorded hardware results (spirv/expected); PENDING where nothing is recorded yet."""
    names = list(WORKLOADS) if a.workload == 'all' else a.workload.split(',')
    rows = [verify_one(n, Path(a.expected), a.layout, a.timing) for n in names]
    for r in rows:
        line = f'{r["status"]:8} {r["workload"]:11}'
        if 'words_differing' in r: line += f' differing words {sum(r["words_differing"].values()):6}'
        if r.get('hardware_matches_reference') is False: line += '  [hardware does not match the numpy reference]'
        if 'timing' in r: line += f'  time measured {r["timing"].get("measured_us")} us predicted {r["timing"].get("predicted_us")} us'
        if r.get('why'): line += f'  ({r["why"]})'
        print(line, flush=True)
    from collections import Counter
    c = Counter(r['status'] for r in rows)
    print(json.dumps({'summary': dict(c)}))
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
