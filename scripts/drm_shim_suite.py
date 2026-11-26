#!/usr/bin/env python3
"""The hardware-recorded test suite (testsuite/, recorded on the M2 Max through direct DRM by
scripts/record_suite.py with tests/drm_compute.c) replayed through the simulator's ioctl shim (rust/drm_shim):
the same drm_compute client, the same arguments and environment as the recording, with
LD_PRELOAD=libagx_drm_shim.so serving /dev/dri/renderD128. Each case's readback and table are compared word for
word with what the hardware produced (the deterministic parts, as suite_check does).

  drm_shim_suite.py [--client tests/drm_compute binary] [--shim libagx_drm_shim.so] [-k SUBSTR] [-j N] [-v]

Builds the client (cc) and the shim (cargo) when they are missing. No GPU is used.
"""
import argparse
import ast
import base64
import concurrent.futures as cf
import gzip
import json
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SHIM_DIR = os.path.join(ROOT, "rust", "drm_shim")


def shared_word(nbytes):
    return 0x4D | 1 << 8 | 0x24 << 10 | ((nbytes // 256) & 255) << 24


def words(b64):
    b = base64.b64decode(b64)
    return list(struct.unpack("<%dI" % (len(b) // 4), b))


def run_case(path, a):
    case = json.load(gzip.open(path))
    lay = ast.literal_eval(case["layout"]) if isinstance(case["layout"], str) else case["layout"]
    inp = ast.literal_eval(case["input"]) if isinstance(case["input"], str) else case["input"]
    exp = ast.literal_eval(case["expect"]) if isinstance(case["expect"], str) else case["expect"]
    launch = case["launch"]
    grid, wg = launch["grid"], launch["wg"]
    count = lay["uniforms"]["u5"] if isinstance(lay.get("uniforms"), dict) and "u5" in lay["uniforms"] else grid[0]
    table = words(inp["table"])
    with tempfile.TemporaryDirectory() as t:
        code = bytes.fromhex(case["code"])
        open(os.path.join(t, "c.bin"), "wb").write(code + bytes(16))
        open(os.path.join(t, "t.bin"), "wb").write(struct.pack("<%dI" % len(table), *table))
        env = dict(os.environ, AGX_SHIM_DRM="1", LD_PRELOAD=a.shim, AGX_LOCAL_X=str(wg[0]),
                   AGX_TABLE_OUT=os.path.join(t, "t.out"))
        if grid[1] > 1 or grid[2] > 1 or wg[1] > 1 or wg[2] > 1 or grid[0] != (count + 31) // 32 * 32:
            env.update({"AGX_GLOBAL_X": str(grid[0]), "AGX_GLOBAL_Y": str(grid[1]), "AGX_GLOBAL_Z": str(grid[2]),
                        "AGX_LOCAL_Y": str(wg[1]), "AGX_LOCAL_Z": str(wg[2])})
        if launch.get("shared_bytes"):
            env["AGX_SHARED_WORD"] = str(shared_word(launch["shared_bytes"]))
        if a.no_timing:
            env["AGX_SHIM_TIMING"] = "0"
        r = subprocess.run([a.client, "/dev/dri/renderD128", os.path.join(t, "c.bin"), str(count),
                            os.path.join(t, "rb"), "1", str(launch["regs"]), os.path.join(t, "t.bin")],
                           capture_output=True, text=True, timeout=a.timeout, env=env)
        lines = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")]
        rep = [x for x in lines if "rep" in x]
        if r.returncode or len(rep) != 1:
            return "ERROR", (r.stdout + r.stderr)[-300:].replace("\n", " "), None
        if not all(rep[0]["fw_ts"]):
            return "FAULT", r.stderr.strip()[-300:].replace("\n", " "), None
        rounded = (count + 31) // 32 * 32
        n = 8 + rounded + 8 + 16
        out = list(struct.unpack("<%dI" % n, open(os.path.join(t, "rb"), "rb").read()[:4 * n]))
        tab = list(struct.unpack("<%dI" % len(table), open(os.path.join(t, "t.out"), "rb").read()[:4 * len(table)]))
    bad = []
    det = case.get("deterministic", True)
    if case.get("deterministic_out", det) and "out" in exp:
        want = words(exp["out"])
        bad += [("out", i, g, w) for i, (g, w) in enumerate(zip(out, want)) if g != w]
    if case.get("deterministic_table", det) and "table" in exp:
        want = words(exp["table"])
        # order-dependent words (contended atomics) are not compared: ignore_table [start, end, step], as suite_check
        skip = set()
        for s0, e0, st in case.get("ignore_table") or []:
            skip.update(range(s0, min(e0, len(want)), max(st, 1)))
        bad += [("table", i, g, w) for i, (g, w) in enumerate(zip(tab, want)) if g != w and i not in skip]
    ns = rep[0]["fw_ts"][1] - rep[0]["fw_ts"][0]
    hw = case.get("fw_ns")
    timing = "sim %.1f us (hw %s us)" % (ns / 24.0, "/".join("%.1f" % (x / 1000) for x in hw) if hw else "?")
    if bad:
        return "FAIL", "%d words differ, first %s" % (len(bad), [(k, i, hex(g), hex(w)) for k, i, g, w in bad[:3]]), ns
    return "PASS", timing, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default=os.path.join(ROOT, "tests", "build", "drm_compute"))
    ap.add_argument("--shim", default=os.path.join(SHIM_DIR, "target", "release", "libagx_drm_shim.so"))
    ap.add_argument("-k", action="append")
    ap.add_argument("-j", type=int, default=os.cpu_count())
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-timing", action="store_true", help="functional only (AGX_SHIM_TIMING=0)")
    ap.add_argument("--json")
    a = ap.parse_args()
    if not os.path.exists(a.shim):
        subprocess.run(["cargo", "build", "--release"], cwd=SHIM_DIR, check=True)
    if not os.path.exists(a.client):
        os.makedirs(os.path.dirname(a.client), exist_ok=True)
        subprocess.run(["cc", "-O2", "-I", os.path.join(SHIM_DIR, "include"), "-I", "/usr/include/drm", "-o", a.client,
                        os.path.join(ROOT, "tests", "drm_compute.c"), "-lpthread"], check=True)
    cases = []
    for cat in sorted(os.listdir(os.path.join(ROOT, "testsuite"))):
        d = os.path.join(ROOT, "testsuite", cat)
        if os.path.isdir(d):
            cases += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".json.gz")]
    if a.k:
        cases = [c for c in cases if any(k in c for k in a.k)]
    res = {}
    with cf.ThreadPoolExecutor(a.j) as ex:
        futs = {ex.submit(run_case, c, a): c for c in cases}
        for fu in cf.as_completed(futs):
            c = futs[fu]
            try:
                st, detail, ns = fu.result()
            except Exception as e:  # noqa: BLE001
                st, detail, ns = "ERROR", repr(e)[-300:], None
            res[os.path.relpath(c, ROOT)] = {"status": st, "detail": detail, "sim_ticks": ns}
            if a.v or st != "PASS":
                print("%-6s %s  %s" % (st, os.path.relpath(c, ROOT), detail), flush=True)
    counts = {}
    for r in res.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("----", ", ".join("%s %d" % kv for kv in sorted(counts.items())), "of", len(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    sys.exit(0 if counts.get("PASS", 0) == len(res) else 1)


main()
