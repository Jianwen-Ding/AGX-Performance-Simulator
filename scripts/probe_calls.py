#!/usr/bin/env python3
"""call / ret probes (owner-approved), staged so control can never leave the program.

  probe_calls.py run --stage 1 [--only NAME] [--dry]      GPU (CONFIRM through the playground's make target)
  probe_calls.py analyse [DIR|BUNDLE]                      offline

Every stage runs the *variant itself* in agx_run first (the simulator models call / ret from applegpu's pseudocode and
refuses any target outside the program), plus the surrogate program through probe_variants. The link-register base of
the simulator is AGX_CALL_BASE, default 0x2000 as stage 1 measured it.

Stage 1, 6-byte `call off` with a target inside the program, no ret:
  C1a_next      call +6 (the next instruction); the link register r1 is stored
  C1b_skip      call +12 over a marker write: does the jump happen?
  C1c_pad       C1a with 12 bytes of extra code before it: does the link move by 12 (linear in pc)?
"""
import argparse, json, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import probe_variants as PV                               # noqa: E402

N, WG, REGS = 32, 32, 64
OFF_LINK, OFF_AFTER, OFF_SKIP = 3000, 3100, 3200          # table word offsets (+ lane)
RAW = Path('/scratch/apple-m2-playground/call-probes')
BUNDLE = ROOT.parent / 'apple-documentation-playground/results/2026-10-17-call-ret/records.json.gz'
MASK48 = (1 << 48) - 1
LINK_BASE = 0x2000                                         # measured on G14C by stage 1 (C1a-C1c)


def call6(off):
    """call off: 0xC010 | off << 16 (6 bytes, pc-relative, applegpu CallInstructionDesc)"""
    return (0xC010 | (off & 0xffffffff) << 16).to_bytes(6, 'little')


def site_offset():
    import run_drm_compute as drm
    return drm.control_main()[1]


def stage1(pad_bytes=0, skip=False, name=None):
    import record_suite as RS
    import coverage as cov
    st = RS.agx_loop.store
    pre = RS.e('iadd r20, r1, 0')
    for reg, off in ((21, OFF_LINK), (22, OFF_AFTER), (25, OFF_SKIP)):
        pre += RS.mov(27, off) + RS.e(f'iadd r{reg}, r20, r27')
    pre += RS.mov(24, 0) + RS.mov(26, 0x33330000) + RS.e('iadd r26, r26, r20')
    pre += RS.mov(28, 0) * (pad_bytes // 6)                  # padding: 6-byte mov_imm of an unused register
    call_at = site_offset() + len(pre)                       # byte offset of the call in the whole program
    link_sim = LINK_BASE + call_at + 6                       # what the simulator model stores
    mid = RS.mov(24, 0x0bad0001) if skip else b''            # the instruction the call should jump over
    off = 6 + len(mid)
    suf = RS.e('iadd r23, r1, 0') + RS.e('iadd r1, r20, 0')
    suf += st(23, 8, 21) + st(26, 8, 22) + st(24, 8, 25) + RS.e('wait 0') + RS.e('iadd r2, r26, 0')
    surrogate = RS.mov(1, link_sim)
    t = [0] * 3300
    words = 8 + N + 8 + 16

    def common(o, tb):
        return (len(o) == words and o[:8] == [0xcdcdcdcd] * 8 and o[8:8 + N] == [0x33330000 + i for i in range(N)]
                and o[-16:] == [0xa5a50000 + i for i in range(16)]
                and [tb[OFF_AFTER + i] for i in range(N)] == [0x33330000 + i for i in range(N)])

    def sur_oracle(o, tb):      # surrogate: r1 = link_sim, no jump, so the skipped marker is written
        return (common(o, tb) and [tb[OFF_LINK + i] for i in range(N)] == [link_sim] * N
                and [tb[OFF_SKIP + i] for i in range(N)] == ([0x0bad0001] * N if skip else [0] * N))

    def var_oracle(o, tb):      # variant in the simulator: jump taken, link = pc + 6, marker skipped
        return (common(o, tb) and [tb[OFF_LINK + i] for i in range(N)] == [link_sim] * N
                and [tb[OFF_SKIP + i] for i in range(N)] == [0] * N)

    # the call is the same 6 bytes as the surrogate; when skip is set, mid sits between it and the target, so the body
    # order is call, mid, suffix. The surrogate program therefore needs mid too (it just falls through).
    return PV.Probe(name, pre, surrogate + mid, call6(off) + mid, suf, MASK48, N, WG, REGS, t, sur_oracle,
                    variant_oracle=var_oracle)


def probes(stage):
    if stage == 1:
        return [stage1(0, False, 'C1a_next'), stage1(0, True, 'C1b_skip'), stage1(12, False, 'C1c_pad')]
    raise SystemExit(f'stage {stage} is not defined yet')


def analyse(src):
    import gzip
    src = Path(src)
    recs = ([json.loads(p.read_text()) for p in sorted(src.glob('*.json'))] if src.is_dir()
            else sorted(json.load(gzip.open(src, 'rt'))['records'], key=lambda r: r['name']))
    rows = []
    for r in recs:
        if 'fault' in r:
            rows.append({'probe': r['name'], 'fault': r['fault'][:160]}); continue
        t1, out = r['table_after'], r['out'][8:8 + N]
        link = sorted({t1[OFF_LINK + i] for i in range(N)})
        rows.append({'probe': r['name'], 'deterministic': r['deterministic'],
                     'link_values': [f'{x:#x}' for x in link][:4],
                     'continued_after_call': sum(t1[OFF_AFTER + i] == 0x33330000 + i for i in range(N)),
                     'skip_marker_written': sum(t1[OFF_SKIP + i] != 0 for i in range(N)),
                     'harness_store_ok': sum(out[i] == 0x33330000 + i for i in range(N))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--stage', type=int, required=True); r.add_argument('--only')
    r.add_argument('--dry', action='store_true'); r.add_argument('--out', type=Path, default=RAW)
    s = sub.add_parser('analyse'); s.add_argument('dir', type=Path, nargs='?', default=BUNDLE)
    a = ap.parse_args()
    if a.cmd == 'run':
        PV.cov.install()
        ps = probes(a.stage)
        if a.only:
            ps = [p for p in ps if p.name == a.only]
        PV.run(ps, a.out, dry=a.dry)
    else:
        for row in analyse(a.dir):
            print(json.dumps(row))


if __name__ == '__main__':
    main()
