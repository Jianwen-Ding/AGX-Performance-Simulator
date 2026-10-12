#!/usr/bin/env python3
"""AGX coverage suite runner: record the coverage catalog on the GPU (Asahi Linux, direct DRM) and check the
functional simulator / cost model against it.

    coverage.py list     [--area A,B] [--risk safe|may_fault] [--kind functional|timing]
    coverage.py validate [--area A,B] [--limit N]               offline: every case builds, assembles and decodes
    coverage.py record   [--area A,B] [--risky] [--redo] [--limit N] [--match REGEX]   hardware: record expected values
    coverage.py check    [--timing]                             offline: suite_check on coverage/expected (+ timing)
    coverage.py status                                          recorded / faulted / pending per area

The catalog (coverage_catalog.py) has no expected values: `record` runs each case twice on the GPU and stores the
result in the hardware test-suite format (coverage/expected/<area>/<name>.json.gz, same as testsuite/), so
`rust/target/release/suite_check coverage/expected` checks the functional simulator against it. Timing cases are
instruction-bench forms (run_bench.py latency / simd_issue / gpu modes) stored as coverage/timing/<name>.json.

Assembly goes through the Rust assembler (rust/target/release/agx_asm, bit-identical to applegpu's), so neither
command needs applegpu; raw forms the assembler has no syntax for are written `@field=value\\tTEXT` (agx_asm
field edits). `make rust` first.

Recording is resumable: a case whose output file exists is skipped (--redo re-records). Every outcome is appended to
coverage/log.jsonl; faults go to coverage/faults.json and count against the gpu_safety hang budget. `may_fault`
cases (encodings never run, or next to ones that hung the GPU) run only with --risky.
"""
import argparse, json, os, re, struct, subprocess, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
ASM = Path(os.environ.get('AGX_ASM', ROOT / 'rust/target/release/agx_asm'))
COV = Path(os.environ.get('AGX_COVERAGE_DIR', ROOT / 'coverage'))
EXPECTED, TIMING = COV / 'expected', COV / 'timing'


# ---------------------------------------------------------------------------------------------------------------
# Rust assembler backend (replaces applegpu in agx_loop and the helpers the catalog uses)

class Asm:
    def __init__(self):
        if not ASM.exists():
            sys.exit(f'{ASM} not found: run `make rust` (or set AGX_ASM)')
        self.p = subprocess.Popen([str(ASM)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.cache = {}

    def ask(self, line):
        if line not in self.cache:
            self.p.stdin.write(line.replace('\n', ' ') + '\n'); self.p.stdin.flush()
            self.cache[line] = self.p.stdout.readline().rstrip('\n')
        r = self.cache[line]
        if r.startswith('ERROR'):
            raise ValueError(f'{r[6:]} [{line!r}]')
        return r

    def encode(self, text, want=None):
        """agx_loop.encode; '@F=V\\tTEXT' = raw field edit (no round-trip check: the edit's disassembly is reported)"""
        if text.startswith('@'):
            return bytes.fromhex(self.ask(text).split('\t')[0])
        return bytes.fromhex(self.ask(text + ('\t' + want if want else '')))

    def info(self, code):
        """first instruction of code: (name, size, disassembly, fields) or None"""
        try:
            name, size, dis, f = self.ask('?!' + bytes(code[:16]).hex()).split('\t')
        except ValueError:
            return None
        return name, int(size), dis, [(k, int(v)) for k, v in (x.split('=') for x in f.split())]

    def patch(self, spec, code):
        return bytes.fromhex(self.ask(f'@{spec}\t!{bytes(code).hex()}').split('\t')[0])


class _Desc:
    def __init__(self, info):
        self.name, self.size, self.dis, self.fields = info

    def decode_size(self, n): return self.size
    def disassemble(self, n): return self.dis
    def decode_fields(self, n): return list(self.fields)


class _Shim:
    """the decode half of applegpu's module interface (opcode_to_number / get_instruction_descriptor)"""
    def __init__(self, asm): self.asm = asm
    def opcode_to_number(self, b): return int.from_bytes(bytes(b[:16]).ljust(16, b'\0'), 'little')

    def get_instruction_descriptor(self, n):
        i = self.asm.info(n.to_bytes(16, 'little'))
        return _Desc(i) if i else None


STORE_REF = bytes.fromhex('45112c0e00c01200')   # device_store 0, i32, x, r2, u6_u7, r1, unsigned, 0 (the control's)
ASMB = None


def install():
    """route every encoder the catalog uses through agx_asm"""
    global ASMB
    if ASMB:
        return ASMB
    ASMB = asm = Asm()
    import agx_loop, record_suite as RS, run_bench as B, run_drm_tg_layout as TL, run_drm_atomic as DA, run_agx_timestamps
    import run_drm_cpu_clock as CLK                     # run_bench's clock_read encodes through it
    shim = _Shim(asm)
    agx_loop._tools = lambda: (None, shim)
    for m in (agx_loop, RS, B, TL, CLK):
        if hasattr(m, 'e'):
            m.e = asm.encode
        if hasattr(m, 'enc'):
            m.enc = asm.encode
    agx_loop.encode = asm.encode

    def store(reg, base_uniform, index_reg):
        return asm.patch(f'R={2 * reg},A={2 * base_uniform},O={2 * index_reg}', STORE_REF)

    def store_regbase(reg, base_reg, index_reg):
        return asm.patch(f'R={2 * reg},At=0,A={2 * base_reg},O={2 * index_reg}', STORE_REF)

    def store_like(load_text):
        lf = dict(asm.info(asm.encode(load_text))[3]); rf = dict(asm.info(STORE_REF)[3])
        spec = ','.join(f'{k}={lf[k]}' for k in ('F', 'Fx', 'mask', 's', 'Ou', 'At', 'A', 'O', 'Ot', 'R', 'Rt', 'g') if k in lf and k in rf)
        return asm.patch(spec, STORE_REF)

    def atomic(op, dst, idx, src, ret):
        if op not in range(11):
            raise ValueError('only Mesa-named atomic ops 0..10 may run')
        b = DA.mesa_atomic(op, 2 * dst if ret else 0, 16, 2 * idx, 2 * src, ret).to_bytes(8, 'little')
        i = asm.info(b)
        if not i or i[0] != 'atomic' or i[1] != 8:
            raise ValueError(f'atomic encoding does not decode as atomic: {b.hex()}')
        return b

    agx_loop.store, agx_loop.store_regbase, RS.store_like, DA.atomic = store, store_regbase, store_like, atomic
    return asm


RAW_OK = {bytes.fromhex('7800')}   # threadgroup barrier variant (barrier-0x68-vs-0x78), not in applegpu's tables


def gate(body):
    """record_suite.gate on the Rust decoder: every instruction decodes; nothing undecodable or control-transferring"""
    install()
    p = 0
    while p < len(body):
        if body[p:p + 2] in RAW_OK:
            p += 2; continue
        i = ASMB.info(body[p:p + 16])
        if i is None:
            raise RuntimeError(f'undecodable at {p}: {body[p:p + 8].hex()}')
        if i[0].startswith('TODO') or i[0] in ('trap', 'call', 'ret', 'doorbell'):
            raise RuntimeError(f'refusing {i[0]} at {p}')
        p += i[1]


# ---------------------------------------------------------------------------------------------------------------

def cases(a):
    import coverage_catalog as C
    out = C.catalog()
    if getattr(a, 'area', None):
        areas = set(a.area.split(','))
        out = [c for c in out if c.area in areas]
    if getattr(a, 'risk', None):
        out = [c for c in out if c.risk == a.risk]
    if getattr(a, 'kind', None):
        out = [c for c in out if c.kind == a.kind]
    if getattr(a, 'match', None):
        out = [c for c in out if re.search(a.match, c.id)]
    return out


def path_of(c):
    if c.kind == 'timing':
        return TIMING / f'{c.id.split("/", 1)[1]}.json'
    return EXPECTED / f'{c.id}.json.gz'


def bench_name(c):
    return 'cov_' + c.id.split('/', 1)[1]


def bench_body(c):
    """the run_bench programs for a timing case (registers it as a run_bench OPS entry)"""
    import bench_ops, run_bench as B
    spec = c.build()
    bench_ops.OPS[bench_name(c)] = (spec['template'], spec['class'])
    return spec


def cmd_list(a):
    cs = cases(a)
    for c in cs:
        print(f'{c.id}\t{c.risk}\t{c.kind}\t{c.finding}\t{c.description}')
    by = Counter((c.area, c.risk) for c in cs)
    print(json.dumps({'cases': len(cs), 'by_area': {f'{ar}/{r}': n for (ar, r), n in sorted(by.items())}}), file=sys.stderr)


def cmd_validate(a):
    install()
    import run_bench as B
    fails, n = defaultdict(list), 0
    cs = cases(a)
    if a.limit:
        per = defaultdict(int); keep = []
        for c in cs:
            per[c.area] += 1
            if per[c.area] <= a.limit: keep.append(c)
        cs = keep
    for c in cs:
        n += 1
        try:
            for t, want in c.texts:
                ASMB.encode(t, want)
            if c.kind == 'timing':
                bench_body(c)
                B.straight(bench_name(c)); B.loop(bench_name(c), 8, 16, 10)
            else:
                item = c.build()
                if len(item) not in (9, 11):
                    raise ValueError(f'tuple of {len(item)}')
                cat, name, desc, body, lanes, wg, regs, shared, table = item[:9]
                if f'{cat}/{name}' != c.id:
                    raise ValueError(f'id {c.id} != {cat}/{name}')
                if not (0 < wg <= 1024 and lanes > 0 and len(body) % 2 == 0 and len(table) < 1 << 22):
                    raise ValueError(f'bad launch: lanes {lanes} wg {wg} table {len(table)}')
                gate(body)
        except Exception as ex:
            fails[c.area].append((c.id, str(ex)[:300]))
    for area, fs in sorted(fails.items()):
        for cid, msg in fs[:a.show]:
            print(f'FAIL {cid}: {msg}')
        if len(fs) > a.show:
            print(f'... {len(fs) - a.show} more in {area}')
    total = sum(len(f) for f in fails.values())
    print(json.dumps({'validated': n, 'failed': total, 'failed_by_area': {k: len(v) for k, v in sorted(fails.items())}}))
    return 1 if total else 0


def log(rec):
    COV.mkdir(parents=True, exist_ok=True)
    rec['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with open(COV / 'log.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def faults():
    p = COV / 'faults.json'
    return json.loads(p.read_text()) if p.exists() else {}


def cmd_record(a):
    install()
    import gpu_safety, record_suite as RS, run_bench as B
    cs = [c for c in cases(a) if a.risky or c.risk == 'safe']
    known_faults = faults()
    todo = [c for c in cs if a.redo or (not path_of(c).exists() and c.id not in known_faults)]
    if a.limit:
        todo = todo[:a.limit]
    print(json.dumps({'selected': len(cs), 'to_record': len(todo)}), flush=True)
    gpu_safety.require_budget(1)
    scratch = Path(os.environ.get('AGX_SCRATCH', '/scratch/apple-m2-playground')) / 'coverage'
    R = B.Runner(scratch / 'functional')
    for c in todo:
        gpu_safety.require_budget(1)
        try:
            if c.kind == 'timing':
                bench_body(c)
                out = scratch / 'timing' / bench_name(c)
                argv = sys.argv
                sys.argv = ['run_bench.py', '--ops', bench_name(c), '--output', str(out)] + (['--modes', a.modes] if a.modes else [])
                try:
                    B.main()
                finally:
                    sys.argv = argv
                rec = json.loads((out / 'record.json').read_text())
                if not rec.get('complete'):
                    raise RuntimeError(rec.get('error', 'incomplete'))
                res = rec['ops'][bench_name(c)]
                res.update({'id': c.id, 'form': c.description, 'finding': c.finding, 'trips': rec['trips'], 'steps': rec['steps'],
                            'recorded_utc': rec['utc']})
                TIMING.mkdir(parents=True, exist_ok=True)
                path_of(c).write_text(json.dumps(res, indent=1) + '\n')
                log({'case': c.id, 'state': 'recorded'})
            else:
                item = c.build()
                cat, name, desc, body, n, wg, regs, shared, table = item[:9]
                oracle = item[9] if len(item) > 9 else None
                dims = item[10] if len(item) > 10 else None
                gate(body)
                det = RS.record(R, EXPECTED, cat, name, desc, body, n, wg, regs, shared, table, oracle, dims)
                log({'case': c.id, 'state': 'recorded', 'deterministic': det})
        except Exception as ex:
            known_faults[c.id] = {'error': str(ex)[:500], 'risk': c.risk, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            (COV / 'faults.json').write_text(json.dumps(known_faults, indent=1, sort_keys=True) + '\n')
            log({'case': c.id, 'state': 'fault', 'error': str(ex)[:300]})
            if 'budget' in str(ex).lower():
                raise


def predict_timing(c, spec, mode):
    """model cycles per step for one bench mode, in run_bench's units (cycles from the reference iadd = 2 per step)"""
    import cost_hw, run_bench as B
    ch, count, lx = B.MODES[mode]
    groups = 1 if mode != 'gpu' else 4
    trips = 1000
    a = B.loop(bench_name(c), ch, 16, trips); b = B.loop(bench_name(c), ch, 64, trips)
    return cost_hw.predict(a, b, trips * 48, groups)   # per instruction step, whatever the chain count


def cmd_check(a):
    exe = ROOT / 'rust/target/release/suite_check'
    rc = 0
    if EXPECTED.exists():
        rc = subprocess.run([str(exe), str(EXPECTED)] + (['-v'] if a.verbose else [])).returncode
    else:
        print('no coverage/expected yet: run `coverage.py record` on the GPU first')
    if a.timing and TIMING.exists():
        install()
        import coverage_catalog as C
        by_id = {c.id: c for c in C.catalog() if c.kind == 'timing'}
        rows = []
        for p in sorted(TIMING.glob('*.json')):
            r = json.loads(p.read_text())
            c = by_id.get(r['id'])
            if not c:
                continue
            spec = bench_body(c)
            for mode in ('latency', 'simd_issue'):
                m = r.get(mode, {}).get('cycles_per_step')
                if m is None:
                    continue
                try:
                    pr = predict_timing(c, spec, mode)
                except Exception as ex:
                    pr = None
                rows.append((c.id, mode, m, pr))
        errs = [abs(pr - m) / m for _, _, m, pr in rows if pr is not None and m]
        for cid, mode, m, pr in rows:
            if a.verbose or pr is None or abs(pr - m) > 0.1 * m:
                print(f'{cid}\t{mode}\tmeasured {m:.3f}\tmodel {"-" if pr is None else f"{pr:.3f}"}')
        if errs:
            errs.sort()
            print(json.dumps({'timing_rows': len(rows), 'median_rel_error': errs[len(errs) // 2], 'within_10pct': sum(e <= 0.1 for e in errs),
                              'model_unknown': sum(pr is None for *_, pr in rows)}))
    return rc


def cmd_status(a):
    import coverage_catalog as C
    fs = faults()
    tab = defaultdict(Counter)
    for c in C.catalog():
        st = 'recorded' if path_of(c).exists() else 'fault' if c.id in fs else 'pending'
        tab[c.area][(c.risk, st)] += 1
    print(f'{"area":<14}{"safe rec":>10}{"safe pend":>10}{"risky rec":>10}{"risky pend":>11}{"faults":>8}')
    for area, ct in sorted(tab.items()):
        print(f'{area:<14}{ct["safe", "recorded"]:>10}{ct["safe", "pending"]:>10}{ct["may_fault", "recorded"]:>10}'
              f'{ct["may_fault", "pending"]:>11}{ct["safe", "fault"] + ct["may_fault", "fault"]:>8}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('list', 'validate', 'record', 'check', 'status'):
        p = sub.add_parser(name)
        if name in ('list', 'validate', 'record'):
            p.add_argument('--area'); p.add_argument('--match', help='regex on case ids')
            p.add_argument('--kind', choices=['functional', 'timing'])
        if name == 'list':
            p.add_argument('--risk', choices=['safe', 'may_fault'])
        if name in ('validate', 'record'):
            p.add_argument('--limit', type=int, default=0, help='validate: per area; record: total')
        if name == 'validate':
            p.add_argument('--show', type=int, default=10, help='failures shown per area')
        if name == 'record':
            p.add_argument('--risky', action='store_true', help='also run may_fault cases')
            p.add_argument('--redo', action='store_true', help='re-record cases that already have a file')
            p.add_argument('--modes', help='timing modes (default run_bench: latency,simd_issue,gpu)')
        if name == 'check':
            p.add_argument('--timing', action='store_true', help='also compare timing cases with the cost model')
            p.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    sys.exit({'list': cmd_list, 'validate': cmd_validate, 'record': cmd_record, 'check': cmd_check, 'status': cmd_status}[a.cmd](a) or 0)


if __name__ == '__main__':
    main()
