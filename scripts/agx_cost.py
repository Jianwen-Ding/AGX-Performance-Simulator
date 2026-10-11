#!/usr/bin/env python3
"""AGX (G14C) cycle cost model and simulator. No GPU access.

    python3 scripts/agx_cost.py code.bin [--groups G] [--annotate] [--units real|ref] [--allow-unknown]

Units. Hardware numbers in this repository are in *reference units* (a dependent 8-byte
`iadd x, x, 1` step = 2 units). Against the GPU's DVFS table every inferred clock is low by
1.049, so 1 unit = 1.049 real cycles. The simulator converts every parameter to real cycles
first, rounds latencies and issue intervals to whole real cycles, and simulates in real cycles.
`--units ref` divides the result by 1.049 for comparison with stored measurements.

Pipeline
  decode   : applegpu decodes each instruction; registers come from the operand objects
             (Reg16 / Reg32 / Reg64 / RegisterTuple, in 16-bit halves); atomics use named fields.
             Every descriptor maps to an explicit class; an instruction without one raises
             ValueError (or, with --allow-unknown, is costed as a 1-cycle ALU op and reported).
  trace    : the dynamic path for one SIMD group from applegpu's emulator. Uniforms and memory
             read as zero and instructions it cannot execute are skipped; when a branch or
             loop condition depends on a uniform or on a skipped instruction's result the trace
             is flagged (TRACE_WARNINGS) because the trip count is then unknown.
  simulate : one core running G identical SIMD groups, cycle by cycle (real cycles):
             * in-order issue, <= 1 instruction per group per cycle; an instruction occupies its
               group for its issue interval (multi-cycle classes block the whole group: the
               "group" rule chosen in the cost-model finding);
             * a source is ready `latency` after its producer issued; an 8-entry per-group result
               cache: a value evicted from it is readable only RF_WRITEBACK after its producer;
             * per core and cycle: <= 4 issues, <= 4 32-bit register-operand reads, and per
               execution unit at most its measured throughput (token buckets);
             * loads/stores/atomics hold a scoreboard slot until done; `wait s` drains slot s;
             * a 32-bit write right after a 16-bit write of the same register (or the reverse)
               costs 1 cycle; a taken backward jump costs BRANCH extra cycles.
  stats    : per pc: executions, `issue` (cycles it occupied its group) and `stall` (cycles
             its group was free but it could not issue: dependencies, units, ports, waits).
Parameters: results/2026-10-11-instruction-bench/bench.json.gz (per-class latency, issue interval,
per-core throughput) and results/2026-10-11-cost-model/calibration.json (loop, get_sr, memory).
Classes marked UNMEASURED reuse the nearest measured cost and are listed in the output.
"""
import argparse, contextlib, gzip, io, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agx_loop

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO/'results/2026-10-11-instruction-bench/bench.json.gz'
CALIB = REPO/'results/2026-10-11-cost-model/calibration.json'
REAL_CYCLES_PER_UNIT = 1.049
ISSUE_PER_CORE = 4
READ_PORTS = 4.0          # 32-bit register-operand reads per core per real cycle
RESULT_CACHE = 8          # per SIMD group: the last 8 written registers read at producer latency
RF_WRITEBACK = 18         # real cycles from producer issue until an evicted value is readable (17 units)
RF_WRITEBACK16 = 18

ALU = {   # descriptor name -> benchmark row
    'iadd': 'iadd', 'imadd': 'imadd', 'bitop': 'and', 'bfi': 'bfi', 'bfeil': 'bfeil', 'extr': 'extr',
    'shlhi': 'shlhi', 'shrhi': 'shrhi', 'asr': 'asr', 'asrh': 'asrh', 'bitrev': 'bitrev', 'popcount': 'popcount',
    'ffs': 'ffs', 'intl': 'intl', 'iunop': 'bitrev', 'icmpsel': 'icmpsel', 'fcmpsel': 'fcmpsel', 'mov_imm': 'mov_imm',
    'fadd32': 'fadd32', 'fmul32': 'fmul32', 'fmadd32': 'fmadd32', 'fadd16': 'fadd16', 'fmul16': 'fmul16',
    'fmadd16': 'fmadd16', 'convert': 'convert_f2i', 'floor': 'floor', 'ceil': 'floor', 'trunc': 'floor',
    'rint': 'rint', 'rcp': 'rcp', 'rsqrt': 'rsqrt', 'rsqrt_special': 'rsqrt_special', 'log2': 'log2', 'exp2': 'exp2',
    'sin_pt_1': 'sin_pt_1', 'sin_pt_2': 'sin_pt_2', 'funop': 'floor', 'dfdx': 'sin_pt_1', 'dfdy': 'sin_pt_1',
    'simd_shuffle': 'simd_shuffle', 'simd_shuffle_down': 'simd_shuffle_down', 'simd_shuffle_up': 'simd_shuffle_down',
    'simd_shuffle_xor': 'simd_shuffle_xor', 'simd_shuffle_rotate_up': 'simd_shuffle_down', 'simd_shuf_op': 'simd_shuffle',
    'quad_shuffle': 'quad_shuffle', 'quad_shuffle_down': 'quad_shuffle', 'quad_shuffle_up': 'quad_shuffle',
    'quad_shuffle_xor': 'quad_shuffle', 'quad_shuffle_rotate_up': 'quad_shuffle',
    'icmp_ballot': 'icmp_ballot', 'fcmp_ballot': 'fcmp_ballot', 'icmp_quad_ballot': 'icmp_ballot', 'fcmp_quad_ballot': 'fcmp_ballot',
    'get_sr': 'get_sr', 'TODO.sr_thing': 'get_sr',
}
UNMEASURED_ALU = {'iunop', 'ceil', 'trunc', 'funop', 'dfdx', 'dfdy', 'simd_shuffle_up', 'simd_shuffle_rotate_up', 'simd_shuf_op',
                  'quad_shuffle_down', 'quad_shuffle_up', 'quad_shuffle_xor', 'quad_shuffle_rotate_up',
                  'icmp_quad_ballot', 'fcmp_quad_ballot', 'TODO.sr_thing'}
REDUCE = re.compile(r'^(simd|quad)_(prefix_)?(iadd|fadd|fmul|and|or|xor|min|max)')
LOADS = {'device_load', 'device_load.TODO', 'threadgroup_load', 'stack_load', 'async_load', 'texture_sample',
         'texture_load', 'iter', 'iterproj', 'TODO.ldcf', 'TODO.ld/st_tile'}
STORES = {'device_store', 'device_store.TODO', 'threadgroup_store', 'stack_store', 'uniform_store', 'image_write',
          'image_write_block', 'async_store', 'st_var', 'st_var_final', 'tex_state_store', 'sampler_state_store'}
MEASURED_MEMORY = {'device_load', 'device_store', 'threadgroup_load', 'threadgroup_store'}
SPACE = {'device_load': 'device', 'device_load.TODO': 'device', 'device_store': 'device', 'device_store.TODO': 'device',
         'atomic': 'device', 'threadgroup_load': 'tg', 'threadgroup_store': 'tg', 'threadgroup_atomic': 'tg',
         'stack_load': 'stack', 'stack_store': 'stack'}
CTL_AS_ALU = True
CROSS_BUMP = True
ARBITRATION = 'hash'       # 'rr': round robin by tick; 'gto': greedy-then-oldest
SFU_PIPE = True          # sfupairs.json: per-group transcendental pipe (spacing, slow stage, in-order writeback)
SFU_SPACING = 2.0        # cycles between transcendental issues of one SIMD group
PIPE_CLASSES = ('sfu', 'imadbf')   # sfuimad.json: integer multiply / bit-field / convert share the in-order writeback
CORE_IMAD = 'fast'       # core unit used by the imadbf class ('fast' = shared with fast transcendentals, or 'own')
SFU_GROUP_BLOCK = 2.0    # cycles a transcendental occupies its SIMD group (not the op's full issue interval)
SFU_WB_GAP = 2.0         # cycles between transcendental writebacks of one SIMD group (in order)
SFU_SLOW = {'rcp': 3.0, 'rsqrt': 4.0, 'rsqrt_special': 4.0}   # cycles in the slow stage
# core-level transcendental pipe shared by the SIMD groups of a core (sfupairs.json, sfumix.json, sfurepeat.json):
# rcp/rsqrt use a slow unit, the others a fast one, at the benchmark's 4-group costs (rcp 1.6, rsqrt 1.02, others
# 0.92 cycles); writebacks in order 1 cycle apart. Replaces the shared 'sfu' token bucket. A 12-point grid fit
# (sfu-coregrid.json, rsqrt 2.3) fitted the pair probes better but broke the rsqrt benchmark row (+126%).
# Not captured: rcp with 8 chains vs 4 (232 vs 197 units, repeatable), rsqrt+rcp (258 vs 191 predicted) and
# run-to-run variation of mixed-latency pairs (log2+floor 207 vs 232).
CORE_SFU = {'wb_gap': 1.0, 'slow': {'rcp': 1.6, 'rsqrt': 1.02, 'rsqrt_special': 1.02}, 'fast': 0.92}
CONTROL = {'if_icmp', 'if_fcmp', 'else_icmp', 'else_fcmp', 'while_icmp', 'while_fcmp', 'pop_exec', 'push_exec',
           'jmp_exec_any', 'jmp_exec_none', 'stop', 'trap', 'TODO.unk51', 'no_var'}
BARRIERS = {'threadgroup_barrier', 'memory_barrier', 'wait_pix', 'signal_pix', 'TODO.unkE800'}
GROUP_BLOCKING = {'simd_iadd', 'simd_prefix_iadd', 'simd_fadd', 'simd_min.u', 'quad_iadd'}
UNIT_GROUPS = [['iadd', 'isub', 'iadd_lsl2', 'iadd_imm', 'iadd_cache'], ['imadd', 'imul', 'imadd_cache'],
               ['fadd32', 'fadd32_imm', 'fadd32_cache', 'fadd32_cache2'], ['fmul32', 'fmul32_cache'],
               ['fmadd32', 'fmadd32_imm', 'fmadd32_cache'], ['and', 'xor']]
# execution units shared by several instruction classes (sfumix.json: different transcendentals contend
# across SIMD groups like identical ones). One bucket per unit, refilled at 1 unit-cycle per cycle; an
# instruction consumes 1 / (its measured instructions per cycle per core). Single-class loops are unchanged.
SHARED_UNITS = {'sfu': ['rcp', 'rsqrt', 'rsqrt_special', 'log2', 'exp2', 'sin_pt_1', 'sin_pt_2', 'floor', 'rint'],
                'imadbf': ['imadd', 'imul', 'imadd_cache', 'bfi', 'bfeil', 'extr', 'shlhi', 'shrhi', 'asr', 'asrh',
                           'bitrev', 'popcount', 'ffs', 'intl', 'convert_f2i', 'convert_i2f']}
TRACE_WARNINGS = []


def regs_of(op):
    """Operand object -> set of 16-bit half indices (empty for non-register operands)."""
    t = type(op).__name__
    if t == 'Reg16': return {op.n}
    if t == 'Reg32': return {2*op.n, 2*op.n+1}
    if t == 'Reg64': return set(range(2*op.n, 2*op.n+4))
    if t == 'RegisterTuple': return set().union(*[regs_of(r) for r in op.registers]) if op.registers else set()
    return set()


def is_stack_adjust(n):
    """Mesa's stack_adjust (agx_pack.c AGX_OPCODE_STACK_ADJUST): opcode 0xb5 shared with stack_store."""
    fixed = 0x10100b5 | 1 << 26 | 2 << 36 | 1 << 47
    size_bits = 0xf << 20 | 0xf << 32 | 0xff << 56
    return n & ((1 << 64)-1) & ~size_bits == fixed


class Params:
    def __init__(self, bench=BENCH, calib=CALIB, load_latency=None):
        K = REAL_CYCLES_PER_UNIT
        b = json.loads(gzip.open(bench).read())['ops']
        self.rows = {}
        for k, v in b.items():
            self.rows[k] = {'lat': v['latency']['cycles_per_step']*K,
                            'ii': max(1.0, v['simd_issue']['cycles_per_step']*K),
                            'ipc': v['gpu']['lanes_per_cycle_per_core']/32/K, 'tmpl': v['template']}
        self.cal = json.loads(Path(calib).read_text()) if Path(calib).exists() else {}
        c = self.cal
        self.rows['get_sr'] = {'lat': c.get('get_sr_latency', 3)*K, 'ii': max(1.0, c.get('measured', {}).get('get_sr_issue_cycles', 2)*K),
                               'ipc': 4.0, 'tmpl': 'get_sr'}
        self.unit_cap = {k: v['ipc'] for k, v in self.rows.items()}
        self.unit_bucket = {k: k for k in self.rows}
        self.shared_of = {k: u for u, ks in SHARED_UNITS.items() for k in ks}
        if c.get('merge_sfu_imad'): self.shared_of = {k: 'sfu' for k in self.shared_of}
        for grp in UNIT_GROUPS:
            best = max(self.rows[k]['ipc'] for k in grp if k in self.rows)
            for k in grp:
                if k in self.rows: self.unit_cap[k] = best; self.unit_bucket[k] = grp[0]
        # device load: fast when no ALU instruction issued since the last wait, slow otherwise; ALU
        # instructions issued while a load is pending defer its completion (load-alu-grid.json)
        self.load_fast = (c.get('device_load_fast_ref', 41.6))*K
        self.load_latency = load_latency or (c.get('device_load_slow_ref', 62.0))*K
        # completion is at least this long after the last ALU instruction issued while the load is pending
        self.defer_fast = c.get('device_load_defer_fast_ref', 31.0)*K
        self.defer_slow = c.get('device_load_defer_slow_ref', 56.0)*K
        self.defer_delayed_extra = c.get('device_load_defer_delayed_extra_ref', 5.0)*K
        self.tg_defer = c.get('tg_load_defer_ref', 20.0)*K
        self.load_ii = max(1.0, c.get('device_load_extra_inflight_ref', 2.6)*K)
        g = lambda k, d: c.get(k, d)*K
        self.mem = {   # per memory class: (fast, slow) issue occupancy, latency / completion, in real cycles
            'device_store': {'issue': (g('device_store_fast_issue_ref', 3.1), g('device_store_slow_issue_ref', 35.7)),
                             'done': (g('device_store_done_fast_ref', 18.3), g('device_store_done_slow_ref', 4.7))},
            'threadgroup_load': {'lat': (g('tg_load_fast_ref', 19.4), g('tg_load_slow_ref', 38.4)), 'issue': (g('tg_load_issue_ref', 21.0),)*2},
            'threadgroup_store': {'issue': (g('tg_store_fast_issue_ref', 17.1), g('tg_store_slow_issue_ref', 36.2)), 'done': (0.0, 0.0)}}
        self.space_switch = g('mem_space_switch_ref', 13.3)
        self.quiet = {'load': g('mem_quiet_load_ref', 20.0), 'tg': g('mem_quiet_tg_ref', 20.0), 'device_store': g('mem_quiet_device_store_ref', 33.6)}
        self.wait_idle = g('wait_idle_ref', 4.89)
        self.wait_drain = g('wait_drain_ref', 17.0)
        self.store_after_wait = g('store_after_wait_ref', 10.6)    # storeseries.json, from the previous wait's issue
        self.wait_idle_big = g('wait_idle_big_ref', 10.9)
        self.wait_run_long = c.get('wait_run_long', 31)
        self.branch = c.get('branch_taken_real')        # set by fit_branch() if missing
        self.allow_unknown = False

    def row(self, name):
        return self.rows.get(name)


def decode(code, P):
    _, g = agx_loop._tools()
    out, pc = [], 0
    while pc < len(code):
        n = g.opcode_to_number(code[pc:pc+16]); d = g.get_instruction_descriptor(n)
        if d is None: raise ValueError(f'undecodable instruction at {pc}')
        size = d.decode_size(n); n &= (1 << 8*size)-1
        ops = d.decode_operands(n); name = d.name
        text = ' '.join(str(d.disassemble(n)).split())
        f = dict(d.decode_fields(n))
        regops = [(o, regs_of(o)) for o in ops if regs_of(o)]
        rec = {'pc': pc, 'size': size, 'name': name, 'text': text, 'dst': set(), 'src': set(), 'reads32': 0,
               'unit': None, 'kind': None, 'unmeasured': False, 'slot': 0}
        if name == 'stack_store' and is_stack_adjust(n):
            rec.update(kind='control', name='stack_adjust', text='stack_adjust #%d (words)' % ((n >> 20 & 0xf) | (n >> 32 & 0xf) << 4 | (n >> 56) << 8))
        elif name in LOADS:
            rec.update(kind='load', slot=f.get('g', 0), unmeasured=name not in MEASURED_MEMORY)
            if regops: rec['dst'] = regops[0][1]; rec['src'] = set().union(set(), *[r for _, r in regops[1:]])
        elif name in STORES:
            rec.update(kind='store', slot=f.get('g', 0), unmeasured=name not in MEASURED_MEMORY)
            rec['src'] = set().union(set(), *[r for _, r in regops])
        elif name == 'atomic':
            ret = bool(f.get('Rt')); R = f.get('R', 0) | (f.get('Rx', 0) << 6)
            width = 4 if f.get('op') == 3 else 2                     # cmpxchg sources a register pair
            S = f.get('S', 0)
            dst = {R, R+1} if ret else set()
            src = set().union(set(), *[r for _, r in regops]) - dst | set(range(S, S+width))
            rec.update(kind='load' if ret else 'store', slot=f.get('g', 0), dst=dst, src=src, unmeasured=True)
        elif name == 'threadgroup_atomic':
            ret = bool(f.get('Ra')); R = f.get('R', 0) | (f.get('Rx', 0) << 6)
            dst = {R, R+1} if ret else set()
            src = set().union(set(), *[r for _, r in regops]) - dst
            rec.update(kind='tg_atomic', dst=dst, src=src, unmeasured=True)
        elif name == 'wait':
            rec.update(kind='wait', slot=int(str(ops[0])) if ops else 0)
        elif name in CONTROL:
            rec['kind'] = 'control'
            rec['src'] = set().union(set(), *[r for o, r in regops if not (type(o).__name__ == 'Reg16' and o.n == 0)])
            # exec-stack instructions read and write the r0l nesting counter: back-to-back push/pop form a dependency
            # chain of control_latency (ctlspacer.json: 1.9 units per instruction alone, hidden by interleaved ALU work)
            if name not in ('jmp_exec_any', 'jmp_exec_none', 'stop', 'trap'): rec['src'] |= {0}
            rec['dst'] = {0} if name not in ('stop', 'trap') else set()
            rec['uniform_src'] = any(type(o).__name__.startswith('UReg') for o in ops)
        elif name in BARRIERS:
            rec.update(kind='barrier', unmeasured=True)
        elif name in ALU or REDUCE.match(name):
            rec['kind'] = 'alu'
            if regops:
                rec['dst'] = regops[0][1]; rec['src'] = set().union(set(), *[r for _, r in regops[1:]])
            if name == 'get_sr': rec['src'] = set()
            wide = {h//2 for o, r in regops[1:] if type(o).__name__ in ('Reg32', 'Reg64') for h in r}
            rec['reads32'] = len(wide)
            rec['cache'] = any('cache' in getattr(o, 'flags', []) for o, _ in regops[1:])
            imm = sum(1 for o in ops if type(o).__name__ == 'Immediate')
            if REDUCE.match(name):
                key = name if P.row(name) else ('quad_iadd' if name.startswith('quad') else
                                                 ('simd_prefix_iadd' if 'prefix' in name else 'simd_iadd'))
                rec['unmeasured'] = not P.row(name)
            else:
                key = ALU[name]
                rec['unmeasured'] = name in UNMEASURED_ALU
                if name == 'iadd' and any(len(r) == 4 for _, r in regops[:1]): key = 'iadd64'
                if name == 'imadd' and any(len(r) == 4 for _, r in regops[:1]): key = 'imadd_wide'
                # operand variant: the immediate row only when at most one 32-bit register is read
                if imm and rec['reads32'] <= 1 and P.row(key+'_imm'): key += '_imm'
                elif rec['cache'] and P.row(key+'_cache'): key += '_cache'
            rec['unit'] = key
        else:
            if not P.allow_unknown:
                raise ValueError(f'agx_cost: no cost class for {name!r} at pc {pc} ({text})')
            rec.update(kind='alu', unit='mov_imm', unmeasured=True)
            if regops: rec['dst'] = regops[0][1]; rec['src'] = set().union(set(), *[r for _, r in regops[1:]])
        out.append(rec); pc += size
    return mark_loops(out, P)


def mark_loops(instrs, P):
    """Tag each instruction with its innermost backward-jump loop size in bytes (None outside loops)
    and each wait with the length of the run of consecutive waits it belongs to (waitruns.json:
    a run of >= 31 consecutive waits costs ~11 units per wait instead of ~4.9)."""
    loops = []
    for r in instrs:
        if r['name'] == 'jmp_exec_any':
            m = re.search(r'pc-(\d+)', r['text'])
            if m:
                start = r['pc']-int(m.group(1)); loops.append((start, r['pc']+r['size']))
    for r in instrs:
        sizes = [b-a for a, b in loops if a <= r['pc'] < b]
        r['loop_bytes'] = min(sizes) if sizes else None
    i = 0
    while i < len(instrs):
        if instrs[i]['name'] == 'wait':
            j = i
            while j < len(instrs) and instrs[j]['name'] == 'wait': j += 1
            for k in range(i, j): instrs[k]['wait_run'] = j-i
            i = j
        else:
            i += 1
    return instrs


def trace(code, active=32, max_steps=500_000):
    """Dynamic pc sequence for one SIMD group. Returns (pcs, skipped instruction names); appends to
    TRACE_WARNINGS when control flow depends on a uniform or on a skipped instruction's result."""
    _, g = agx_loop._tools()
    st = g.CoreState(num_registers=128, uniforms=g.Uniforms(), device_memory=agx_loop._Memory(agx_loop.TABLE_VA, [0]*65536))
    for t in range(g.SIMD_WIDTH):
        st.set_reg16(0, t, 0 if t < active else 1); st.set_reg32(1, t, t)
    st.exec = [t < active for t in range(g.SIMD_WIDTH)]
    pcs, pc, skipped, tainted = [], 0, set(), set()
    TRACE_WARNINGS.clear()
    while pc < len(code):
        n = g.opcode_to_number(code[pc:pc+16]); d = g.get_instruction_descriptor(n)
        size = d.decode_size(n); n &= (1 << 8*size)-1
        pcs.append(pc)
        if len(pcs) > max_steps: raise RuntimeError('trace step limit (non-terminating loop?)')
        if d.name in ('stop', 'trap'): break
        ops = d.decode_operands(n)
        rs = [regs_of(o) for o in ops]
        if d.name in CONTROL and d.name not in ('jmp_exec_any', 'jmp_exec_none', 'pop_exec'):
            uses = set().union(set(), *rs[1:])
            if any(type(o).__name__.startswith('UReg') for o in ops) or uses & tainted:
                w = f'pc {pc}: {d.name} condition depends on a uniform or a skipped result (trip count unknown)'
                if w not in TRACE_WARNINGS: TRACE_WARNINGS.append(w)
        if d.name in ('jmp_exec_any', 'jmp_exec_none'):
            off = dict(d.decode_fields(n))['off']; off = off-(1 << 32) if off & (1 << 31) else off
            take = any(st.exec) if d.name == 'jmp_exec_any' else not any(st.exec)
            pc = pc+off if take else pc+size; continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                d.exec(n, st)
            dst = next((r for r in rs if r), set())
            tainted -= dst
            if set().union(set(), *rs[1:]) & tainted: tainted |= dst
        except Exception:
            skipped.add(d.name)
            dst = next((r for r in rs if r), set())
            tainted |= dst
        pc += size
    return pcs, skipped


def fit_branch(P):
    """Taken-branch penalty so that an empty counted-loop trip costs the measured 35.25 units."""
    import cost_hw as H
    P.branch = 0
    c = [simulate(decode(H.counted(b'', t), P), trace(H.counted(b'', t))[0], P, 1) for t in (200, 1000)]
    target = P.cal.get('measured', {}).get('loop_trip_cycles_empty', 35.25)*REAL_CYCLES_PER_UNIT
    P.branch = max(0.0, target-(c[1]-c[0])/800)
    return P.branch


TOKSCALE = 100             # unit/port/issue tokens per tick (fractional instruction costs stay exact)
TICKS = 10                 # simulation quantum: 0.1 real cycle (latencies such as 2.098 stay fractional)


def simulate(instrs, pcs, P, groups=1, detail=False):
    """Simulate `groups` identical SIMD groups on one core; returns real cycles (and per-pc stats).
    Time advances in ticks of 1/TICKS cycle; when nothing can issue, it jumps to the next tick at
    which a dependency, slot, group or resource can change."""
    if P.branch is None: fit_branch(P)
    T = TICKS
    tk = lambda c: int(round(c*T))
    by_pc = {r['pc']: r for r in instrs}
    cal = P.cal
    W = tk(cal.get('rf_writeback_real', RF_WRITEBACK)); W16 = tk(cal.get('rf_writeback16_real', RF_WRITEBACK16))
    NC = cal.get('result_cache_entries', RESULT_CACHE)
    BR = tk(P.branch); CL = tk(cal.get('control_latency', 2)); CI = tk(cal.get('control_issue_cycles', 1))
    nxt = [0]*groups; free = [0]*groups
    ready = [defaultdict(int) for _ in range(groups)]
    crossed = [dict() for _ in range(groups)]   # half -> ready tick after its one while-crossing bump
    slots = [defaultdict(list) for _ in range(groups)]
    rcache = [[] for _ in range(groups)]; wtime = [dict() for _ in range(groups)]; last_w = [dict() for _ in range(groups)]
    tok = {}; port = READ_PORTS*T; issue_tok = ISSUE_PER_CORE*T
    load_busy = [0]*groups                   # tick until which a load of this group is outstanding
    last_alu = [-10**9]*groups               # tick of the group's last ALU issue (quiet-time rule)
    alu_done = [-10**9]*groups               # latest completion tick of the group's ALU results (wait drain)
    last_wait = [-10**9]*groups              # issue tick of the group's previous wait
    wait_done = [-10**9]*groups              # completion tick of the group's previous wait
    last_idle_wait = [-10**9]*groups         # issue tick of the group's previous wait with nothing pending
    sfu_iss = [-10**9]*groups; sfu_wb = [-10**9]*groups; sfu_slow = [-10**9]*groups   # transcendental pipe state
    core_sfu = {'wb': -10**9, 'slow': -10**9, 'fast': -10**9, 'imad': -10**9}
    alu_since_mem = [False]*groups           # an ALU instruction issued since the group's last memory instruction
    last_space = [None]*groups
    pending = [[] for _ in range(groups)]       # [slot, completion tick, latency ticks, dst halves] of in-flight loads
    last_t = 0
    stats = defaultdict(lambda: {'n': 0, 'stall': 0, 'issue': 0, 'branch': 0})

    def src_ready(gi, h):
        r = ready[gi][h]; reg = h//2
        if reg in wtime[gi] and reg not in rcache[gi]:
            t0, narrow = wtime[gi][reg]; r = max(r, t0+(W16 if narrow else W))
        return r

    def bucket_of(u):
        """(bucket, tokens per instruction, bucket rate per cycle) for a benchmark row."""
        sh = P.shared_of.get(u)
        if sh in (PIPE_CLASSES if SFU_PIPE else ('sfu',)) and CORE_SFU: return 'sfu_core_free', 0, 1.0
        if sh: return sh, int(round(T*TOKSCALE/max(P.unit_cap[u], 1e-3))), 1.0
        return P.unit_bucket[u], T*TOKSCALE, P.unit_cap[u]

    def refill(dt):
        nonlocal port, issue_tok
        # buckets are kept in units of 1/T instructions (or reads); rates per tick
        port = min(port+READ_PORTS*dt, (READ_PORTS+4)*T)
        issue_tok = min(issue_tok+ISSUE_PER_CORE*dt, ISSUE_PER_CORE*T)
        for b_ in tok:
            if b_ in SHARED_UNITS: tok[b_] = min(tok[b_]+dt*TOKSCALE, 2*T*TOKSCALE)        # 1 unit-cycle per cycle
            else: c_ = P.unit_cap.get(b_, 1.0); tok[b_] = min(tok[b_]+c_*dt*TOKSCALE, (c_+1)*T*TOKSCALE)

    t = 0; done = 0; last_gi = 0; last_iss = [0]*groups
    while done < groups:
        issued = 0
        if ARBITRATION == 'hash':
            k0 = ((t//T)*2654435761 >> 7) % groups; order = list(range(groups)); order = order[k0:]+order[:k0]
        elif ARBITRATION == 'gto':
            # greedy-then-oldest: the group that issued last keeps priority, then the one idle longest
            order = sorted(range(groups), key=lambda g_: (g_ != last_gi, last_iss[g_]))
        else:
            order = list(range(groups)); k0 = (t//T) % groups; order = order[k0:]+order[:k0]
        wake = []
        for gi in order:
            if nxt[gi] >= len(pcs): continue
            if free[gi] > t: wake.append(free[gi]); continue
            r = by_pc[pcs[nxt[gi]]]
            sr = max((src_ready(gi, h) for h in r['src']), default=0)
            if sr > t: wake.append(sr); continue
            if r['kind'] == 'wait':
                pend = slots[gi][r['slot']]
                if pend and max(pend) > t: wake.append(max(pend)); continue
                if last_wait[gi]+tk(P.wait_idle) > t: wake.append(last_wait[gi]+tk(P.wait_idle)); continue   # wait issue spacing
            if issue_tok < T: wake.append(t+1); continue
            u = r['unit']
            if u:
                bk, cost, cap = bucket_of(u)
                if bk not in tok: tok[bk] = (cap+1)*T*TOKSCALE
                need_port = min(r['reads32'], READ_PORTS)*T
                if tok[bk] < cost or port < need_port: wake.append(t+1); continue
            if r['dst'] and len(r['dst']) in (1, 2):
                wid = len(r['dst'])
                blk = [wtime[gi][reg][0]+2*T for reg in {h//2 for h in r['dst']}
                       if reg in last_w[gi] and {last_w[gi][reg], wid} == {1, 2} and wtime[gi][reg][0]+2*T > t]
                if blk: wake.append(max(blk)); continue
            sfu_op = SFU_PIPE and u and P.shared_of.get(u) in PIPE_CLASSES
            if sfu_op:
                L_ = tk(P.rows[u]['lat'])
                gate = max(sfu_iss[gi]+tk(SFU_SPACING), sfu_wb[gi]+tk(SFU_WB_GAP)-L_,
                           sfu_slow[gi] if u in SFU_SLOW else -10**9)
                if CORE_SFU and groups > 1:
                    cs_ = CORE_SFU
                    gate = max(gate, core_sfu['wb']+tk(cs_['wb_gap'])-L_,
                               core_sfu['slow'] if u in cs_['slow'] else
                               core_sfu['imad'] if (P.shared_of.get(u) == 'imadbf' and CORE_IMAD == 'own') else core_sfu['fast'])
                if gate > t: wake.append(gate); continue
            # ---- issue at tick t
            issue_tok -= T
            if u:
                bk, cost, _ = bucket_of(u); tok[bk] -= cost; port -= min(r['reads32'], READ_PORTS)*T
            kind = r['kind']; row = P.rows.get(u) if u else None; branch = 0
            if kind == 'alu':
                lat, ii = tk(row['lat']), tk(row['ii'])
                if sfu_op:
                    # the pipe, not the SIMD group, carries the transcendental's issue interval
                    # (rcp chain + 3 iadd: 72.6 units, not 83.3; log2+rcp alternating = pure log2)
                    sfu_iss[gi] = t; sfu_wb[gi] = max(sfu_wb[gi], t+lat)
                    if u in SFU_SLOW: sfu_slow[gi] = t+tk(SFU_SLOW[u])
                    ii = tk(SFU_GROUP_BLOCK)
                    if CORE_SFU and groups > 1:
                        core_sfu['wb'] = max(core_sfu['wb'], t+lat)
                        if u in CORE_SFU['slow']: core_sfu['slow'] = t+tk(CORE_SFU['slow'][u])
                        elif P.shared_of.get(u) == 'imadbf':
                            core_sfu['imad' if CORE_IMAD == 'own' else 'fast'] = t+tk(1.0/P.unit_cap[u])
                        else: core_sfu['fast'] = t+tk(CORE_SFU['fast'])
                for h in r['dst']: ready[gi][h] = t+lat
                last_alu[gi] = t; alu_done[gi] = max(alu_done[gi], t+lat); alu_since_mem[gi] = True
                for p_ in pending[gi]:          # ALU issue defers in-flight loads
                    newdone = t+p_[2]
                    if newdone > p_[1]:
                        p_[1] = newdone
                        for h in p_[3]: ready[gi][h] = max(ready[gi][h], newdone)
                        slots[gi][p_[0]].append(newdone)     # wait drains to the latest completion
            elif kind == 'load':
                tgop = r['name'] == 'threadgroup_load'
                delay = max(0, last_alu[gi]+tk(P.quiet['tg' if tgop else 'load'])-t)     # quiet-time rule
                if tgop:
                    lat, ii = delay+tk(P.mem['threadgroup_load']['lat'][0]), tk(P.mem['threadgroup_load']['issue'][0])
                else:
                    # multiload.json: a device load issued after ALU work while another load is pending starts only
                    # when the pending loads have completed (back-to-back loads still overlap)
                    if alu_since_mem[gi] and pending[gi]:
                        delay = max(delay, max(p_[1] for p_ in pending[gi])-t)
                    lat = delay+tk(P.load_fast); ii = tk(P.load_ii)
                cls_ = 'tg' if tgop else 'device_load'
                if {cls_, last_space[gi]} == {'tg', 'device_store'}: ii += tk(P.space_switch)
                last_space[gi] = cls_
                load_busy[gi] = max(load_busy[gi], t+lat); alu_since_mem[gi] = False
                for h in r['dst']: ready[gi][h] = t+lat
                slots[gi][r['slot']].append(t+lat)
                dfr = tk(P.tg_defer) if tgop else tk(P.defer_fast)+(tk(P.defer_delayed_extra) if delay else 0)
                pending[gi].append([r['slot'], t+lat, delay+dfr, r['dst']])
            elif kind == 'store':
                m_ = P.mem.get(r['name'], P.mem['device_store'])
                q = P.quiet['tg' if r['name'].startswith('threadgroup') else 'device_store']
                delay = max(0, last_alu[gi]+tk(q)-t)
                if not r['name'].startswith('threadgroup'):
                    # storeseries.json: a device store starts >= 10.6 units after the previous idle wait issued
                    # (a wait that waited for memory does not delay it: spaces.json 'dev store, tg load, wait')
                    # (iadd, k idle waits, store: 4.9 units per extra wait from k = 4; 10.2-10.7 for k = 4..16)
                    delay = max(delay, last_idle_wait[gi]+tk(P.store_after_wait)-t)
                ii = delay+tk(m_['issue'][0]); slow = 0
                st_done = max(t+ii+tk(m_['done'][1]), t+tk(m_['issue'][0])+tk(m_['done'][0]))   # completion overlaps the quiet delay
                cls_ = 'tg' if r['name'].startswith('threadgroup') else 'device_store'
                if {cls_, last_space[gi]} == {'tg', 'device_store'}: ii += tk(P.space_switch)   # spaces.json
                last_space[gi] = cls_
                slots[gi][r['slot']].append(st_done); alu_since_mem[gi] = False
            elif kind == 'tg_atomic':
                ii = T
                for h in r['dst']: ready[gi][h] = t+tk(P.load_latency)
            elif kind == 'wait':
                # a wait cannot issue sooner than wait_idle after the previous wait issued, and completes no earlier
                # than wait_drain after the latest ALU result (waitdrain.json, waitdrain-slots.json); pending slot work is handled by the slot check above. The long-run
                # ~11-unit mode of many consecutive waits is not understood and not modelled.
                done_ = max(t+T, alu_done[gi]+tk(P.wait_drain))
                if not slots[gi][r['slot']]: last_idle_wait[gi] = t      # nothing was pending on the slot
                ii = done_-t; last_wait[gi] = t; wait_done[gi] = done_; slots[gi][r['slot']] = []
                pending[gi] = [p_ for p_ in pending[gi] if p_[0] != r['slot']]
            elif kind == 'control':
                ii = CI
                if CROSS_BUMP and r['src']-{0}:
                    # ctlspacer.json: a result produced before a register-compare control instruction (while_icmp)
                    # and read after it arrives one control latency later, once however many it crosses
                    # (while,iadd chain 3.8 not 2.0; while,iadd,iadd 5.7; while,while,iadd 4.1; push/pop: no effect)
                    for h, rr in list(ready[gi].items()):
                        if rr+CL > t and crossed[gi].get(h) != rr:
                            ready[gi][h] = rr+CL; crossed[gi][h] = rr+CL
                for h in r['dst']: ready[gi][h] = t+CL
                # ctlspacer.json: control instructions count as ALU work for the wait drain and the memory quiet time
                if CTL_AS_ALU:
                    last_alu[gi] = t; alu_done[gi] = max(alu_done[gi], t+CL)
                if r['name'] == 'jmp_exec_any' and nxt[gi]+1 < len(pcs) and pcs[nxt[gi]+1] < r['pc']:
                    branch = BR
            else:
                ii = T
            if kind in ('alu', 'load') and r['dst']:
                for reg in sorted({h//2 for h in r['dst']}):
                    if reg in rcache[gi]: rcache[gi].remove(reg)
                    rcache[gi].append(reg); wtime[gi][reg] = (t, len(r['dst']) == 1)
                    last_w[gi][reg] = len(r['dst']) if len(r['dst']) <= 2 else 4
                    if len(rcache[gi]) > NC: rcache[gi].pop(0)
            occ = max(T, ii)+branch            # "group" rule: the issue interval occupies the whole SIMD group
            st = stats[r['pc']]; st['n'] += 1
            st['stall'] += (t-free[gi]) if nxt[gi] else 0
            st['issue'] += occ; st['branch'] += branch
            free[gi] = t+occ; last_t = max(last_t, free[gi]); last_gi = gi; last_iss[gi] = t
            nxt[gi] += 1; issued += 1
            if nxt[gi] >= len(pcs): done += 1
        if done >= groups: break
        nt = t+1 if issued or not wake else max(t+1, min(wake))
        refill(nt-t); t = nt
        if t > 50_000_000*T: raise RuntimeError('simulation limit')
    total = last_t/T
    if detail:
        for st in stats.values():
            for k in ('stall', 'issue', 'branch'): st[k] /= T
        return total, stats
    return total


def to_ref(real_cycles):
    return real_cycles/REAL_CYCLES_PER_UNIT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('code', type=Path)
    ap.add_argument('--groups', type=int, default=1)
    ap.add_argument('--load-latency', type=int, help='device load-to-use latency in real cycles')
    ap.add_argument('--units', choices=['real', 'ref'], default='real')
    ap.add_argument('--allow-unknown', action='store_true')
    ap.add_argument('--annotate', action='store_true')
    a = ap.parse_args()
    P = Params(load_latency=a.load_latency); P.allow_unknown = a.allow_unknown
    code = a.code.read_bytes()
    ins = decode(code, P); pcs, skipped = trace(code)
    cyc, stats = simulate(ins, pcs, P, a.groups, detail=True)
    scale = 1.0 if a.units == 'real' else 1/REAL_CYCLES_PER_UNIT
    print(json.dumps({'instructions': len(ins), 'dynamic': len(pcs), 'groups': a.groups, 'units': a.units,
                      'cycles': round(cyc*scale, 1), 'emulator_skipped': sorted(skipped), 'warnings': TRACE_WARNINGS,
                      'unmeasured_classes': sorted({r['name'] for r in ins if r['unmeasured']})}))
    if a.annotate:
        print(f"{'pc':>6}  {'n':>7}  {'issue':>8}  {'stall':>8}  {'class':16s} instruction")
        for r in ins:
            s = stats.get(r['pc'], {'n': 0, 'stall': 0, 'issue': 0})
            flag = ' *unmeasured' if r['unmeasured'] else ''
            print(f"{r['pc']:6d}  {s['n']:7d}  {s['issue']*scale:8.0f}  {s['stall']*scale:8.0f}  {(r['unit'] or r['kind']):16s} {r['text']}{flag}")


if __name__ == '__main__':
    main()
