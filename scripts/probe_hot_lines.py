#!/usr/bin/env python3
"""probe_hot_lines.py: do loads of the same lines by every workgroup on every core (a broadcast operand, llama.cpp's
k_bin_bcast bias row) cost more than loads of distinct lines?

Each thread (workgroups of 128) runs ITER trips of: load a[gid + trip * N*128] and b[...], wait, add, store
out[gid + trip * N*128] (k_bin_bcast's shape). BCAST=1: b's index is the lane's word in one of the 8 blocks of 128 words picked by workgroup id mod 8 (every workgroup reads one of the same 1024 words); 0: b is
indexed like a.

usage: probe_hot_lines.py [hw|sim|both]; env N="40 512 2048" BCAST="0 1" ITER=2 SERIAL=0 GRID2D=0 STRIDE=N*128 REGSTORE=0 NOSTORE=0 ROWW=0 (with GRID2D) OUTMB=0 (OUT's buffer object size) PRE=0 POST=0 IFWRAP=0 B2B=5000
"""
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tg_alu as P

A, B, OUT = 0x1100000000, 0x1110000000, 0x1120000000
FREE = (1, 2, 3, 4, 5, 8, 9, 14)  # registers the probe does not use (PRE / POST)


def build(iters, bcast):
    code = P.e('iadd r0l, 0, 0')
    code += P.e('get_sr r10, sr80', 'get_sr r10, sr80 (thread_position_in_grid.x)')
    code += P.e('get_sr r11, sr48', 'get_sr r11, sr48 (thread_position_in_threadgroup.x)')
    code += P.e('get_sr r15, sr0', 'get_sr r15, sr0 (threadgroup_position_in_grid.x)')
    if os.environ.get('ROWW'):
        # ROWW=w (with GRID2D=x): row-major rows of w words, one row per grid row (k_bin_bcast: x = 512 threads in
        # 4 workgroups writing 512 (ne0) of each 896-word row): index = y * w + x
        code += P.e('get_sr r17, sr1', 'get_sr r17, sr1 (threadgroup_position_in_grid.y)')
        code += P.e('imadd r10, r17, u15, r10')
    # broadcast index: (wg mod 7) * 128 + lane, i.e. one of the ROW-word row's 7 blocks (ROW = 896)
    if bcast:
        code += P.e('and r16, r15, 7')  # 8 blocks: 1024 words, close to 896
        code += P.e('imadd r21, r16, 128, r11')
    else:
        code += P.e('iadd r21, r10, 0')
    # PRE=n / POST=n: n independent iadds (8 registers round robin) before the first load / after the last store
    # (k_bin_bcast runs ~80 ALU instructions of index arithmetic before its loads)
    for i in range(int(os.environ.get('PRE', '0'))):
        code += P.e(f'iadd r{FREE[i % 8]}, r{FREE[i % 8]}, 1')
    wrap = os.environ.get('IFWRAP') == '1'  # each trip inside an always-true if (k_bin_bcast's loop is)
    for i in range(iters):
        if wrap:
            code += P.e('if_fcmp r0l, eq, 0.0, 0.0, 1', 'push_exec r0l, 1')
        if i:
            code += P.e('iadd r10, r10, u14') + (b'' if bcast else P.e('iadd r21, r21, u14'))
        code += P.e('device_load 0, i32, x, r12, u8_u9, r10, unsigned')
        if os.environ.get('SERIAL') == '1':  # wait for each load before the next (k_bin_bcast's compiled order)
            code += P.e('wait 0')
        code += P.e('device_load 0, i32, x, r13, u6_u7, r21, unsigned')
        code += P.e('wait 0')
        code += P.e('fadd32 r12, r12, r13')
        if os.environ.get('NOSTORE') == '1':  # no store (its cost = the difference)
            pass
        elif os.environ.get('REGSTORE') == '1':
            # the store's address in a 64-bit register pair per lane (k_bin_bcast's compiled form)
            code += P.e('iadd r6, u4, 0') + P.e('iadd r7, u5, 0') + P.e('iadd r30, r10, 0') + P.mov(31, 0)
            code += P.e('iadd r24_r25, r6_r7, r30_r31, lsl 2')
            code += P.e('device_store 0, i32, x, r12, r24_r25, 0, signed, lsl 0, 0', 'device_store 0, i32, x, r12, r24_r25, 0, signed, 0')
        else:
            code += P.e('device_store 0, i32, x, r12, u4_u5, r10, unsigned, lsl 0, 0', 'device_store 0, i32, x, r12, u4_u5, r10, unsigned, 0')
        if wrap:
            code += P.e('pop_exec r0l, 1')
    for i in range(int(os.environ.get('POST', '0'))):
        code += P.e(f'iadd r{FREE[i % 8]}, r{FREE[i % 8]}, 1')
    return code + P.e('stop') + bytes(64)


def args(f, n, iters):
    tot = n * 128
    sz = tot * iters * 4 + 65536
    if os.environ.get('ROWW'):
        sz = max(sz, (tot // int(os.environ['GRID2D'])) * int(os.environ['ROWW']) * 4 + 65536)
    # GRID2D=X: the same threads as a 2-D grid X wide (k_bin_bcast: 512 x 512); the indices then repeat over y
    x = int(os.environ.get('GRID2D', '0'))
    grid = f'{x},{tot // x},1' if x else f'{tot},1,1'
    return [f, '--grid', grid, '--wg', '128,1,1', '--regs', os.environ.get('REGS', '64'),
            '--uniform', f'8={A & 0xffffffff}', '--uniform', f'9={A >> 32}', '--uniform', f'6={B & 0xffffffff}',
            '--uniform', f'7={B >> 32}', '--uniform', f'4={OUT & 0xffffffff}', '--uniform', f'5={OUT >> 32}',
            # STRIDE=k: a thread's next trip is k words on (default the whole grid: another page)
            '--uniform', f'14={os.environ.get("STRIDE", tot)}', '--uniform', f'15={os.environ.get("ROWW", 0)}', '--alloc', f'{A:#x}={sz}', '--alloc', f'{B:#x}={sz}', '--alloc', f'{OUT:#x}={max(sz, int(os.environ.get("OUTMB", "0")) << 20)}'] + (
            # EXTRA_MB=n: also map n MiB that the kernel never touches (captures map ~800 MB)
            ['--alloc', f'0x1200000000={int(os.environ["EXTRA_MB"]) << 20}'] if os.environ.get('EXTRA_MB') else [])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sim'
    iters = int(os.environ.get('ITER', '2'))
    if mode in ('hw', 'both'):
        import gpu_safety; gpu_safety.require_budget(1)
    with tempfile.TemporaryDirectory() as d:
        for bc in [int(x) for x in os.environ.get('BCAST', '0 1').split()]:
            f = os.path.join(d, f'k{bc}.bin'); open(f, 'wb').write(build(iters, bc))
            for n in [int(x) for x in os.environ.get('N', '40 512 2048').split()]:
                a = args(f, n, iters)
                chk = subprocess.run([P.RUN] + a, capture_output=True, text=True)
                if '"instructions"' not in chk.stdout:
                    raise RuntimeError('emulator gate: ' + chk.stdout[-300:] + chk.stderr[-300:])
                out = [f'BCAST {bc} N {n} ITER {iters}']
                if mode in ('hw', 'both'):
                    h = subprocess.run([P.HW] + a + ['--back-to-back', os.environ.get('B2B', '5000'), '--timeout-ms', '10000'],
                                       capture_output=True, text=True)
                    m = re.search(r'"b2b_median_fw_ns":(\d+)', h.stdout)
                    out.append(f'hw {int(m.group(1)) / 1000:9.1f}' if m else 'hw FAILED ' + (h.stdout + h.stderr)[-200:])
                if mode in ('sim', 'both'):
                    r = subprocess.run([P.DISP] + a + ['--launch-profile', 'runtime'], capture_output=True, text=True)
                    out.append(f"sim {json.loads(r.stdout.strip().splitlines()[-1])['time_us']:9.1f}")
                print('  '.join(out), flush=True)


if __name__ == '__main__':
    main()
