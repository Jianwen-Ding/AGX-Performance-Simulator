import subprocess,sys
A='/save/workspace/AGX-Performance-Simulator/rust/target/release/agx_asm'
def e(t):
    o=subprocess.run([A],input=t+'\n',capture_output=True,text=True).stdout.split()[0]
    assert o!='ERROR',t; return bytes.fromhex(o)
def prog(cond):
    c=[e('iadd r0l, 0, 0'),e('get_sr r1, sr80 (thread_position_in_grid.x)'),e('iadd r4, r1, 64'),e('and r5, r1, u3')]
    c+= [e(cond)]
    c+= [e('get_sr r2, sr58 (active_thread_index_in_simdgroup)'),e('get_sr r3, sr59'),
         e('device_store 0, i32, x, r2, u0_u1, r1, unsigned, lsl 2, 0'),e('device_store 0, i32, x, r3, u0_u1, r4, unsigned, lsl 2, 0'),
         e('pop_exec r0l, 1'),e('wait 0'),e('stop')]
    return b''.join(c)+bytes(64)
open('lt.bin','wb').write(prog('if_icmp r0l, ult, r1, u2, 1'))     # tid < u2
open('msk.bin','wb').write(prog('if_icmp r0l, seq, r5, u2, 1'))    # (tid & u3) == u2
