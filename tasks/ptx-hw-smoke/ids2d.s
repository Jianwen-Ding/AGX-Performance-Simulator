	.file	"ids2d.ll"
	.text
	.globl	ids2d                           // -- Begin function ids2d
	.p2align	1
	.type	ids2d,@function
ids2d:                                  // @ids2d
	// uniform registers: 12 x 16-bit (threadgroups_per_grid at u3)
	// general purpose registers: 6 x 32-bit
	// threadgroup memory: 0 bytes
	// stack size: 0 bytes
// %bb.0:
	mov_imm r0l, 0
	get_sr r1, sr80 (thread_position_in_grid.x)
	get_sr r2, sr0 (threadgroup_position_in_grid.x)
	get_sr r3, sr1 (threadgroup_position_in_grid.y)
	get_sr r4, sr49 (thread_position_in_threadgroup.y)
	bfi r5, 0, r2, 24
	bfi r3, 0, r3, 16
	get_sr r2, sr81 (thread_position_in_grid.y)
	bfi r4, 0, r4, 8
	or	r3, r5, r3
	get_sr r5, sr48 (thread_position_in_threadgroup.x)
	or	r3, r3, r4
	and	r4, r1, 1
	or	r3, r3, r5
	if_icmp r0l, nueq, r4, 0, 1
	jmp_exec_none .LBB0_2
// %bb.1:                               // %flip
	mov_imm r4, -2147483648
	xor	r3, r3, r4
.LBB0_2:                                // %done
	pop_exec r0l, 1
	imadd r1, r2, u2, r1
	device_store 0, i32, x, r3, u0_u1, r1, signed
	stop
	trap
	trap
	trap
	trap
	trap
	trap
	trap
	trap
.Lfunc_end0:
	.size	ids2d, .Lfunc_end0-ids2d
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
