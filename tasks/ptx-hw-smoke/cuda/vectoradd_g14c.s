	.file	"llvm-link"
	.text
	.globl	_Z9vectorAddPKfS0_Pfi           // -- Begin function _Z9vectorAddPKfS0_Pfi
	.p2align	1
	.type	_Z9vectorAddPKfS0_Pfi,@function
_Z9vectorAddPKfS0_Pfi:                  // @_Z9vectorAddPKfS0_Pfi
	// uniform registers: 20 x 16-bit (threadgroups_per_grid at u7)
	// general purpose registers: 4 x 32-bit
	// threadgroup memory: 0 bytes
	// stack size: 0 bytes
// %bb.0:                               // %"428"
	mov_imm r0l, 0
	get_sr r1, sr4 (threads_per_threadgroup.x)
	get_sr r2, sr0 (threadgroup_position_in_grid.x)
	get_sr r3, sr48 (thread_position_in_threadgroup.x)
	imadd r1, r1, r2, r3
	icmpsel slt, r2, r1, u6, 1, 0
	if_icmp r0l, nueq, r2, 0, 1
	jmp_exec_none .LBB0_2
// %bb.1:                               // %"200"
	device_load 0, i32, x, r2, u0_u1, r1, signed
	device_load 0, i32, x, r3, u2_u3, r1, signed
	wait 0
	fadd32 r2, r2, r3
	device_store 0, i32, x, r2, u4_u5, r1, signed
.LBB0_2:                                // %"202"
	pop_exec r0l, 1
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
	.size	_Z9vectorAddPKfS0_Pfi, .Lfunc_end0-_Z9vectorAddPKfS0_Pfi
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
	.section	.agx.kernels,"",@progbits
	.p2align	2, 0x0
	.long	112
	.long	1
	.long	21
	.ascii	"_Z9vectorAddPKfS0_Pfi"
	.zero	3
	.long	20
	.long	14
	.long	4294967295
	.long	4
	.long	0
	.long	0
	.long	4
	.long	0
	.long	8
	.long	0
	.long	0
	.long	8
	.long	4
	.long	0
	.long	8
	.long	8
	.long	0
	.long	4
	.long	12
