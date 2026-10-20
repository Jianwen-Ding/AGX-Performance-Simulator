	.file	"vadd.ll"
	.text
	.globl	vadd                            // -- Begin function vadd
	.p2align	1
	.type	vadd,@function
vadd:                                   // @vadd
	// uniform registers: 18 x 16-bit (threadgroups_per_grid at u6)
	// general purpose registers: 4 x 32-bit
	// threadgroup memory: 0 bytes
	// stack size: 0 bytes
// %bb.0:
	get_sr r1, sr80 (thread_position_in_grid.x)
	device_load 0, i32, x, r2, u0_u1, r1, signed
	device_load 0, i32, x, r3, u2_u3, r1, signed
	wait 0
	fadd32 r2, r2, r3
	device_store 0, i32, x, r2, u4_u5, r1, signed
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
	.size	vadd, .Lfunc_end0-vadd
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
