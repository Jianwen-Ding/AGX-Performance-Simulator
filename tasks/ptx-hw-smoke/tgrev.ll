; Reverses each 64-thread workgroup's slice through threadgroup memory, then atomically counts the threads.
@tg = internal addrspace(3) global [64 x i32] undef
define agx_kernel void @tgrev(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %count) {
  %g = call i32 @llvm.agx.thread.position.in.grid.x()
  %t = call i32 @llvm.agx.thread.index.in.threadgroup()
  %pi = getelementptr i32, ptr addrspace(1) %in, i32 %g
  %v = load i32, ptr addrspace(1) %pi
  %ps = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  store i32 %v, ptr addrspace(3) %ps
  call void @llvm.agx.threadgroup.barrier()
  %r = sub i32 63, %t
  %pr = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %r
  %w = load i32, ptr addrspace(3) %pr
  %po = getelementptr i32, ptr addrspace(1) %out, i32 %g
  store i32 %w, ptr addrspace(1) %po
  %old = atomicrmw add ptr addrspace(1) %count, i32 1 monotonic
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.index.in.threadgroup()
declare void @llvm.agx.threadgroup.barrier()
