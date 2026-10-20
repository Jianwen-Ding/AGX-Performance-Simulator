; c[i] = a[i] + b[i], one thread per element
define agx_kernel void @vadd(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %i
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %i
  %pc = getelementptr float, ptr addrspace(1) %c, i32 %i
  %x = load float, ptr addrspace(1) %pa
  %y = load float, ptr addrspace(1) %pb
  %s = fadd float %x, %y
  store float %s, ptr addrspace(1) %pc
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
