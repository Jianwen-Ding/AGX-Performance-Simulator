; out[y * width + x] = (tg.x << 24) | (tg.y << 16) | (tid.y << 8) | tid.x on a 2D grid, with a branch
define agx_kernel void @ids2d(ptr addrspace(1) %out, i32 %width) {
  %gx = call i32 @llvm.agx.thread.position.in.grid.x()
  %gy = call i32 @llvm.agx.thread.position.in.grid.y()
  %lx = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %ly = call i32 @llvm.agx.thread.position.in.threadgroup.y()
  %tx = call i32 @llvm.agx.threadgroup.position.in.grid.x()
  %ty = call i32 @llvm.agx.threadgroup.position.in.grid.y()
  %a = shl i32 %tx, 24
  %b = shl i32 %ty, 16
  %c = shl i32 %ly, 8
  %v0 = or i32 %a, %b
  %v1 = or i32 %v0, %c
  %v2 = or i32 %v1, %lx
  %odd = and i32 %gx, 1
  %isodd = icmp ne i32 %odd, 0
  br i1 %isodd, label %flip, label %done
flip:
  %vf = xor i32 %v2, -2147483648
  br label %done
done:
  %v = phi i32 [ %v2, %0 ], [ %vf, %flip ]
  %row = mul i32 %gy, %width
  %idx = add i32 %row, %gx
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %idx
  store i32 %v, ptr addrspace(1) %p
  ret void
}
declare i32 @llvm.agx.thread.position.in.grid.x()
declare i32 @llvm.agx.thread.position.in.grid.y()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()
declare i32 @llvm.agx.thread.position.in.threadgroup.y()
declare i32 @llvm.agx.threadgroup.position.in.grid.x()
declare i32 @llvm.agx.threadgroup.position.in.grid.y()
