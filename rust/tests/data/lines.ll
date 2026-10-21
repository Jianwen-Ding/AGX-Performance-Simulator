define agx_kernel void @k(ptr addrspace(1) %p, i32 %v) !dbg !5 {
  %a = add i32 %v, 3, !dbg !8
  store i32 %a, ptr addrspace(1) %p, !dbg !9
  ret void, !dbg !10
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "t", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "k.cu", directory: "/src")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "k", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 2, column: 5, scope: !5)
!9 = !DILocation(line: 3, column: 7, scope: !5)
!10 = !DILocation(line: 4, column: 1, scope: !5)
