; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@2 = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8

define void @parallel_vector_add(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14) {
  %structArg = alloca { ptr, ptr, ptr }, align 8
  %.reloaded = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %.reloaded9 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %.reloaded10 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %10, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %11, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %12, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %13, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %14, 4, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, ptr %6, 1
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %7, 2
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, ptr %1, 1
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %2, 2
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 %3, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, i64 %4, 4, 0
  br label %entry

entry:                                            ; preds = %15
  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, ptr %.reloaded, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, ptr %.reloaded9, align 8
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, ptr %.reloaded10, align 8
  br label %omp_parallel

omp_parallel:                                     ; preds = %entry
  %gep_.reloaded = getelementptr { ptr, ptr, ptr }, ptr %structArg, i32 0, i32 0
  store ptr %.reloaded, ptr %gep_.reloaded, align 8
  %gep_.reloaded9 = getelementptr { ptr, ptr, ptr }, ptr %structArg, i32 0, i32 1
  store ptr %.reloaded9, ptr %gep_.reloaded9, align 8
  %gep_.reloaded10 = getelementptr { ptr, ptr, ptr }, ptr %structArg, i32 0, i32 2
  store ptr %.reloaded10, ptr %gep_.reloaded10, align 8
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @parallel_vector_add..omp_par, ptr %structArg)
  br label %omp.par.exit

omp.par.exit:                                     ; preds = %omp_parallel
  ret void
}

; Function Attrs: nounwind
define internal void @parallel_vector_add..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
omp.par.entry:
  %gep_.reloaded = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 0
  %loadgep_.reloaded = load ptr, ptr %gep_.reloaded, align 8, !align !1
  %gep_.reloaded9 = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 1
  %loadgep_.reloaded9 = load ptr, ptr %gep_.reloaded9, align 8, !align !1
  %gep_.reloaded10 = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 2
  %loadgep_.reloaded10 = load ptr, ptr %gep_.reloaded10, align 8, !align !1
  %p.lastiter = alloca i32, align 4
  %p.lowerbound = alloca i64, align 8
  %p.upperbound = alloca i64, align 8
  %p.stride = alloca i64, align 8
  %tid.addr.local = alloca i32, align 4
  %1 = load i32, ptr %tid.addr, align 4
  store i32 %1, ptr %tid.addr.local, align 4
  %tid = load i32, ptr %tid.addr.local, align 4
  %2 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded, align 8
  %3 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded9, align 8
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %loadgep_.reloaded10, align 8
  br label %omp.region.after_alloca2

omp.region.after_alloca2:                         ; preds = %omp.par.entry
  br label %omp.region.after_alloca

omp.region.after_alloca:                          ; preds = %omp.region.after_alloca2
  br label %omp.par.region

omp.par.region:                                   ; preds = %omp.region.after_alloca
  br label %omp.par.region1

omp.par.region1:                                  ; preds = %omp.par.region
  br label %omp.wsloop.region

omp.wsloop.region:                                ; preds = %omp.par.region1
  br label %omp_loop.preheader

omp_loop.preheader:                               ; preds = %omp.wsloop.region
  store i64 0, ptr %p.lowerbound, align 4
  store i64 1023, ptr %p.upperbound, align 4
  store i64 1, ptr %p.stride, align 4
  %omp_global_thread_num7 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_for_static_init_8u(ptr @1, i32 %omp_global_thread_num7, i32 34, ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride, i64 1, i64 0)
  %5 = load i64, ptr %p.lowerbound, align 4
  %6 = load i64, ptr %p.upperbound, align 4
  %7 = sub i64 %6, %5
  %8 = add i64 %7, 1
  br label %omp_loop.header

omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
  %omp_loop.iv = phi i64 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
  br label %omp_loop.cond

omp_loop.cond:                                    ; preds = %omp_loop.header
  %omp_loop.cmp = icmp ult i64 %omp_loop.iv, %8
  br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit

omp_loop.exit:                                    ; preds = %omp_loop.cond
  call void @__kmpc_for_static_fini(ptr @1, i32 %omp_global_thread_num7)
  %omp_global_thread_num8 = call i32 @__kmpc_global_thread_num(ptr @1)
  call void @__kmpc_barrier(ptr @2, i32 %omp_global_thread_num8)
  br label %omp_loop.after

omp_loop.after:                                   ; preds = %omp_loop.exit
  br label %omp.region.cont3

omp.region.cont3:                                 ; preds = %omp_loop.after
  br label %omp.region.cont

omp.region.cont:                                  ; preds = %omp.region.cont3
  br label %omp.par.pre_finalize

omp.par.pre_finalize:                             ; preds = %omp.region.cont
  br label %.fini

.fini:                                            ; preds = %omp.par.pre_finalize
  br label %omp.par.exit.exitStub

omp_loop.body:                                    ; preds = %omp_loop.cond
  %9 = add i64 %omp_loop.iv, %5
  %10 = mul i64 %9, 1
  %11 = add i64 %10, 0
  br label %omp.loop_nest.region

omp.loop_nest.region:                             ; preds = %omp_loop.body
  %12 = call ptr @llvm.stacksave.p0()
  br label %omp.loop_nest.region5

omp.loop_nest.region5:                            ; preds = %omp.loop_nest.region
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, 1
  %14 = getelementptr inbounds nuw float, ptr %13, i64 %11
  %15 = load float, ptr %14, align 4
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 1
  %17 = getelementptr inbounds nuw float, ptr %16, i64 %11
  %18 = load float, ptr %17, align 4
  %19 = fadd float %15, %18
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %21 = getelementptr inbounds nuw float, ptr %20, i64 %11
  store float %19, ptr %21, align 4
  call void @llvm.stackrestore.p0(ptr %12)
  br label %omp.loop_nest.region6

omp.loop_nest.region6:                            ; preds = %omp.loop_nest.region5
  br label %omp.region.cont4

omp.region.cont4:                                 ; preds = %omp.loop_nest.region6
  br label %omp_loop.inc

omp_loop.inc:                                     ; preds = %omp.region.cont4
  %omp_loop.next = add nuw i64 %omp_loop.iv, 1
  br label %omp_loop.header

omp.par.exit.exitStub:                            ; preds = %.fini
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #1

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64) #0

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) #0

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) #2

; Function Attrs: nounwind
declare !callback !2 void @__kmpc_fork_call(ptr, i32, ptr, ...) #0

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i64 8}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
