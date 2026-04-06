; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @matmul(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %14, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %15, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %16, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %17, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %19, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %18, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %20, 4, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %8, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %9, 2
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %10, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %12, 4, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %11, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %13, 4, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %1, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %2, 2
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %3, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %5, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %4, 3, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %6, 4, 1
  br label %43

43:                                               ; preds = %79, %21
  %44 = phi i64 [ %80, %79 ], [ 0, %21 ]
  %45 = icmp slt i64 %44, 4
  br i1 %45, label %46, label %81

46:                                               ; preds = %43
  br label %47

47:                                               ; preds = %77, %46
  %48 = phi i64 [ %78, %77 ], [ 0, %46 ]
  %49 = icmp slt i64 %48, 4
  br i1 %49, label %50, label %79

50:                                               ; preds = %47
  br label %51

51:                                               ; preds = %54, %50
  %52 = phi i64 [ %76, %54 ], [ 0, %50 ]
  %53 = icmp slt i64 %52, 4
  br i1 %53, label %54, label %77

54:                                               ; preds = %51
  %55 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %56 = mul nuw nsw i64 %44, 4
  %57 = add nuw nsw i64 %56, %52
  %58 = getelementptr inbounds nuw float, ptr %55, i64 %57
  %59 = load float, ptr %58, align 4
  %60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %61 = mul nuw nsw i64 %52, 4
  %62 = add nuw nsw i64 %61, %48
  %63 = getelementptr inbounds nuw float, ptr %60, i64 %62
  %64 = load float, ptr %63, align 4
  %65 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %66 = mul nuw nsw i64 %44, 4
  %67 = add nuw nsw i64 %66, %48
  %68 = getelementptr inbounds nuw float, ptr %65, i64 %67
  %69 = load float, ptr %68, align 4
  %70 = fmul float %59, %64
  %71 = fadd float %69, %70
  %72 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %73 = mul nuw nsw i64 %44, 4
  %74 = add nuw nsw i64 %73, %48
  %75 = getelementptr inbounds nuw float, ptr %72, i64 %74
  store float %71, ptr %75, align 4
  %76 = add i64 %52, 1
  br label %51

77:                                               ; preds = %51
  %78 = add i64 %48, 1
  br label %47

79:                                               ; preds = %47
  %80 = add i64 %44, 1
  br label %43

81:                                               ; preds = %43
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
