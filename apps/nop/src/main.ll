; ModuleID = 'main.cu'
source_filename = "main.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.timespec = type { i64, i64 }
%struct.CUpti_Activity = type { i32, [4 x i8] }
%struct.dim3 = type { i32, i32, i32 }

$_Z15initializeCUPTIv = comdat any

$_Z23collectTimestampOffsetsv = comdat any

$_ZN4dim3C2Ejjj = comdat any

$_Z17flushCUPTIBuffersv = comdat any

$_Z17printKernelTimingv = comdat any

$_Z12disableCUPTIv = comdat any

@_ZL10start_time = internal global i64 0, align 8
@_ZL8end_time = internal global i64 0, align 8
@_ZL15kernel_duration = internal global i64 0, align 8
@.str = private unnamed_addr constant [44 x i8] c"[LOG] Running kernel with %d iterations...\0A\00", align 1
@.str.1 = private unnamed_addr constant [77 x i8] c"[OFFSET] CUPTI Timestamp: %lu, CPU Timestamp #1: %lu, CPU Timestamp #2: %lu\0A\00", align 1
@__const._Z23collectTimestampOffsetsv.sleepTime = private unnamed_addr constant %struct.timespec { i64 0, i64 10000000 }, align 8
@.str.2 = private unnamed_addr constant [26 x i8] c"[KERNEL] Start Time: %lu\0A\00", align 1
@.str.3 = private unnamed_addr constant [24 x i8] c"[KERNEL] End Time: %lu\0A\00", align 1
@.str.4 = private unnamed_addr constant [24 x i8] c"[KERNEL] Duration: %lu\0A\00", align 1

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z15bufferRequestedPPhPmS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %8 = call noalias ptr @malloc(i64 noundef 32776) #9
  store ptr %8, ptr %7, align 8
  %9 = load ptr, ptr %7, align 8
  %10 = ptrtoint ptr %9 to i64
  %11 = add i64 %10, 8
  %12 = sub i64 %11, 1
  %13 = and i64 %12, -8
  %14 = inttoptr i64 %13 to ptr
  %15 = load ptr, ptr %4, align 8
  store ptr %14, ptr %15, align 8
  %16 = load ptr, ptr %5, align 8
  store i64 32768, ptr %16, align 8
  %17 = load ptr, ptr %6, align 8
  store i64 0, ptr %17, align 8
  ret void
}

; Function Attrs: nounwind allocsize(0)
declare noalias ptr @malloc(i64 noundef) #1

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z15bufferCompletedP8CUctx_stjPhmm(ptr noundef %0, i32 noundef %1, ptr noundef %2, i64 noundef %3, i64 noundef %4) #2 {
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca ptr, align 8
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8
  store i32 %1, ptr %7, align 4
  store ptr %2, ptr %8, align 8
  store i64 %3, ptr %9, align 8
  store i64 %4, ptr %10, align 8
  %14 = load i64, ptr %10, align 8
  %15 = icmp ugt i64 %14, 0
  br i1 %15, label %16, label %43

16:                                               ; preds = %5
  store ptr null, ptr %11, align 8
  br label %17

17:                                               ; preds = %41, %16
  %18 = load ptr, ptr %8, align 8
  %19 = load i64, ptr %10, align 8
  %20 = call i32 @cuptiActivityGetNextRecord(ptr noundef %18, i64 noundef %19, ptr noundef %11)
  %21 = icmp eq i32 0, %20
  br i1 %21, label %22, label %42

22:                                               ; preds = %17
  %23 = load ptr, ptr %11, align 8
  %24 = getelementptr inbounds %struct.CUpti_Activity, ptr %23, i32 0, i32 0
  %25 = load i32, ptr %24, align 8
  %26 = icmp eq i32 %25, 3
  br i1 %26, label %27, label %41

27:                                               ; preds = %22
  %28 = load ptr, ptr %11, align 8
  %29 = getelementptr inbounds i8, ptr %28, i64 16
  store ptr %29, ptr %12, align 8
  %30 = load ptr, ptr %11, align 8
  %31 = getelementptr inbounds i8, ptr %30, i64 24
  store ptr %31, ptr %13, align 8
  %32 = load ptr, ptr %12, align 8
  %33 = load i64, ptr %32, align 8
  store i64 %33, ptr @_ZL10start_time, align 8
  %34 = load ptr, ptr %13, align 8
  %35 = load i64, ptr %34, align 8
  store i64 %35, ptr @_ZL8end_time, align 8
  %36 = load ptr, ptr %13, align 8
  %37 = load i64, ptr %36, align 8
  %38 = load ptr, ptr %12, align 8
  %39 = load i64, ptr %38, align 8
  %40 = sub i64 %37, %39
  store i64 %40, ptr @_ZL15kernel_duration, align 8
  br label %41

41:                                               ; preds = %27, %22
  br label %17, !llvm.loop !7

42:                                               ; preds = %17
  br label %43

43:                                               ; preds = %42, %5
  %44 = load ptr, ptr %8, align 8
  call void @free(ptr noundef %44) #10
  ret void
}

declare i32 @cuptiActivityGetNextRecord(ptr noundef, i64 noundef, ptr noundef) #3

; Function Attrs: nounwind
declare void @free(ptr noundef) #4

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local void @_Z25__device_stub__ptx_kernelPi(ptr noundef %0) #5 {
  %2 = alloca ptr, align 8
  %3 = alloca %struct.dim3, align 8
  %4 = alloca %struct.dim3, align 8
  %5 = alloca i64, align 8
  %6 = alloca ptr, align 8
  %7 = alloca { i64, i32 }, align 8
  %8 = alloca { i64, i32 }, align 8
  store ptr %0, ptr %2, align 8
  %9 = alloca ptr, i64 1, align 16
  %10 = getelementptr ptr, ptr %9, i32 0
  store ptr %2, ptr %10, align 8
  %11 = call i32 @__cudaPopCallConfiguration(ptr %3, ptr %4, ptr %5, ptr %6)
  %12 = load i64, ptr %5, align 8
  %13 = load ptr, ptr %6, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %3, i64 12, i1 false)
  %14 = getelementptr inbounds { i64, i32 }, ptr %7, i32 0, i32 0
  %15 = load i64, ptr %14, align 8
  %16 = getelementptr inbounds { i64, i32 }, ptr %7, i32 0, i32 1
  %17 = load i32, ptr %16, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %8, ptr align 8 %4, i64 12, i1 false)
  %18 = getelementptr inbounds { i64, i32 }, ptr %8, i32 0, i32 0
  %19 = load i64, ptr %18, align 8
  %20 = getelementptr inbounds { i64, i32 }, ptr %8, i32 0, i32 1
  %21 = load i32, ptr %20, align 8
  %22 = call noundef i32 @cudaLaunchKernel(ptr noundef @_Z25__device_stub__ptx_kernelPi, i64 %15, i32 %17, i64 %19, i32 %21, ptr noundef %9, i64 noundef %12, ptr noundef %13)
  br label %23

23:                                               ; preds = %1
  ret void
}

declare i32 @__cudaPopCallConfiguration(ptr, ptr, ptr, ptr)

declare i32 @cudaLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #6

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #7 {
  %1 = alloca i32, align 4
  %2 = alloca [4 x i32], align 16
  %3 = alloca ptr, align 8
  %4 = alloca %struct.dim3, align 4
  %5 = alloca %struct.dim3, align 4
  %6 = alloca { i64, i32 }, align 4
  %7 = alloca { i64, i32 }, align 4
  store i32 0, ptr %1, align 4
  call void @llvm.memset.p0.i64(ptr align 16 %2, i8 0, i64 16, i1 false)
  call void @_Z15initializeCUPTIv()
  %8 = call noundef i32 @_ZL10cudaMallocIiE9cudaErrorPPT_m(ptr noundef %3, i64 noundef 16)
  %9 = load ptr, ptr %3, align 8
  %10 = getelementptr inbounds [4 x i32], ptr %2, i64 0, i64 0
  %11 = call i32 @cudaMemcpy(ptr noundef %9, ptr noundef %10, i64 noundef 16, i32 noundef 1)
  %12 = call i32 (ptr, ...) @printf(ptr noundef @.str, i32 noundef 40000000)
  call void @_Z23collectTimestampOffsetsv()
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %4, i32 noundef 54938, i32 noundef 1, i32 noundef 1)
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %5, i32 noundef 4, i32 noundef 1, i32 noundef 1)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %6, ptr align 4 %4, i64 12, i1 false)
  %13 = getelementptr inbounds { i64, i32 }, ptr %6, i32 0, i32 0
  %14 = load i64, ptr %13, align 4
  %15 = getelementptr inbounds { i64, i32 }, ptr %6, i32 0, i32 1
  %16 = load i32, ptr %15, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %7, ptr align 4 %5, i64 12, i1 false)
  %17 = getelementptr inbounds { i64, i32 }, ptr %7, i32 0, i32 0
  %18 = load i64, ptr %17, align 4
  %19 = getelementptr inbounds { i64, i32 }, ptr %7, i32 0, i32 1
  %20 = load i32, ptr %19, align 4
  %21 = call i32 @__cudaPushCallConfiguration(i64 %14, i32 %16, i64 %18, i32 %20, i64 noundef 0, ptr noundef null)
  %22 = icmp ne i32 %21, 0
  br i1 %22, label %25, label %23

23:                                               ; preds = %0
  %24 = load ptr, ptr %3, align 8
  call void @_Z25__device_stub__ptx_kernelPi(ptr noundef %24) #11
  br label %25

25:                                               ; preds = %23, %0
  %26 = call i32 @cudaDeviceSynchronize()
  call void @_Z17flushCUPTIBuffersv()
  call void @_Z17printKernelTimingv()
  %27 = load ptr, ptr %3, align 8
  %28 = call i32 @cudaFree(ptr noundef %27)
  call void @_Z12disableCUPTIv()
  %29 = load i32, ptr %1, align 4
  ret i32 %29
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #8

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_Z15initializeCUPTIv() #2 comdat {
  %1 = call i32 @cuptiActivityEnable(i32 noundef 3)
  %2 = call i32 @cuptiActivityRegisterCallbacks(ptr noundef @_Z15bufferRequestedPPhPmS1_, ptr noundef @_Z15bufferCompletedP8CUctx_stjPhmm)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define internal noundef i32 @_ZL10cudaMallocIiE9cudaErrorPPT_m(ptr noundef %0, i64 noundef %1) #2 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load i64, ptr %4, align 8
  %7 = call i32 @cudaMalloc(ptr noundef %5, i64 noundef %6)
  ret i32 %7
}

declare i32 @cudaMemcpy(ptr noundef, ptr noundef, i64 noundef, i32 noundef) #3

declare i32 @printf(ptr noundef, ...) #3

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_Z23collectTimestampOffsetsv() #2 comdat {
  %1 = alloca i32, align 4
  %2 = alloca i64, align 8
  %3 = alloca %struct.timespec, align 8
  %4 = alloca %struct.timespec, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca %struct.timespec, align 8
  store i32 0, ptr %1, align 4
  br label %8

8:                                                ; preds = %32, %0
  %9 = load i32, ptr %1, align 4
  %10 = icmp slt i32 %9, 100
  br i1 %10, label %11, label %35

11:                                               ; preds = %8
  store i64 0, ptr %2, align 8
  %12 = call i32 @clock_gettime(i32 noundef 1, ptr noundef %3) #10
  %13 = call i32 @cuptiGetTimestamp(ptr noundef %2)
  %14 = call i32 @clock_gettime(i32 noundef 1, ptr noundef %4) #10
  %15 = getelementptr inbounds %struct.timespec, ptr %3, i32 0, i32 0
  %16 = load i64, ptr %15, align 8
  %17 = mul i64 %16, 1000000000
  %18 = getelementptr inbounds %struct.timespec, ptr %3, i32 0, i32 1
  %19 = load i64, ptr %18, align 8
  %20 = add i64 %17, %19
  store i64 %20, ptr %5, align 8
  %21 = getelementptr inbounds %struct.timespec, ptr %4, i32 0, i32 0
  %22 = load i64, ptr %21, align 8
  %23 = mul i64 %22, 1000000000
  %24 = getelementptr inbounds %struct.timespec, ptr %4, i32 0, i32 1
  %25 = load i64, ptr %24, align 8
  %26 = add i64 %23, %25
  store i64 %26, ptr %6, align 8
  %27 = load i64, ptr %2, align 8
  %28 = load i64, ptr %5, align 8
  %29 = load i64, ptr %6, align 8
  %30 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i64 noundef %27, i64 noundef %28, i64 noundef %29)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 @__const._Z23collectTimestampOffsetsv.sleepTime, i64 16, i1 false)
  %31 = call i32 @nanosleep(ptr noundef %7, ptr noundef null)
  br label %32

32:                                               ; preds = %11
  %33 = load i32, ptr %1, align 4
  %34 = add nsw i32 %33, 1
  store i32 %34, ptr %1, align 4
  br label %8, !llvm.loop !9

35:                                               ; preds = %8
  ret void
}

declare i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64 noundef, ptr noundef) #3

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) unnamed_addr #0 comdat align 2 {
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store i32 %1, ptr %6, align 4
  store i32 %2, ptr %7, align 4
  store i32 %3, ptr %8, align 4
  %9 = load ptr, ptr %5, align 8
  %10 = getelementptr inbounds %struct.dim3, ptr %9, i32 0, i32 0
  %11 = load i32, ptr %6, align 4
  store i32 %11, ptr %10, align 4
  %12 = getelementptr inbounds %struct.dim3, ptr %9, i32 0, i32 1
  %13 = load i32, ptr %7, align 4
  store i32 %13, ptr %12, align 4
  %14 = getelementptr inbounds %struct.dim3, ptr %9, i32 0, i32 2
  %15 = load i32, ptr %8, align 4
  store i32 %15, ptr %14, align 4
  ret void
}

declare i32 @cudaDeviceSynchronize() #3

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_Z17flushCUPTIBuffersv() #2 comdat {
  %1 = call i32 @cuptiActivityFlushAll(i32 noundef 0)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_Z17printKernelTimingv() #2 comdat {
  %1 = load i64, ptr @_ZL15kernel_duration, align 8
  %2 = icmp ugt i64 %1, 0
  br i1 %2, label %3, label %10

3:                                                ; preds = %0
  %4 = load i64, ptr @_ZL10start_time, align 8
  %5 = call i32 (ptr, ...) @printf(ptr noundef @.str.2, i64 noundef %4)
  %6 = load i64, ptr @_ZL8end_time, align 8
  %7 = call i32 (ptr, ...) @printf(ptr noundef @.str.3, i64 noundef %6)
  %8 = load i64, ptr @_ZL15kernel_duration, align 8
  %9 = call i32 (ptr, ...) @printf(ptr noundef @.str.4, i64 noundef %8)
  br label %10

10:                                               ; preds = %3, %0
  ret void
}

declare i32 @cudaFree(ptr noundef) #3

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_Z12disableCUPTIv() #2 comdat {
  %1 = call i32 @cuptiActivityDisable(i32 noundef 3)
  ret void
}

declare i32 @cuptiActivityEnable(i32 noundef) #3

declare i32 @cuptiActivityRegisterCallbacks(ptr noundef, ptr noundef) #3

; Function Attrs: nounwind
declare i32 @clock_gettime(i32 noundef, ptr noundef) #4

declare i32 @cuptiGetTimestamp(ptr noundef) #3

declare i32 @nanosleep(ptr noundef, ptr noundef) #3

declare i32 @cuptiActivityFlushAll(i32 noundef) #3

declare i32 @cuptiActivityDisable(i32 noundef) #3

declare i32 @cudaMalloc(ptr noundef, i64 noundef) #3

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="true" }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #9 = { nounwind allocsize(0) }
attributes #10 = { nounwind }
attributes #11 = { "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 3]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = distinct !{!9, !8}
