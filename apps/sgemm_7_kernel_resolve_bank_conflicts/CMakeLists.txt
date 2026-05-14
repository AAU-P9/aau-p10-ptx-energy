find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_7_kernel_resolve_bank_conflicts main.cu)

target_link_libraries(sgemm_7_kernel_resolve_bank_conflicts lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)