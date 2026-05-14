find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_3_kernel_shared_mem_blocking main.cu)

target_link_libraries(sgemm_3_kernel_shared_mem_blocking lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)