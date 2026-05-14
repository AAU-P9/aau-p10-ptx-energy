find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_2_kernel_global_mem_coalesce main.cu)

target_link_libraries(sgemm_2_kernel_global_mem_coalesce lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)