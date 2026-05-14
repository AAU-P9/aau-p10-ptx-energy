find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_10_kernel_warptiling main.cu)

target_link_libraries(sgemm_10_kernel_warptiling lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)