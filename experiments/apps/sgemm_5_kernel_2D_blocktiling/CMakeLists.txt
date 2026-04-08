find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_5_kernel_2D_blocktiling main.cu)

target_link_libraries(sgemm_5_kernel_2D_blocktiling lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)