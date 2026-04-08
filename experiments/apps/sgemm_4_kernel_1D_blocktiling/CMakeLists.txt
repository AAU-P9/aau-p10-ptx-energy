find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_4_kernel_1D_blocktiling main.cu)

target_link_libraries(sgemm_4_kernel_1D_blocktiling lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)