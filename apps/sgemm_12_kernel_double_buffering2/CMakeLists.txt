find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_12_kernel_double_buffering2 main.cu)

target_link_libraries(sgemm_12_kernel_double_buffering2 lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)