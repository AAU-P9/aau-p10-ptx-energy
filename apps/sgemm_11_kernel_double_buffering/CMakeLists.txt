find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_11_kernel_double_buffering main.cu)

target_link_libraries(sgemm_11_kernel_double_buffering lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)