find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_9_kernel_autotuned main.cu)

target_link_libraries(sgemm_9_kernel_autotuned lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)