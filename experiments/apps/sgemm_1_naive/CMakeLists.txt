find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_1_naive main.cu)

target_link_libraries(sgemm_1_naive lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)