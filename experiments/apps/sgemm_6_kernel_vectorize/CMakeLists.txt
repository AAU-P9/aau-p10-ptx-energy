find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_6_kernel_vectorize main.cu)

target_link_libraries(sgemm_6_kernel_vectorize lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)