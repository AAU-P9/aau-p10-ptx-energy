find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_8_kernel_bank_extra_col main.cu)

target_link_libraries(sgemm_8_kernel_bank_extra_col lib cuda CUDA::cublas Tracy::TracyClient CUDA::cupti CUDA::nvml)