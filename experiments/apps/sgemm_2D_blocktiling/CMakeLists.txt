find_package(CUDAToolkit REQUIRED)

add_executable(kernel_2d_blocktiling main.cu)

target_link_libraries(kernel_2d_blocktiling lib cuda Tracy::TracyClient CUDA::cupti CUDA::nvml)
