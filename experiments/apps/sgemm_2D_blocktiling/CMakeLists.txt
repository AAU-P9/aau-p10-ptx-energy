find_package(CUDAToolkit REQUIRED)

add_executable(kernel_2d_blocktiling main.cu)

target_include_directories(kernel_2d_blocktiling PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(kernel_2d_blocktiling lib cuda Tracy::TracyClient CUDA::cupti CUDA::nvml)
