#include "cupti_timing.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "ptx_meta.h"
#define ITERATIONS 1

__global__ void ptx_kernel(void* __restrict__ sink) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int a_0 = tid; long long d_0 = 0;
    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);

    #pragma unroll 1
    for (int i = 0; i < ITERATIONS; ++i) {
        asm volatile (
            "cvt.s64.s32 %0, %1;" : "=l"(d_0) : "r"(a_0)
        );
    }
    ((long long*)sink)[tid] = d_0;
}

int main() {
METRICS_KERNEL_START

    void* sink;
    cudaMalloc(&sink, sizeof(long long) * 1 * 1);
    
    ptx_kernel<<<1, 1>>>(sink);
    EXPORT_N("gridDim_x", 1);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", 1);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);
    cudaDeviceSynchronize();
    cudaFree(sink);
    
    
METRICS_KERNEL_END
return 0;
}
