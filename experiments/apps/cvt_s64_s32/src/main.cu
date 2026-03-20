#include "cupti_timing.h"

#define ITERATIONS 40000000

__global__ void ptx_kernel(long long *out)
{
    int tid = threadIdx.x;
    int tmp_s32 = tid;
    long long tmp_s64 = 0;

    // Repeat the instruction in a C loop
    for(int i = 0; i < ITERATIONS; ++i)
    {
        asm volatile (
            "cvt.s64.s32 %0, %1;\n\t"
            : "=l"(tmp_s64)
            : "r"(tmp_s32)
        );

        ++tmp_s32;
    }

    out[tid] = tmp_s64;
}

int main()
{
    long long h[4] = {0};
    long long *d;

    // Initialize CUPTI profiling
    initializeCUPTI();

    cudaMalloc(&d, 4*sizeof(long long));
    cudaMemcpy(d, h, 4*sizeof(long long), cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

    // Get CPU/GPU offsets
    collectTimestampOffsets();

    // Run kernel
    ptx_kernel<<<1,4>>>(d);
    cudaDeviceSynchronize();

    // Possibly read back results (not necessary for timing, but included for completeness)
    // cudaMemcpy(h, d, 4*sizeof(int), cudaMemcpyDeviceToHost);

    // Flush all activity buffers
    flushCUPTIBuffers();

    printKernelTiming();

    // Clean up
    cudaFree(d);
    
    disableCUPTI();
}
