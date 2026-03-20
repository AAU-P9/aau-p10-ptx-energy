#include "cupti_timing.h"

#define ITERATIONS 40000000

__global__ void ptx_kernel(long long *out)
{
    int tid = threadIdx.x;
    long long tmp = tid;

    // Repeat the instruction in a C loop
    for(int i = 0; i < ITERATIONS; ++i)
    {
        asm volatile (
            "add.s64 %0, %0, 1;\n\t"
            : "+l"(tmp)
        );
    }

    out[tid] = tmp;
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

    // Flush all activity buffers
    flushCUPTIBuffers();

    printKernelTiming();

    // Clean up
    cudaFree(d);

    disableCUPTI();
}
