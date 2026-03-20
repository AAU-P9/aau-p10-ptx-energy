#include "cupti_timing.h"

#define ITERATIONS 40000000

__global__ void ptx_kernel(unsigned long long *out)
{
    int tid = threadIdx.x;
    unsigned long long tmp = (unsigned long long)tid + 1ULL;

    // Repeat the instruction in a C loop
    for(int i = 0; i < ITERATIONS; ++i)
    {
        asm volatile (
            "shl.b64 %0, %0, 1;\n\t"
            : "+l"(tmp)
        );
    }

    out[tid] = tmp;
}

int main()
{
    unsigned long long h[4] = {0};
    unsigned long long *d;

    // Initialize CUPTI profiling
    initializeCUPTI();

    cudaMalloc(&d, 4*sizeof(unsigned long long));
    cudaMemcpy(d, h, 4*sizeof(unsigned long long), cudaMemcpyHostToDevice);

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
