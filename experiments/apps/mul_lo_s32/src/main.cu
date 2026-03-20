#include "cupti_timing.h"

#define ITERATIONS 40000000

__global__ void ptx_kernel(int *out)
{
    int tid = threadIdx.x;
    int tmp = tid;

    // Repeat the instruction in a C loop
    for(int i = 0; i < ITERATIONS; ++i)
    {
        asm volatile (
            "mul.lo.s32 %0, %0, 3;\n\t"
            : "+r"(tmp)               // %0 is a register mapped to tmp
        );
    }

    out[tid] = tmp;
}

int main()
{
    int h[4] = {0}; 
    int *d;

    // Initialize CUPTI profiling
    initializeCUPTI();

    cudaMalloc(&d, 4*sizeof(int));
    cudaMemcpy(d, h, 4*sizeof(int), cudaMemcpyHostToDevice);

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
