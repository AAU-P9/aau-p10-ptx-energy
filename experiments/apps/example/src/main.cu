#include "cupti_timing.h"

__global__ void ptx_kernel(int *out, int iterations)
{
    int tid = threadIdx.x;
    int tmp = tid;

    // Repeat the instruction in a C loop
    for(int i = 0; i < iterations; ++i)
    {
        asm volatile (
            "mov.u32 %0, %0;\n\t"  // move tmp to tmp (self-move)
            : "+r"(tmp)             // %0 is a register mapped to tmp
        );
    }
}

int main()
{
    int h[4] = {0}; 
    int *d;
    int iterations = 100000000; // 100 million iterations

    // Initialize CUPTI profiling
    METRICS_KERNEL_START

    cudaMalloc(&d, 4*sizeof(int));
    cudaMemcpy(d, h, 4*sizeof(int), cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel with %d iterations...\n", iterations);

    // Get CPU/GPU offsets

    // Run kernel
    ptx_kernel<<<1,4>>>(d, iterations);
    cudaDeviceSynchronize();

    // Possibly read back results (not necessary for timing, but included for completeness)
    // cudaMemcpy(h, d, 4*sizeof(int), cudaMemcpyDeviceToHost);

    // Flush all activity buffers
    METRICS_KERNEL_END

    // Clean up
    cudaFree(d);
    
    disableCUPTI();
}
