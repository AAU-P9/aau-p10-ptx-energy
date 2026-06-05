#include "cupti_timing.h"

#define ITERATIONS 20000000

__global__ void ptx_kernel(int *out)
{
    int tid = threadIdx.x;
    int tmp = tid;

    // Repeat the instruction in a C loop
    for(int i = 0; i < ITERATIONS; ++i)
    {
        
    }
}

int main()
{
    int h[4] = {0}; 
    int *d;

    // Initialize CUPTI profiling
    METRICS_KERNEL_START

    cudaMalloc(&d, 4*sizeof(int));
    cudaMemcpy(d, h, 4*sizeof(int), cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

    // Get CPU/GPU offsets

    // Run kernel
    ptx_kernel<<<64,1024>>>(d);
    cudaDeviceSynchronize();

    // Possibly read back results (not necessary for timing, but included for completeness)
    // cudaMemcpy(h, d, 4*sizeof(int), cudaMemcpyDeviceToHost);

    // Flush all activity buffers
    METRICS_KERNEL_END

    // Clean up
    cudaFree(d);
    
    disableCUPTI();
}
