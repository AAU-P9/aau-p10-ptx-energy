#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>

#ifndef _GRID_DIM
#define _GRID_DIM 1
#endif

#ifndef _BLOCK_DIM
#define _BLOCK_DIM 256
#endif

#ifndef ITERATIONS
#define ITERATIONS 40000000
#endif

__global__ void ptx_kernel()
{
    int tid = threadIdx.x;
    int tmp = tid;

    // Repeat the instruction in a C loop
    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for(int i = 0; i < ITERATIONS; ++i)
    {
        asm volatile (     
            "mov.u32 %0, %0;\n\t"  // move tmp to tmp (self-move)
            : "+r"(tmp)             // %0 is a register mapped to tmp
        );
    }
}
    
int main(int argc, char *argv[])
{
    int h[4] = {0}; 
    int *d;

    // Initialize CUPTI profiling
    initializeCUPTI();

    cudaMalloc(&d, 4*sizeof(int));
    cudaMemcpy(d, h, 4*sizeof(int), cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel mov_32u with %d iterations...\n", ITERATIONS);

    // Get CPU/GPU offsets
    collectTimestampOffsets();

    // Run kernel
    printf("[LOG] Launching kernel with gridDim=%d and blockDim=%d...\n", _GRID_DIM, _BLOCK_DIM);
    
    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>();

    printf("[LOG] Kernel launched. Waiting for completion...\n");

    // Verify that the kernel didn't have any errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to launch vector_mul kernel (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }

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
