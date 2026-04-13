#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>

#define ITERATIONS 40000000

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
        asm volatile (
            "mov.u32 %0, %0;\n\t"  // move tmp to tmp (self-move)
            : "+r"(tmp)             // %0 is a register mapped to tmp
        );
    }
}
    
int main(int argc, char *argv[])
{
    // Read the grid and block dimensions from command line arguments
    if (argc != 3) {
        printf("Usage: %s <gridDim> <blockDim>\n", argv[0]);
        return -1;
    }

    int _gridDim = atoi(argv[1]);
    int _blockDim = atoi(argv[2]);

    int h[4] = {0}; 
    int *d;

    // Initialize CUPTI profiling
    METRICS_KERNEL_START

    cudaMalloc(&d, 4*sizeof(int));
    cudaMemcpy(d, h, 4*sizeof(int), cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel mov_mov_u32 with %d iterations...\n", ITERATIONS);

    JSON_START

    // Get CPU/GPU offsets

    // Run kernel
    printf("[LOG] Launching kernel with gridDim=%d and blockDim=%d...\n", _gridDim, _blockDim);
    
    ptx_kernel<<<_gridDim, _blockDim>>>();

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
    METRICS_KERNEL_END

    EXPORT(_blockDim);
    EXPORT_N("gridDim", _gridDim);
    EXPORT(ITERATIONS);
    JSON_END

    // Clean up
    cudaFree(d);
    
    disableCUPTI();
}
