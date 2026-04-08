#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>

#define ITERATIONS 40000000

__global__ void ptx_kernel(unsigned long long *out)
{
    int tid = threadIdx.x;
    unsigned long long tmp = (unsigned long long)tid + 1ULL;

    // Repeat the instruction in a C loop
    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for(int i = 0; i < ITERATIONS; ++i)
    {
        asm volatile (
            "shl.b64 %0, %0, 1;\n\t"
            : "+l"(tmp)
        );
    }

    out[tid] = tmp;
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
    printf("[LOG] Launching kernel with gridDim=%d and blockDim=%d...\n", _gridDim, _blockDim);

    ptx_kernel<<<_gridDim,_blockDim>>>(d);

    cudaDeviceSynchronize();

    // Flush all activity buffers
    flushCUPTIBuffers();

    printKernelTiming();

    // Clean up
    cudaFree(d);

    disableCUPTI();
}
