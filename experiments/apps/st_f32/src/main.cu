#include "cupti_timing.h"

#define ITERATIONS 40000000

__global__ void ptx_kernel(float *out)
{
    int tid = threadIdx.x;
    float tmp = (float)tid;
    float *ptr = out + tid;

    // Repeat the instruction in a C loop
    for(int i = 0; i < ITERATIONS; ++i)
    {
        asm volatile (
            "st.global.f32 [%0], %1;\n\t"
            :
            : "l"(ptr), "f"(tmp)
        );

        tmp += 1.0f;
    }
}

int main()
{
    float h_out[4] = {0.0f};
    float *d_out;

    // Initialize CUPTI profiling
    initializeCUPTI();

    cudaMalloc(&d_out, 4*sizeof(float));
    cudaMemcpy(d_out, h_out, 4*sizeof(float), cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

    // Get CPU/GPU offsets
    collectTimestampOffsets();

    // Run kernel
    printf("[LOG] Launching kernel with gridDim=%d and blockDim=%d...\n", _gridDim, _blockDim);

    ptx_kernel<<<_gridDim,_blockDim>>>(d_out);

    cudaDeviceSynchronize();

    // Flush all activity buffers
    flushCUPTIBuffers();

    printKernelTiming();

    // Clean up
    cudaFree(d_out);

    disableCUPTI();
}
