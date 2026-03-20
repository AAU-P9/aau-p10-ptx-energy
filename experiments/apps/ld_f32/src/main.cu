#include "cupti_timing.h"

#define ITERATIONS 40000000

__global__ void ptx_kernel(const float *in, float *out)
{
    int tid = threadIdx.x;
    float tmp = 0.0f;
    const float *ptr = in + tid;

    // Repeat the instruction in a C loop
    for(int i = 0; i < ITERATIONS; ++i)
    {
        asm volatile (
            "ld.global.f32 %0, [%1];\n\t"
            : "=f"(tmp)
            : "l"(ptr)
        );
    }

    out[tid] = tmp;
}

int main()
{
    float h_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_out[4] = {0.0f};
    float *d_in;
    float *d_out;

    // Initialize CUPTI profiling
    initializeCUPTI();

    cudaMalloc(&d_in, 4*sizeof(float));
    cudaMalloc(&d_out, 4*sizeof(float));
    cudaMemcpy(d_in, h_in, 4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, 4*sizeof(float), cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

    // Get CPU/GPU offsets
    collectTimestampOffsets();

    // Run kernel
    ptx_kernel<<<1,4>>>(d_in, d_out);
    cudaDeviceSynchronize();

    // Flush all activity buffers
    flushCUPTIBuffers();

    printKernelTiming();

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);

    disableCUPTI();
}
