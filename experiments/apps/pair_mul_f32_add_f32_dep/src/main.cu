#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>

// Dependent pair: mul.f32 -> add.f32
// The FP32 multiply result feeds directly into the FP32 add (RAW dependency).
// Isolates the cost of a chained FP pipeline vs independent FP ops.

#ifndef _GRID_DIM
#define _GRID_DIM 256
#endif

#ifndef _BLOCK_DIM
#define _BLOCK_DIM 256
#endif

#ifndef ITERATIONS
#define ITERATIONS 40000000
#endif

__global__ void ptx_kernel(float *out)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 1.0001f + (float)tid * 0.00001f;
    float b = 0.9999f;
    float acc = 0.0f;

    for (int i = 0; i < ITERATIONS; ++i) {
        asm volatile(
            "mul.f32 %0, %0, %2;\n\t"   // a = a * b
            "add.f32 %1, %0, %1;\n\t"   // acc = a + acc  (RAW dep: reads a from mul)
            : "+f"(a), "+f"(acc)
            : "f"(b)
        );
    }

    out[tid % 1024] = acc;
}

int main()
{
    METRICS_KERNEL_START

    float *d_out;
    cudaMalloc(&d_out, 1024 * sizeof(float));

    printf("[LOG] pair_mul_f32_add_f32_dep: %d iterations, dep pair\n", ITERATIONS);
    printf("[LOG] Launching with gridDim=%d blockDim=%d\n", _GRID_DIM, _BLOCK_DIM);

    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>(d_out);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", _GRID_DIM);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", _BLOCK_DIM);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(d_out);
    disableCUPTI();
}
