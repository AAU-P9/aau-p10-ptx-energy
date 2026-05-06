#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>

// Independent pair: mul.f32 -> add.f32
// Multiply and add each operate on separate, unrelated registers.
// Baseline for the dependent variant to isolate RAW stall cost in FP pipeline.

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
    float c = 1.0002f;
    float d = 0.5f;

    for (int i = 0; i < ITERATIONS; ++i) {
        asm volatile(
            "mul.f32 %0, %0, %1;\n\t"   // a = a * b
            "add.f32 %2, %2, %3;\n\t"   // c = c + d  (no dep on a or b)
            : "+f"(a), "+f"(b), "+f"(c), "+f"(d)
        );
    }

    out[tid % 1024] = a + c;
}

int main()
{
    METRICS_KERNEL_START

    float *d_out;
    cudaMalloc(&d_out, 1024 * sizeof(float));

    printf("[LOG] pair_mul_f32_add_f32_indep: %d iterations, indep pair\n", ITERATIONS);
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
