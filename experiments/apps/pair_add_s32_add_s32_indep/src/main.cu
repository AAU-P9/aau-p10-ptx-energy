#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>

// Independent pair: add.s32 -> add.s32
// Both adds operate on separate registers — no RAW dependency between them.
// Baseline comparison against the dependent variant.

#ifndef _GRID_DIM
#define _GRID_DIM 256
#endif

#ifndef _BLOCK_DIM
#define _BLOCK_DIM 256
#endif

#ifndef ITERATIONS
#define ITERATIONS 40000000
#endif

__global__ void ptx_kernel(int *out)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int a = tid;
    int b = tid + 1;

    for (int i = 0; i < ITERATIONS; ++i) {
        asm volatile(
            "add.s32 %0, %0, 1;\n\t"    // a = a + 1
            "add.s32 %1, %1, 1;\n\t"    // b = b + 1  (no dep on a)
            : "+r"(a), "+r"(b)
        );
    }

    out[tid % 1024] = a + b;
}

int main()
{
    METRICS_KERNEL_START

    int *d_out;
    cudaMalloc(&d_out, 1024 * sizeof(int));

    printf("[LOG] pair_add_s32_add_s32_indep: %d iterations, indep pair\n", ITERATIONS);
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
