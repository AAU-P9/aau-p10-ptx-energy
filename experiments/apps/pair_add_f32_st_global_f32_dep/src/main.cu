#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Dependent pair: add.f32 -> st.global.f32
// The FP add result is immediately written to global memory (RAW dep: store value = add output).
// Compute-then-write pattern common in reduction outputs and elementwise kernels.

#ifndef _GRID_DIM
#define _GRID_DIM 256
#endif

#ifndef _BLOCK_DIM
#define _BLOCK_DIM 256
#endif

#ifndef ITERATIONS
#define ITERATIONS 1000000
#endif

#define DATA_LEN (1024 * 1024)

__global__ void ptx_kernel(float *out, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float val = (float)tid * 0.001f;

    for (int i = 0; i < ITERATIONS; ++i) {
        float result;
        asm volatile("add.f32 %0, %1, %2;" : "=f"(result) : "f"(val), "f"((float)i));
        out[(tid + i * stride) % len] = result;
    }
}

int main()
{
    METRICS_KERNEL_START

    int len = DATA_LEN;
    float *d_out;
    cudaMalloc(&d_out, len * sizeof(float));

    printf("[LOG] pair_add_f32_st_global_f32_dep: %d iterations, dep pair add->st.global\n", ITERATIONS);
    printf("[LOG] Launching with gridDim=%d blockDim=%d\n", _GRID_DIM, _BLOCK_DIM);

    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>(d_out, len);
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
