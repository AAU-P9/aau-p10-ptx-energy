#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Dependent pair: ld.global.f32 -> add.f32
// The loaded value feeds directly into the FP add (RAW dependency through memory).
// The loop address depends on i to prevent the compiler from hoisting the load.
// Compiles to ld.global.f32 followed immediately by add.f32 in the hot loop.

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

__global__ void ptx_kernel(const float * __restrict__ data, float *out, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float acc = 0.0f;

    for (int i = 0; i < ITERATIONS; ++i) {
        // Address depends on i: forces a real ld.global each iteration.
        float val = data[(tid + i * stride) % len];
        asm volatile("add.f32 %0, %0, %1;" : "+f"(acc) : "f"(val));
    }

    out[tid % len] = acc;
}

int main()
{
    METRICS_KERNEL_START

    int len = DATA_LEN;
    float *h_data = (float *)malloc(len * sizeof(float));
    for (int i = 0; i < len; i++)
        h_data[i] = 1.0f + (float)(i % 100) * 0.0001f;

    float *d_data, *d_out;
    cudaMalloc(&d_data, len * sizeof(float));
    cudaMalloc(&d_out,  len * sizeof(float));
    cudaMemcpy(d_data, h_data, len * sizeof(float), cudaMemcpyHostToDevice);

    printf("[LOG] pair_ld_add_f32_dep: %d iterations, dep pair ld->add\n", ITERATIONS);
    printf("[LOG] Launching with gridDim=%d blockDim=%d\n", _GRID_DIM, _BLOCK_DIM);

    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>(d_data, d_out, len);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", _GRID_DIM);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", _BLOCK_DIM);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(d_data);
    cudaFree(d_out);
    free(h_data);
    disableCUPTI();
}
