#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Dependent pair: ld.global.s32 -> add.s32
// Integer global load feeding an integer add (RAW dep).
// Integer counterpart to ld.global.f32->add.f32 — compares int vs FP load latency.

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

__global__ void ptx_kernel(const int * __restrict__ data, int *out, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int acc = 0;

    for (int i = 0; i < ITERATIONS; ++i) {
        int val = data[(tid + i * stride) % len];
        asm volatile("add.s32 %0, %0, %1;" : "+r"(acc) : "r"(val));
    }

    out[tid % len] = acc;
}

int main()
{
    METRICS_KERNEL_START

    int len = DATA_LEN;
    int *h_data = (int *)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++)
        h_data[i] = i % 1000;

    int *d_data, *d_out;
    cudaMalloc(&d_data, len * sizeof(int));
    cudaMalloc(&d_out,  len * sizeof(int));
    cudaMemcpy(d_data, h_data, len * sizeof(int), cudaMemcpyHostToDevice);

    printf("[LOG] pair_ld_s32_add_s32_dep: %d iterations, dep pair ld.s32->add.s32\n", ITERATIONS);
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
