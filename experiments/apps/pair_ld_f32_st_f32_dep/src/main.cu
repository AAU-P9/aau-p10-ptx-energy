#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Dependent pair: ld.global.f32 -> st.global.f32
// Loaded value is immediately stored to a different address (RAW dep: store value = loaded value).
// Classic memory copy pattern — measures the cost of a load feeding a store.

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

__global__ void ptx_kernel(const float * __restrict__ src, float *dst, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = 0; i < ITERATIONS; ++i) {
        int addr = (tid + i * stride) % len;
        float val = src[addr];
        dst[addr] = val;
    }
}

int main()
{
    METRICS_KERNEL_START

    int len = DATA_LEN;
    float *h_src = (float *)malloc(len * sizeof(float));
    for (int i = 0; i < len; i++)
        h_src[i] = (float)i * 0.001f;

    float *d_src, *d_dst;
    cudaMalloc(&d_src, len * sizeof(float));
    cudaMalloc(&d_dst, len * sizeof(float));
    cudaMemcpy(d_src, h_src, len * sizeof(float), cudaMemcpyHostToDevice);

    printf("[LOG] pair_ld_f32_st_f32_dep: %d iterations, dep pair ld->st\n", ITERATIONS);
    printf("[LOG] Launching with gridDim=%d blockDim=%d\n", _GRID_DIM, _BLOCK_DIM);

    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>(d_src, d_dst, len);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", _GRID_DIM);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", _BLOCK_DIM);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_src);
    disableCUPTI();
}
