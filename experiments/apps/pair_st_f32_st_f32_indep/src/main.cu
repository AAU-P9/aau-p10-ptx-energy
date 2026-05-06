#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Independent pair: st.global.f32 -> st.global.f32
// Two consecutive stores to independent addresses — no RAW dependency.
// Measures store throughput and whether back-to-back stores can be pipelined.

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

__global__ void ptx_kernel(float *dst0, float *dst1, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float val = (float)tid * 0.001f;

    for (int i = 0; i < ITERATIONS; ++i) {
        int addr = (tid + i * stride) % len;
        dst0[addr] = val;
        dst1[addr] = val + 1.0f;
    }
}

int main()
{
    METRICS_KERNEL_START

    int len = DATA_LEN;
    float *d_dst0, *d_dst1;
    cudaMalloc(&d_dst0, len * sizeof(float));
    cudaMalloc(&d_dst1, len * sizeof(float));

    printf("[LOG] pair_st_f32_st_f32_indep: %d iterations, indep pair st->st\n", ITERATIONS);
    printf("[LOG] Launching with gridDim=%d blockDim=%d\n", _GRID_DIM, _BLOCK_DIM);

    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>(d_dst0, d_dst1, len);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x", _GRID_DIM);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", _BLOCK_DIM);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(d_dst0);
    cudaFree(d_dst1);
    disableCUPTI();
}
