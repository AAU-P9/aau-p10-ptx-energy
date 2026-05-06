#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Independent pair: ld.global.f32 -> ld.global.f32
// Two back-to-back loads from independent addresses — no RAW dependency.
// Measures whether consecutive loads can overlap in the memory pipeline.

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
        // Two loads from separate strided addresses — independent of each other.
        float val0 = data[(tid + i * stride) % len];
        float val1 = data[(tid + i * stride + stride / 2) % len];
        acc += val0 + val1;
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

    printf("[LOG] pair_ld_f32_ld_f32_indep: %d iterations, indep pair ld->ld\n", ITERATIONS);
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
