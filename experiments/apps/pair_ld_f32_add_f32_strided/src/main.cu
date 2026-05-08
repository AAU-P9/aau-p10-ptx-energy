#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Dependent pair: ld.global.f32 -> add.f32  (strided / non-coalesced access)
// Thread N reads element N*32, so consecutive threads are 128 bytes apart.
// Each thread in a warp needs its own cache line — 32 transactions vs 1 for coalesced.
// Same instruction pair as pair_ld_f32_add_f32_coalesced; only the access pattern differs.

#ifndef _GRID_DIM
#define _GRID_DIM 256
#endif

#ifndef _BLOCK_DIM
#define _BLOCK_DIM 256
#endif

#ifndef ITERATIONS
#define ITERATIONS 1000000
#endif

// Must be large enough to hold strided accesses: threads * 32 * ITERATIONS
#define DATA_LEN (1024 * 1024 * 16)

// Stride between consecutive threads (in elements). 32 = one full cache line per thread.
#define THREAD_STRIDE 32

__global__ void ptx_kernel(const float * __restrict__ data, float *out, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float acc = 0.0f;

    // Non-coalesced: thread N reads element N*THREAD_STRIDE.
    // Adjacent threads are THREAD_STRIDE*4 = 128 bytes apart separate cache lines.
    for (int i = 0; i < ITERATIONS; ++i) {
        int addr = (int)(((long long)tid * THREAD_STRIDE + (long long)i * stride * THREAD_STRIDE) % len);
        float val = data[addr];
        asm volatile("add.f32 %0, %0, %1;" : "+f"(acc) : "f"(val));
    }

    out[tid % (len / THREAD_STRIDE)] = acc;
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
    cudaMalloc(&d_out,  (len / THREAD_STRIDE) * sizeof(float));
    cudaMemcpy(d_data, h_data, len * sizeof(float), cudaMemcpyHostToDevice);

    printf("[LOG] pair_ld_f32_add_f32_strided: %d iterations, strided (non-coalesced) ld->add\n", ITERATIONS);
    printf("[LOG] thread_stride=%d, Launching with gridDim=%d blockDim=%d\n", THREAD_STRIDE, _GRID_DIM, _BLOCK_DIM);

    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>(d_data, d_out, len);
    cudaDeviceSynchronize();
    printf("[LOG]here" );

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
    //disableCUPTI();
}
