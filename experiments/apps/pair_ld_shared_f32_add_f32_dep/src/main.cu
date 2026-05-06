#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Dependent pair: ld.shared.f32 -> add.f32
// Shared memory load feeding an FP add (RAW dep).
// Shared memory latency (~20 cycles) is much lower than global (~200 cycles),
// so this pair should show a different energy profile than ld.global.f32->add.f32.

#ifndef _GRID_DIM
#define _GRID_DIM 256
#endif

#ifndef _BLOCK_DIM
#define _BLOCK_DIM 256
#endif

#ifndef ITERATIONS
#define ITERATIONS 40000000
#endif

__global__ void ptx_kernel(const float * __restrict__ global_data, float *out)
{
    __shared__ float smem[256];

    int tid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    // Load block's slice of data into shared memory once.
    smem[tid] = global_data[gid];
    __syncthreads();

    float acc = 0.0f;
    // Hot loop: ld.shared.f32 -> add.f32, both in shared memory.
    for (int i = 0; i < ITERATIONS; ++i) {
        float val = smem[tid % blockDim.x];
        asm volatile("add.f32 %0, %0, %1;" : "+f"(acc) : "f"(val));
    }

    out[gid] = acc;
}

int main()
{
    METRICS_KERNEL_START

    int total = _GRID_DIM * _BLOCK_DIM;
    float *h_data = (float *)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++)
        h_data[i] = 1.0f + (float)(i % 100) * 0.0001f;

    float *d_data, *d_out;
    cudaMalloc(&d_data, total * sizeof(float));
    cudaMalloc(&d_out,  total * sizeof(float));
    cudaMemcpy(d_data, h_data, total * sizeof(float), cudaMemcpyHostToDevice);

    printf("[LOG] pair_ld_shared_f32_add_f32_dep: %d iterations, dep pair ld.shared->add\n", ITERATIONS);
    printf("[LOG] Launching with gridDim=%d blockDim=%d\n", _GRID_DIM, _BLOCK_DIM);

    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>(d_data, d_out);
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
