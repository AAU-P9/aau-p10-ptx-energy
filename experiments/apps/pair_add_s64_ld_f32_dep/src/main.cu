#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Dependent pair: add.s64 -> ld.global.f32
// 64-bit address arithmetic feeds a global load (RAW dep: load address = add output).
// This is the dominant pattern in any indexed or pointer-based kernel — address
// computation and the subsequent memory access are inherently dependent.

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
    // Base pointer as a 64-bit integer for explicit address arithmetic.
    unsigned long long base = (unsigned long long)data;

    for (int i = 0; i < ITERATIONS; ++i) {
        unsigned long long addr;
        float val;
        // add.s64: compute element address from base + offset
        // ld.global.f32: load from the computed address (RAW dep on add.s64)
        asm volatile(
            "add.s64  %0, %2, %3;\n\t"
            "ld.global.f32 %1, [%0];\n\t"
            : "=l"(addr), "=f"(val)
            : "l"(base), "l"((unsigned long long)(((tid + i * stride) % len) * sizeof(float)))
        );
        acc += val;
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

    printf("[LOG] pair_add_s64_ld_f32_dep: %d iterations, dep pair add.s64->ld.global\n", ITERATIONS);
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
