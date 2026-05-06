#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Dependent pair: ld.global.f64 -> add.f64
// Double-precision global load feeding a double-precision add (RAW dep).
// Contrasts with f32 variant — f64 uses wider memory transactions and different FP units.

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

__global__ void ptx_kernel(const double * __restrict__ data, double *out, int len)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    double acc = 0.0;

    for (int i = 0; i < ITERATIONS; ++i) {
        double val = data[(tid + i * stride) % len];
        asm volatile("add.f64 %0, %0, %1;" : "+d"(acc) : "d"(val));
    }

    out[tid % len] = acc;
}

int main()
{
    METRICS_KERNEL_START

    int len = DATA_LEN;
    double *h_data = (double *)malloc(len * sizeof(double));
    for (int i = 0; i < len; i++)
        h_data[i] = 1.0 + (double)(i % 100) * 0.0001;

    double *d_data, *d_out;
    cudaMalloc(&d_data, len * sizeof(double));
    cudaMalloc(&d_out,  len * sizeof(double));
    cudaMemcpy(d_data, h_data, len * sizeof(double), cudaMemcpyHostToDevice);

    printf("[LOG] pair_ld_f64_add_f64_dep: %d iterations, dep pair ld.f64->add.f64\n", ITERATIONS);
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
