#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>

// Dependent chain: cvt.s64.s32 -> shl.b64 -> add.s64
// Exact pattern seen in vector_add address calculation:
//   thread index -> widen to 64-bit -> scale by element size -> add base pointer.
// All three instructions are RAW-dependent on their predecessor.

#ifndef _GRID_DIM
#define _GRID_DIM 256
#endif

#ifndef _BLOCK_DIM
#define _BLOCK_DIM 256
#endif

#ifndef ITERATIONS
#define ITERATIONS 40000000
#endif

__global__ void ptx_kernel(long long *out)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long addr = 0;
    long long base = (long long)tid * 16LL;

    for (int i = 0; i < ITERATIONS; ++i) {
        asm volatile(
            "cvt.s64.s32 %0, %2;\n\t"   // addr = (s64) tid
            "shl.b64     %0, %0, 2;\n\t"  // addr <<= 2  (RAW dep on cvt)
            "add.s64     %1, %0, %1;\n\t" // base += addr (RAW dep on shl)
            : "+l"(addr), "+l"(base)
            : "r"(tid)
        );
    }

    out[tid % 512] = base;
}

int main()
{
    METRICS_KERNEL_START

    long long *d_out;
    cudaMalloc(&d_out, 512 * sizeof(long long));

    printf("[LOG] pair_cvt_shl_add_dep: %d iterations, dep chain cvt->shl->add\n", ITERATIONS);
    printf("[LOG] Launching with gridDim=%d blockDim=%d\n", _GRID_DIM, _BLOCK_DIM);

    ptx_kernel<<<_GRID_DIM, _BLOCK_DIM>>>(d_out);
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
