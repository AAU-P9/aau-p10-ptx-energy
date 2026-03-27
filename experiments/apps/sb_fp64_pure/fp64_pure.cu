#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 10000000

// Pure FP64 arithmetic: DMAD/DADD/DMUL chain.
// On AD103, FP64 is 1:64 of FP32 throughput — expect very different
// power profile due to different execution units and scheduling.
__global__ void KERNEL_LAUNCH_BOUNDS(256, 4)
bench_fp64_pure(double *out, int N) {
    KERNEL_META(bench_fp64_pure,
        _META_FRAG_PARAM_PTR(0, out, f64, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(1, N, i32, dim_n, RANGE(1, 1024) STRIDE(1))
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(N+255)/256 1 1")
        _META_FRAG_CUSTOM("category", "compute_fp64")
        _META_FRAG_CUSTOM("arithmetic_intensity", "inf")
        _META_FRAG_CUSTOM("memory_pattern", "none")
    );

    DEVICE_ASSUME(N > 0);
    DEVICE_ASSUME(N <= 1024);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a = 1.0001 + (double)idx * 0.00001;
    double b = 0.9999;
    double c = 1.0;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; ++i) {
        a = a * b + c;   // DMAD
        b = b + a * c;   // DMAD
        c = c * a + b;   // DMAD
        a = a + b;       // DADD
        b = b * c;       // DMUL
        c = c + a;       // DADD
        a = a * b + c;   // DMAD
        b = b + c * a;   // DMAD
    }

    if (idx < N) {
        out[idx] = a + b + c;
    }
}

int main() {
    initializeCUPTI();

    int N = 1024;
    size_t bytes = N * sizeof(double);

    double *h_out = (double *)malloc(bytes);
    double *d_out;
    cudaMalloc(&d_out, bytes);

    printf("[LOG] bench_fp64_pure: %d iterations, pure FP64 DMAD/DADD/DMUL chain\n", ITERATIONS);
    collectTimestampOffsets();

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    bench_fp64_pure<<<grid_size, block_size>>>(d_out, N);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    flushCUPTIBuffers();
    printKernelTiming();

    cudaFree(d_out);
    free(h_out);
    return 0;
}
