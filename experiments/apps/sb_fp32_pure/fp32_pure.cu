#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 100000000

// Pure FP32 arithmetic: FMAD/FADD/FMUL chain with no memory access.
// Measures baseline FP32 power cost per instruction.
__global__ void KERNEL_LAUNCH_BOUNDS(256, 4)
bench_fp32_pure(float *out, int N) {
    KERNEL_META(bench_fp32_pure,
        _META_FRAG_PARAM_PTR(0, out, f32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(1, N, i32, dim_n, RANGE(1, 1024) STRIDE(1))
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(N+255)/256 1 1")
        _META_FRAG_CUSTOM("category", "compute_fp32")
        _META_FRAG_CUSTOM("arithmetic_intensity", "inf")
        _META_FRAG_CUSTOM("memory_pattern", "none")
    );

    DEVICE_ASSUME(N > 0);
    DEVICE_ASSUME(N <= 1024);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0001f + (float)idx * 0.00001f;
    float b = 0.9999f;
    float c = 1.0f;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; ++i) {
        a = a * b + c;   // FMAD
        b = b + a * c;   // FMAD
        c = c * a + b;   // FMAD
        a = a + b;       // FADD
        b = b * c;       // FMUL
        c = c + a;       // FADD
        a = a * b + c;   // FMAD
        b = b + c * a;   // FMAD
    }

    if (idx < N) {
        out[idx] = a + b + c;
    }
}

int main() {
    METRICS_KERNEL_START

    int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_out = (float *)malloc(bytes);
    float *d_out;
    cudaMalloc(&d_out, bytes);

    printf("[LOG] bench_fp32_pure: %d iterations, pure FP32 FMAD/FADD/FMUL chain\n", ITERATIONS);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    bench_fp32_pure<<<grid_size, block_size>>>(d_out, N);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    METRICS_KERNEL_END

    cudaFree(d_out);
    free(h_out);
    return 0;
}
