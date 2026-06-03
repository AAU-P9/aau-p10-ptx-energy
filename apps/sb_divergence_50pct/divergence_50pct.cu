#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#ifndef ITERATIONS
#define ITERATIONS 10000000
#endif

// Warp divergence: 50% of lanes take if-branch, 50% take else-branch.
// Both paths do equal FP32 work but different operations -> warp serializes.
// Compare against bench_fp32_pure (same total ops, no divergence)
// to isolate the power cost of predicated execution / replay.
__global__ void KERNEL_LAUNCH_BOUNDS(256, 4)
bench_divergence_50pct(float *out, int N) {
    KERNEL_META(bench_divergence_50pct,
        _META_FRAG_PARAM_PTR(0, out, f32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(1, N, i32, dim_n, RANGE(1, 1024) STRIDE(1))
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(N+255)/256 1 1")
        _META_FRAG_CUSTOM("category", "divergence")
        _META_FRAG_CUSTOM("arithmetic_intensity", "inf")
        _META_FRAG_CUSTOM("memory_pattern", "none")
        _META_FRAG_CUSTOM("divergence_ratio", "0.50")
        _META_FRAG_CUSTOM("divergence_granularity", "half_warp")
    );

    DEVICE_ASSUME(N > 0);
    DEVICE_ASSUME(N <= 1024);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    float a = 1.0001f + (float)idx * 0.00001f;
    float b = 0.9999f;
    float c = 1.0f;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; ++i) {
        if (lane < 16) {
            // Path A: FMAD-heavy (4 ops)
            a = a * b + c;
            b = b * c + a;
            c = c * a + b;
            a = a * b + c;
        } else {
            // Path B: FADD/FMUL interleave (6 ops, ~same latency)
            a = a + b;
            b = b * c;
            c = c + a;
            a = a * b;
            b = b + c;
            c = c * a;
        }
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

    printf("[LOG] bench_divergence_50pct: %d iterations, 50%% warp divergence\n", ITERATIONS);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    bench_divergence_50pct<<<grid_size, block_size>>>(d_out, N);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    METRICS_KERNEL_END

    cudaFree(d_out);
    free(h_out);
    return 0;
}
