#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 10000000

// Mixed FP32 + INT32 at 50/50 ratio.
// On Ada, INT32 and FP32 can co-issue on separate datapaths.
// Compare power against fp32_pure + int32_pure to measure
// whether co-issue is additive, sub-additive, or super-additive.
__global__ void KERNEL_LAUNCH_BOUNDS(256, 4)
bench_fp32_int32_mix(float *fout, int *iout, int N) {
    KERNEL_META(bench_fp32_int32_mix,
        _META_FRAG_PARAM_PTR(0, fout, f32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_PTR(1, iout, i32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(2, N, i32, dim_n, RANGE(1, 1024) STRIDE(1))
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(N+255)/256 1 1")
        _META_FRAG_CUSTOM("category", "compute_mixed_fp32_int32")
        _META_FRAG_CUSTOM("arithmetic_intensity", "inf")
        _META_FRAG_CUSTOM("mix_ratio", "50_50")
        _META_FRAG_CUSTOM("memory_pattern", "none")
    );

    DEVICE_ASSUME(N > 0);
    DEVICE_ASSUME(N <= 1024);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float fa = 1.0001f + (float)idx * 0.00001f;
    float fb = 0.9999f;
    float fc = 1.0f;

    int ia = idx + 1;
    int ib = 37;
    int ic = 73;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; ++i) {
        // FP32 ops (4 ops)
        fa = fa * fb + fc;   // FMAD
        fb = fb + fa * fc;   // FMAD
        fc = fc * fa + fb;   // FMAD
        fa = fa + fb;        // FADD

        // INT32 ops (4 ops)
        ia = ia * ib + ic;   // IMAD
        ib = ib + ia * ic;   // IMAD
        ic = ic * ia + ib;   // IMAD
        ia = ia + ib;        // IADD
    }

    if (idx < N) {
        fout[idx] = fa + fb + fc;
        iout[idx] = ia + ib + ic;
    }
}

int main() {
    METRICS_KERNEL_START

    int N = 1024;

    float *h_fout = (float *)malloc(N * sizeof(float));
    int   *h_iout = (int *)malloc(N * sizeof(int));
    float *d_fout;
    int   *d_iout;
    cudaMalloc(&d_fout, N * sizeof(float));
    cudaMalloc(&d_iout, N * sizeof(int));

    printf("[LOG] bench_fp32_int32_mix: %d iterations, 50/50 FP32+INT32 co-issue\n", ITERATIONS);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    bench_fp32_int32_mix<<<grid_size, block_size>>>(d_fout, d_iout, N);

    cudaMemcpy(h_fout, d_fout, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iout, d_iout, N * sizeof(int), cudaMemcpyDeviceToHost);

    METRICS_KERNEL_END

    cudaFree(d_fout);
    cudaFree(d_iout);
    free(h_fout);
    free(h_iout);
    return 0;
}
