#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 10000000

// Register pressure: 32 live float variables -> ~40+ regs/thread.
// On AD103 (65536 regs/SM), at 256 threads/block this caps at
// ~6 warps/SM = ~37% occupancy vs 100% at default reg count.
// Power should show reduced SM utilization but sustained per-thread compute.
__global__ void KERNEL_LAUNCH_BOUNDS(256, 1)
bench_reg_pressure(float *out, int N) {
    KERNEL_META(bench_reg_pressure,
        _META_FRAG_PARAM_PTR(0, out, f32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(1, N, i32, dim_n, RANGE(1, 1024) STRIDE(1))
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(N+255)/256 1 1")
        _META_FRAG_CUSTOM("category", "occupancy_registers")
        _META_FRAG_CUSTOM("arithmetic_intensity", "inf")
        _META_FRAG_CUSTOM("memory_pattern", "none")
        _META_FRAG_CUSTOM("live_fp32_regs", "32")
        _META_FRAG_CUSTOM("target_occupancy", "low")
    );

    DEVICE_ASSUME(N > 0);
    DEVICE_ASSUME(N <= 1024);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float seed = (float)(idx + 1) * 0.00001f;

    // 32 live registers — compiler must keep all alive
    float r00 = seed + 0.01f;  float r01 = seed + 0.02f;
    float r02 = seed + 0.03f;  float r03 = seed + 0.04f;
    float r04 = seed + 0.05f;  float r05 = seed + 0.06f;
    float r06 = seed + 0.07f;  float r07 = seed + 0.08f;
    float r08 = seed + 0.09f;  float r09 = seed + 0.10f;
    float r10 = seed + 0.11f;  float r11 = seed + 0.12f;
    float r12 = seed + 0.13f;  float r13 = seed + 0.14f;
    float r14 = seed + 0.15f;  float r15 = seed + 0.16f;
    float r16 = seed + 0.17f;  float r17 = seed + 0.18f;
    float r18 = seed + 0.19f;  float r19 = seed + 0.20f;
    float r20 = seed + 0.21f;  float r21 = seed + 0.22f;
    float r22 = seed + 0.23f;  float r23 = seed + 0.24f;
    float r24 = seed + 0.25f;  float r25 = seed + 0.26f;
    float r26 = seed + 0.27f;  float r27 = seed + 0.28f;
    float r28 = seed + 0.29f;  float r29 = seed + 0.30f;
    float r30 = seed + 0.31f;  float r31 = seed + 0.32f;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; ++i) {
        r00 = r00 * r01 + r02;  r01 = r01 * r02 + r03;
        r02 = r02 * r03 + r04;  r03 = r03 * r04 + r05;
        r04 = r04 * r05 + r06;  r05 = r05 * r06 + r07;
        r06 = r06 * r07 + r08;  r07 = r07 * r08 + r09;
        r08 = r08 * r09 + r10;  r09 = r09 * r10 + r11;
        r10 = r10 * r11 + r12;  r11 = r11 * r12 + r13;
        r12 = r12 * r13 + r14;  r13 = r13 * r14 + r15;
        r14 = r14 * r15 + r16;  r15 = r15 * r16 + r17;
        r16 = r16 * r17 + r18;  r17 = r17 * r18 + r19;
        r18 = r18 * r19 + r20;  r19 = r19 * r20 + r21;
        r20 = r20 * r21 + r22;  r21 = r21 * r22 + r23;
        r22 = r22 * r23 + r24;  r23 = r23 * r24 + r25;
        r24 = r24 * r25 + r26;  r25 = r25 * r26 + r27;
        r26 = r26 * r27 + r28;  r27 = r27 * r28 + r29;
        r28 = r28 * r29 + r30;  r29 = r29 * r30 + r31;
        r30 = r30 * r31 + r00;  r31 = r31 * r00 + r01;
    }

    if (idx < N) {
        out[idx] = r00 + r01 + r02 + r03 + r04 + r05 + r06 + r07 +
                   r08 + r09 + r10 + r11 + r12 + r13 + r14 + r15 +
                   r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 +
                   r24 + r25 + r26 + r27 + r28 + r29 + r30 + r31;
    }
}

int main() {
    initializeCUPTI();

    int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_out = (float *)malloc(bytes);
    float *d_out;
    cudaMalloc(&d_out, bytes);

    printf("[LOG] bench_reg_pressure: %d iterations, 32 live FP32 regs, reduced occupancy\n", ITERATIONS);
    collectTimestampOffsets();

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    bench_reg_pressure<<<grid_size, block_size>>>(d_out, N);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    flushCUPTIBuffers();
    printKernelTiming();

    cudaFree(d_out);
    free(h_out);
    return 0;
}
