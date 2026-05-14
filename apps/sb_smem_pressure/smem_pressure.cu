#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 10000000
#define SMEM_FLOATS 3072  // 12KB per block -> limits to ~4 blocks/SM on AD103

// Shared memory stress: repeated read/write with bank conflict pattern.
// 12KB smem/block limits occupancy via smem partition (~4 blocks/SM).
// Alternates stride-1 (no conflicts) and stride-2 (2-way conflicts)
// to measure bank conflict power overhead.
__global__ void KERNEL_LAUNCH_BOUNDS(256, 4)
bench_smem_pressure(float *out, int N) {
    KERNEL_META(bench_smem_pressure,
        _META_FRAG_PARAM_PTR(0, out, f32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(1, N, i32, dim_n, RANGE(1, 1024) STRIDE(1))
        _META_FRAG_SHARED(smem, f32, 12288)
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(N+255)/256 1 1")
        _META_FRAG_CUSTOM("category", "occupancy_smem")
        _META_FRAG_CUSTOM("arithmetic_intensity", "low")
        _META_FRAG_CUSTOM("memory_pattern", "shared_bank_conflict_alternating")
        _META_FRAG_CUSTOM("smem_per_block_bytes", "12288")
    );

    __shared__ float smem[SMEM_FLOATS];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    DEVICE_ASSUME(N > 0);
    DEVICE_ASSUME(N <= 1024);

    // Initialize shared memory
    for (int j = tid; j < SMEM_FLOATS; j += blockDim.x) {
        smem[j] = (float)j * 0.001f;
    }
    __syncthreads();

    float acc = 0.0f;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; ++i) {
        // Even iterations: stride-1 (no bank conflicts)
        // Odd iterations:  stride-2 (2-way bank conflicts)
        int stride = 1 + (i & 1);
        int addr = (tid * stride) % SMEM_FLOATS;

        acc += smem[addr];
        smem[(addr + 1) % SMEM_FLOATS] = acc * 0.999f;
    }

    if (idx < N) {
        out[idx] = acc;
    }
}

int main() {
    METRICS_KERNEL_START

    int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_out = (float *)malloc(bytes);
    float *d_out;
    cudaMalloc(&d_out, bytes);

    printf("[LOG] bench_smem_pressure: %d iterations, 12KB smem/block, bank conflict pattern\n", ITERATIONS);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    bench_smem_pressure<<<grid_size, block_size>>>(d_out, N);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    METRICS_KERNEL_END

    cudaFree(d_out);
    free(h_out);
    return 0;
}
