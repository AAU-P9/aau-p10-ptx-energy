#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 10000000

// Pure INT32 arithmetic: IMAD/IADD/bitwise chain.
// On Ada, INT32 can co-issue with FP32 on separate datapath.
// This isolates integer pipe power contribution.
__global__ void KERNEL_LAUNCH_BOUNDS(256, 4)
bench_int32_pure(int *out, int N) {
    KERNEL_META(bench_int32_pure,
        _META_FRAG_PARAM_PTR(0, out, i32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(1, N, i32, dim_n, RANGE(1, 1024) STRIDE(1))
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(N+255)/256 1 1")
        _META_FRAG_CUSTOM("category", "compute_int32")
        _META_FRAG_CUSTOM("arithmetic_intensity", "inf")
        _META_FRAG_CUSTOM("memory_pattern", "none")
    );

    DEVICE_ASSUME(N > 0);
    DEVICE_ASSUME(N <= 1024);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a = idx + 1;
    int b = 37;
    int c = 73;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
    for (int i = 0; i < ITERATIONS; ++i) {
        a = a * b + c;   // IMAD
        b = b + a * c;   // IMAD
        c = c * a + b;   // IMAD
        a = a + b;       // IADD
        b = b * c;       // IMUL
        c = c + a;       // IADD
        a = a ^ b;       // XOR
        b = b | c;       // OR
    }

    if (idx < N) {
        out[idx] = a + b + c;
    }
}

int main() {
    initializeCUPTI();

    int N = 1024;
    size_t bytes = N * sizeof(int);

    int *h_out = (int *)malloc(bytes);
    int *d_out;
    cudaMalloc(&d_out, bytes);

    printf("[LOG] bench_int32_pure: %d iterations, pure INT32 IMAD/IADD/bitwise chain\n", ITERATIONS);
    collectTimestampOffsets();

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    bench_int32_pure<<<grid_size, block_size>>>(d_out, N);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    flushCUPTIBuffers();
    printKernelTiming();

    cudaFree(d_out);
    free(h_out);
    return 0;
}
