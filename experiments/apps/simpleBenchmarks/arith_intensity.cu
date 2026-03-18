#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 500000
#define DATA_LEN (1024 * 1024)

// Compile-time configurable: -DFLOPS_PER_LOAD=N
// Sweep with 1, 4, 16, 64 to trace from memory-bound to compute-bound.
#ifndef FLOPS_PER_LOAD
#define FLOPS_PER_LOAD 16
#endif

// Controlled arithmetic intensity: exactly FLOPS_PER_LOAD FP32 ops per
// global memory load. At 16 FLOP / 4 bytes = 4 FLOP/byte, this sits
// near the roofline knee for AD103 (~700 GB/s BW, ~50 TFLOPS FP32).
__global__ void KERNEL_LAUNCH_BOUNDS(256, 4)
bench_arith_intensity(const float * __restrict__ data, float *out, int dataLen) {
    KERNEL_META(bench_arith_intensity,
        _META_FRAG_PARAM_PTR(0, data, f32, input, READONLY NOALIAS ALIGN(128))
        _META_FRAG_PARAM_PTR(1, out, f32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(2, dataLen, i32, dim_n, RANGE(1, 1048576) MULTIPLE(256))
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(dataLen+255)/256 1 1")
        _META_FRAG_LAYOUT(data, linear, "dataLen")
        _META_FRAG_CUSTOM("category", "roofline_sweep")
        _META_FRAG_CUSTOM("flops_per_load", _STR(FLOPS_PER_LOAD))
        _META_FRAG_CUSTOM("arithmetic_intensity_flop_per_byte", _STR(FLOPS_PER_LOAD) "/4")
        _META_FRAG_CUSTOM("memory_pattern", "coalesced_sequential")
    );

    DEVICE_ASSUME(dataLen > 0);
    ASSUME_MULTIPLE(dataLen, 256);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float acc = 0.0f;
    float tmp = 1.0001f + (float)idx * 0.00001f;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; ++i) {
        // One global load
        int addr = (idx + i * stride) % dataLen;
        float val = data[addr];

        // FLOPS_PER_LOAD compute ops before next load
        #pragma unroll
        for (int f = 0; f < FLOPS_PER_LOAD; ++f) {
            tmp = tmp * val + acc;  // FMAD
        }
        acc = tmp;
    }

    if (idx < dataLen) {
        out[idx] = acc;
    }
}

int main() {
    initializeCUPTI();

    int dataLen = DATA_LEN;
    size_t dataBytes = dataLen * sizeof(float);

    float *h_data = (float *)malloc(dataBytes);
    float *h_out  = (float *)malloc(dataBytes);
    for (int i = 0; i < dataLen; i++) {
        h_data[i] = 1.0f + (float)(i % 100) * 0.0001f;
    }

    float *d_data, *d_out;
    cudaMalloc(&d_data, dataBytes);
    cudaMalloc(&d_out,  dataBytes);
    cudaMemcpy(d_data, h_data, dataBytes, cudaMemcpyHostToDevice);

    printf("[LOG] bench_arith_intensity: %d iterations, %d FLOP/load, AI=%.1f FLOP/byte\n",
           ITERATIONS, FLOPS_PER_LOAD, (float)FLOPS_PER_LOAD / (float)sizeof(float));
    collectTimestampOffsets();

    int block_size = 256;
    int grid_size = (dataLen + block_size - 1) / block_size;
    bench_arith_intensity<<<grid_size, block_size>>>(d_data, d_out, dataLen);

    cudaMemcpy(h_out, d_out, dataBytes, cudaMemcpyDeviceToHost);

    flushCUPTIBuffers();
    printKernelTiming();

    cudaFree(d_data);
    cudaFree(d_out);
    free(h_data);
    free(h_out);
    return 0;
}
