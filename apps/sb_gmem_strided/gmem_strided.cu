#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#ifndef ITERATIONS
#define ITERATIONS 10000000
#endif
#define DATA_LEN (1024 * 1024)
#define STRIDE 128  // 128 * 4B = 512B between warp-adjacent threads

// Strided global memory reads: worst-case coalescing.
// Each thread in a warp reads 512 bytes apart -> 32 different cache lines per warp load.
// Compare against bench_gmem_coalesced to isolate memory controller overhead,
// TLB misses, and DRAM page open cost.
__global__ void KERNEL_LAUNCH_BOUNDS(256, 4)
bench_gmem_strided(const float * __restrict__ data, float *out, int dataLen) {
    KERNEL_META(bench_gmem_strided,
        _META_FRAG_PARAM_PTR(0, data, f32, input, READONLY NOALIAS ALIGN(128))
        _META_FRAG_PARAM_PTR(1, out, f32, output, WRITEONLY ALIGN(128))
        _META_FRAG_PARAM_INT(2, dataLen, i32, dim_n, RANGE(1, 1048576) MULTIPLE(256))
        _META_FRAG_LOOP(main_loop, ITERATIONS, ITERATIONS, false)
        _META_FRAG_LAUNCH(256, 1, 1, "(dataLen+255)/256 1 1")
        _META_FRAG_LAYOUT(data, linear, "dataLen")
        _META_FRAG_CUSTOM("category", "memory_bandwidth")
        _META_FRAG_CUSTOM("arithmetic_intensity", "0.25")
        _META_FRAG_CUSTOM("memory_pattern", "strided_uncoealesced")
        _META_FRAG_CUSTOM("access_stride_bytes", "512")
    );

    DEVICE_ASSUME(dataLen > 0);
    ASSUME_MULTIPLE(dataLen, 256);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    float acc = 0.0f;

    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; ++i) {
        int addr = ((idx * STRIDE) + i) % dataLen;
        acc += data[addr];
    }

    if (idx < dataLen) {
        out[idx] = acc;
    }
}

int main() {
    METRICS_KERNEL_START

    int dataLen = DATA_LEN;
    size_t dataBytes = dataLen * sizeof(float);

    float *h_data = (float *)malloc(dataBytes);
    float *h_out  = (float *)malloc(dataBytes);
    for (int i = 0; i < dataLen; i++) {
        h_data[i] = (float)(i % 1000) * 0.001f;
    }

    float *d_data, *d_out;
    cudaMalloc(&d_data, dataBytes);
    cudaMalloc(&d_out,  dataBytes);
    cudaMemcpy(d_data, h_data, dataBytes, cudaMemcpyHostToDevice);

    printf("[LOG] bench_gmem_strided: %d iterations, stride=%d, worst-case coalescing\n", ITERATIONS, STRIDE);

    int block_size = 256;
    int grid_size = (dataLen + block_size - 1) / block_size;
    bench_gmem_strided<<<grid_size, block_size>>>(d_data, d_out, dataLen);

    cudaMemcpy(h_out, d_out, dataBytes, cudaMemcpyDeviceToHost);

    EXPORT_N("gridDim_x",  grid_size);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", block_size);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    METRICS_KERNEL_END

    cudaFree(d_data);
    cudaFree(d_out);
    free(h_data);
    free(h_out);
    return 0;
}
