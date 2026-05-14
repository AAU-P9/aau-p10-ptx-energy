#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

#ifndef N
#define N (1 << 20)
#endif

#define GRID_SIZE ((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2))
#define SHARED_BYTES (BLOCK_SIZE * 4) // float = 4 bytes

__global__
void reduceSumKernel(const float* input, float* output, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;

    // Load two elements per thread
    if (idx < n)
        sum += input[idx];

    if (idx + blockDim.x < n)
        sum += input[idx + blockDim.x];

    sdata[tid] = sum;

    __syncthreads();

    // Shared memory reduction
    META_LOOP(reduction_loop, (BLOCK_SIZE/2), (BLOCK_SIZE/2), false);
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // Warp-level reduction
    if (tid < 32)
    {
        volatile float* smem = sdata;

        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    // Write block result
    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

int main()
{
    METRICS_KERNEL_START

    const size_t bytes = N * sizeof(float);

    // Host input
    std::vector<float> h_input(N, 1.0f);

    // Device pointers
    float* d_input = nullptr;
    float* d_output = nullptr;

    // Allocate device memory
    cudaMalloc(&d_input, bytes);

    int gridSize = GRID_SIZE;

    cudaMalloc(&d_output, gridSize * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // Launch reduction kernel
    reduceSumKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, N);

    // Export launch configuration
    EXPORT_N("gridDim_x", gridSize);
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", BLOCK_SIZE);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);

    // Copy partial sums back
    std::vector<float> h_output(gridSize);

    cudaMemcpy(
        h_output.data(),
        d_output,
        gridSize * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Final reduction on CPU
    float finalSum = 0.0f;

    for (float v : h_output)
    {
        finalSum += v;
    }

    std::cout << "Final sum = " << finalSum << std::endl;

    METRICS_KERNEL_END

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}