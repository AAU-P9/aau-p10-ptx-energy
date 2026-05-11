#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "cupti_timing.h"
#include "ptx_meta.h"

#ifndef SIZE_TILE
#define SIZE_TILE 32
#endif

#ifndef SIZE_M
#define SIZE_M 256
#endif

#ifndef SIZE_N
#define SIZE_N 256
#endif

#ifndef SIZE_K
#define SIZE_K 256
#endif

__global__
void matmulBiasRelu(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float As[SIZE_TILE][SIZE_TILE];
    __shared__ float Bs[SIZE_TILE][SIZE_TILE];

    int row = blockIdx.y * SIZE_TILE + threadIdx.y;
    int col = blockIdx.x * SIZE_TILE + threadIdx.x;

    float value = 0.0f;

    // Tiled matrix multiplication
    // Loop over tiles: between 1 and ceil(K / TILE) iterations
    META_LOOP(tile_matmul_loop, 1, (SIZE_K + SIZE_TILE - 1) / SIZE_TILE, false);
    for (int t = 0; t < (K + SIZE_TILE - 1) / SIZE_TILE; t++)
    {
        int tiledRow = row;
        int tiledCol = t * SIZE_TILE + threadIdx.x;

        if (tiledRow < M && tiledCol < K)
            As[threadIdx.y][threadIdx.x] =
                A[tiledRow * K + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * SIZE_TILE + threadIdx.y;
        tiledCol = col;

        if (tiledRow < K && tiledCol < N)
            Bs[threadIdx.y][threadIdx.x] =
                B[tiledRow * N + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        
        META_LOOP(tilez_matmul_loop, SIZE_TILE, SIZE_TILE, false);
        for (int i = 0; i < SIZE_TILE; i++)
        {
            value +=
                As[threadIdx.y][i] *
                Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Add bias + ReLU activation
    if (row < M && col < N)
    {
        value += bias[col];

        // ReLU
        value = (value > 0.0f) ? value : 0.0f;

        C[row * N + col] = value;
    }
}

int main()
{
    METRICS_KERNEL_START

    std::vector<float> h_A(SIZE_M * SIZE_K, 1.0f);
    std::vector<float> h_B(SIZE_K * SIZE_N, 1.0f);
    std::vector<float> h_bias(SIZE_N, 0.5f);
    std::vector<float> h_C(SIZE_M * SIZE_N);

    float *d_A, *d_B, *d_bias, *d_C;

    cudaMalloc(&d_A, SIZE_M * SIZE_K * sizeof(float));
    cudaMalloc(&d_B, SIZE_K * SIZE_N * sizeof(float));
    cudaMalloc(&d_bias, SIZE_N * sizeof(float));
    cudaMalloc(&d_C, SIZE_M * SIZE_N * sizeof(float));

    cudaMemcpy(
        d_A,
        h_A.data(),
        SIZE_M * SIZE_K * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B.data(),
        SIZE_K * SIZE_N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_bias,
        h_bias.data(),
        SIZE_N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 block(SIZE_TILE, SIZE_TILE);

    dim3 grid(
        (SIZE_N + SIZE_TILE - 1) / SIZE_TILE,
        (SIZE_M + SIZE_TILE - 1) / SIZE_TILE
    );

    matmulBiasRelu<<<grid, block>>>(
        d_A,
        d_B,
        d_bias,
        d_C,
        SIZE_M,
        SIZE_N,
        SIZE_K
    );

    cudaMemcpy(
        h_C.data(),
        d_C,
        SIZE_M * SIZE_N * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::cout << "C[0] = " << h_C[0] << std::endl;

    // Export launch configuration
    EXPORT_N("gridDim_x", grid.x);
    EXPORT_N("gridDim_y", grid.y);
    EXPORT_N("gridDim_z", grid.z);
    EXPORT_N("blockDim_x", block.x);
    EXPORT_N("blockDim_y", block.y);
    EXPORT_N("blockDim_z", block.z);

    METRICS_KERNEL_END

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);

    return 0;
}