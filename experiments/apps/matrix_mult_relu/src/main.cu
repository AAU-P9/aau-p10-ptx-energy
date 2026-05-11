#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define TILE 16

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
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float value = 0.0f;

    // Tiled matrix multiplication
    for (int t = 0; t < (K + TILE - 1) / TILE; t++)
    {
        int tiledRow = row;
        int tiledCol = t * TILE + threadIdx.x;

        if (tiledRow < M && tiledCol < K)
            As[threadIdx.y][threadIdx.x] =
                A[tiledRow * K + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * TILE + threadIdx.y;
        tiledCol = col;

        if (tiledRow < K && tiledCol < N)
            Bs[threadIdx.y][threadIdx.x] =
                B[tiledRow * N + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++)
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
    const int M = 256;
    const int N = 256;
    const int K = 256;

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_bias(N, 0.5f);
    std::vector<float> h_C(M * N);

    float *d_A, *d_B, *d_bias, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_bias, N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(
        d_A,
        h_A.data(),
        M * K * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B.data(),
        K * N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_bias,
        h_bias.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 block(TILE, TILE);

    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    matmulBiasRelu<<<grid, block>>>(
        d_A,
        d_B,
        d_bias,
        d_C,
        M,
        N,
        K
    );

    cudaMemcpy(
        h_C.data(),
        d_C,
        M * N * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::cout << "C[0] = " << h_C[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);

    return 0;
}