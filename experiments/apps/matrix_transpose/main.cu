#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 1000000
#define TILE_DIM 32
#define SIZE_N 1024

__global__ void matrix_transpose(const float *A, float *B, int N) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Repeat operation for profiling purposes
    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; i++) {

        if (x < N && y < N) {
            // transpose
            B[y * N + x] = A[x * N + y];
        }
    }
}

int main(int argc, char *argv[]) {

    METRICS_KERNEL_START

    size_t bytes = (size_t)SIZE_N * SIZE_N * sizeof(float);

    // Host matrices
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);

    // Initialize host matrices
    for (int i = 0; i < SIZE_N * SIZE_N; i++) {
        h_a[i] = (float)(i % 100);
        h_b[i] = 0.0f;
    }

    // Device matrices
    float *d_a, *d_b;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    dim3 block_size(TILE_DIM, TILE_DIM);

    dim3 grid_size(
        (SIZE_N + block_size.x - 1) / block_size.x,
        (SIZE_N + block_size.y - 1) / block_size.y
    );

    printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

    matrix_transpose<<<grid_size, block_size>>>(d_a, d_b, SIZE_N);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    METRICS_KERNEL_END

    // Optional correctness check
    for (int y = 0; y < SIZE_N; y++) {
        for (int x = 0; x < SIZE_N; x++) {

            float expected = h_a[x * SIZE_N + y];
            float got      = h_b[y * SIZE_N + x];

            if (fabs(expected - got) > 1e-5f) {
                printf("Mismatch at (%d,%d): expected %f got %f\n",
                       x, y, expected, got);
            }
        }
    }

    printf("Transpose successful.\n");

    // Cleanup

    cudaFree(d_a);
    cudaFree(d_b);

    free(h_a);
    free(h_b);

    return 0;
}