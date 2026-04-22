#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>  // for fabs()
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 1000000

#ifndef SIZE_M
#define SIZE_M 256
#endif

#ifndef SIZE_N
#define SIZE_N 256
#endif

#ifndef SIZE_K
#define SIZE_K 256
#endif

__global__ void matrix_mul(float *A, float *B, float *C, int M, int N, int K) {
    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for(int k = 0; k < ITERATIONS; ++k) {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;

        if (row < M && col < K) {
            META_LOOP(inner_loop, SIZE_N, SIZE_N, false);
            for (int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}

int main(int argc, char *argv[]) {

    METRICS_KERNEL_START
    
    size_t bytes_a = SIZE_M * SIZE_N * sizeof(float);
    size_t bytes_b = SIZE_N * SIZE_K * sizeof(float);
    size_t bytes_c = SIZE_M * SIZE_K * sizeof(float);

    // Host matrices
    float *h_a = (float *)malloc(bytes_a);
    float *h_b = (float *)malloc(bytes_b);
    float *h_c = (float *)malloc(bytes_c);

    // Initialize host matrices
    for (int i = 0; i < SIZE_M * SIZE_N; i++) {
        h_a[i] = (float)(i % 100);  // some values for testing
    }
    for (int i = 0; i < SIZE_N * SIZE_K; i++) {
        h_b[i] = (float)(i % 100);  // some values for testing
    }
    for (int i = 0; i < SIZE_M * SIZE_K; i++) {
        h_c[i] = 0.0f;  // initialize result matrix to 0
    }

    // Device matrices
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    // Define block size (e.g., 16x16 threads per block)
    dim3 block_size(32, 32);  // 32x32 threads per block
    dim3 grid_size((SIZE_K + block_size.x - 1) / block_size.x, 
                   (SIZE_M + block_size.y - 1) / block_size.y);

    printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

    printf("[LOG] Launching kernel with grid size (%d, %d) and block size (%d, %d)...\n", 
           grid_size.x, grid_size.y, block_size.x, block_size.y);

    // Launch kernel with 2D grid and block size
    matrix_mul<<<grid_size, block_size>>>(d_a, d_b, d_c, SIZE_M, SIZE_N, SIZE_K);

    // Check for errors after kernel launch
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);
 
    // Export launch configuration
    EXPORT_N("gridDim_x", grid_size.x);
    EXPORT_N("gridDim_y", grid_size.y);
    EXPORT_N("gridDim_z", grid_size.z);
    EXPORT_N("blockDim_x", block_size.x);
    EXPORT_N("blockDim_y", block_size.y);
    EXPORT_N("blockDim_z", block_size.z);

    METRICS_KERNEL_END

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}