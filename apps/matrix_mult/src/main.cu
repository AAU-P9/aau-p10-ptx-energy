#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>  // for fabs()
#include "cupti_timing.h"


#define ITERATIONS 1000

#define M 1024
#define N 1024
#define K 1024


__global__ void matrix_mul(float *A, float *B, float *C) {
    for(int i = 0; i < ITERATIONS; ++i) {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;

        if (row < M && col < K) {
            for (int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}

int main() {
    initializeCUPTI();
    
    size_t bytes_a = M * N * sizeof(float);
    size_t bytes_b = N * K * sizeof(float);
    size_t bytes_c = M * K * sizeof(float);

    // Host matrices
    float *h_a = (float *)malloc(bytes_a);
    float *h_b = (float *)malloc(bytes_b);
    float *h_c = (float *)malloc(bytes_c);

    // Initialize host matrices
    for (int i = 0; i < M * N; i++) {
        h_a[i] = (float)(i % 100);  // some values for testing
    }
    for (int i = 0; i < N * K; i++) {
        h_b[i] = (float)(i % 100);  // some values for testing
    }
    for (int i = 0; i < M * K; i++) {
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
    dim3 block_size(16, 16);
    dim3 grid_size((K + block_size.x - 1) / block_size.x, 
                   (M + block_size.y - 1) / block_size.y);

    printf("[LOG] Running kernel with %d iterations...\n");

    collectTimestampOffsets();

    // Launch kernel with 2D grid and block size
    matrix_mul<<<grid_size, block_size>>>(d_a, d_b, d_c);

    // Check for errors after kernel launch
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    // Verify results
    int errors = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float expected = 0.0f;
            for (int l = 0; l < N; l++) {
                expected += h_a[i * N + l] * h_b[l * K + j];
            }
            if (fabs(h_c[i * K + j] - expected) > 1e-6) {
                errors++;
                if (errors <= 5) {
                    printf("Error at (%d, %d): expected %f, got %f\n", i, j, expected, h_c[i * K + j]);
                }
            }
        }
    }

    if (errors == 0) {
        printf("Matrix multiplication kernel executed successfully!\n");
    } else {
        printf("Test failed with %d errors\n", errors);
    }

    flushCUPTIBuffers();
    printKernelTiming();

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}