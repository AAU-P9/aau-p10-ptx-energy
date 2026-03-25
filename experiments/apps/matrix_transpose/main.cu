#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>  // for fabs()
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"


#define ITERATIONS 1000000

__global__ void matrix_transpose(float *A, float *B) {
    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    for (int i = 0; i < ITERATIONS; i++) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < N && y < N) {
            B[x * N + y] = A[y * N + x];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);

    initializeCUPTI();
    
    size_t bytes = N * N * sizeof(float);

    // Host matrices
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);

    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        h_a[i] = (float)(i % 100);  // some values for testing
    }
    for (int i = 0; i < N * N; i++) {
        h_b[i] = 0.0f;  // initialize result matrix to 0
    }

    // Device matrices
    float *d_a, *d_b;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    // Define block size (e.g., 16x16 threads per block)
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (N + block_size.y - 1) / block_size.y);

    printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

    collectTimestampOffsets();

    // Launch kernel with 2D grid and block size
    matrix_transpose<<<grid_size, block_size>>>(d_a, d_b);

    // Check for errors after kernel launch
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    flushCUPTIBuffers();
    printKernelTiming();

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    return 0;
}