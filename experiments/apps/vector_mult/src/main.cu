#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 10000000

__global__ void vector_mul(float *A, float *B, float *C, int N) {
    META_LOOP(iterations, ITERATIONS, 0, false);
    for (int i = 0; i < ITERATIONS; i++) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < N) {
            C[index] = A[index] * B[index];
        }
    }
}

int main(int argc, char *argv[])
{
    // Read the size of the vectors (N) from command line arguments
    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]); // 1 million elements

    printf("Running vector multiplication with %d iterations...\n", ITERATIONS);

    // Initialize CUPTI profiling
    initializeCUPTI();

    size_t bytes = N * sizeof(float);

    // Host vectors
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    
    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
        h_out[i] = 0.0f;
    }
    
    // Device vectors
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);
    
    // Get CPU/GPU offsets
    collectTimestampOffsets();

    // Launch kernel: 1 block with 256 threads
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    
    vector_mul<<<grid_size, block_size>>>(d_a, d_b, d_out, N);

    printf("[LOG] Kernel launched. Waiting for completion...\n");

    // Verify that the kernel didn't have any errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to launch vector_mul kernel (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Check for kernel launch errors
    cudaDeviceSynchronize();
    // Copy results back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Flush all activity buffers
    flushCUPTIBuffers();

    printKernelTiming();
    
    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);

    // Disable CUPTI profiling
    disableCUPTI();
    
    return 0;
}