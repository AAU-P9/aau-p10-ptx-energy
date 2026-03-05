#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cupti_timing.h"

#define ITERATIONS 10000000
#define N 1024

// Vector add kernel: out[i] = a[i] + b[i]
__global__ void vector_add(float *a, float *b, float *out)
{
    for(int i = 0; i < ITERATIONS; ++i) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < N) {
            out[idx] = a[idx] + b[idx];
        }
    }
}

int main()
{
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

    collectTimestampOffsets();
    
    // Launch kernel: 1 block with 256 threads
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_out);
    
    // Copy results back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    flushCUPTIBuffers();
    printKernelTiming();
    
    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);
    
    return 0;
}
