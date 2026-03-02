#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cupti_timing.h"

#define ITERATIONS 10000000

__global__ void vector_mul(float *A, float *B, float *C, int N) {
    for (int i = 0; i < ITERATIONS; i++) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < N) {
            C[index] = A[index] * B[index];
        }
    }
}

int main()
{
    // Initialize CUPTI profiling
    initializeCUPTI();

    int n = 1024;
    size_t bytes = n * sizeof(float);

    // Host vectors
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    
    // Initialize host vectors
    for (int i = 0; i < n; i++) {
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
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_mul<<<grid_size, block_size>>>(d_a, d_b, d_out, n);
    
    // Check for kernel launch errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy results back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] * h_b[i];  // Correct expected value (multiplication, not addition)
        if (fabs(h_out[i] - expected) > 1e-6) {  // Allow for small floating-point errors
            errors++;
            if (errors <= 5) {
                printf("Error at index %d: expected %f, got %f\n", i, expected, h_out[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("Vector multiplication kernel executed successfully!\n");
        printf("First 5 elements: %.2f, %.2f, %.2f, %.2f, %.2f\n", 
               h_out[0], h_out[1], h_out[2], h_out[3], h_out[4]);
    } else {
        printf("Test failed with %d errors\n", errors);
    }

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