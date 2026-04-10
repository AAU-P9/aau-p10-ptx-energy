#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 10000000

// Arguments for problem size
#ifndef SIZE_N
#define SIZE_N 1024
#endif

// Vector add kernel: out[i] = a[i] + b[i]
__global__ void vector_add(float *a, float   *b, float *out, int _N)
{
    META_LOOP(iterations, ITERATIONS, 0, false);
    for(int i = 0; i < ITERATIONS; ++i) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < _N) {
            out[idx] = a[idx] + b[idx];
        }
    }
}

int main(int argc, char *argv[])
{
    initializeCUPTI();
    
    size_t bytes = SIZE_N * sizeof(float);
    
    // Host vectors
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    
    // Initialize host vectors
    for (int i = 0; i < SIZE_N; i++) {
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
    
    // Launch kernel: 1 block with 1024 threads
    int block_size = 1024;
    int grid_size = (SIZE_N + block_size - 1) / block_size;
    
    printf("[LOG] Launching kernel with grid size %d and block size %d...\n", grid_size, block_size);

    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_out, SIZE_N);

    printf("[LOG] Kernel launched. Waiting for completion...\n");

    // Verify that the kernel didn't have any errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to launch vector_add kernel (error code %s)!\n", cudaGetErrorString(err));
        return -1;
    }
    
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
