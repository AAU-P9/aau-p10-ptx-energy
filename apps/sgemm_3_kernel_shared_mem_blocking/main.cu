/*
 *  Benchmark: UBench
 *  Author: Trasgo Research Group
 *  Source: https://trasgo.infor.uva.es/ubench/
 */
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <vector>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include "cupti_timing.h"
#include "simple_cuda_utils.h"
#include "kernel.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// NVML headers
#include <nvml.h>


void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int M, int N, int K, float alpha, float *A, float *B,
                float beta, float *C) {
  run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
}

// cuBLAS FLOPs ceiling is reached at 8192
#ifndef SIZE_MATRIX
#define SIZE_MATRIX 1024
#endif
std::vector<int> SIZE = {SIZE_MATRIX, 
    /*256, 512, 1024, 2048, 4096,*/
                         /*  8192 */
                        };

long m, n, k;
long max_size = SIZE[SIZE.size() - 1];

float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

float *A = nullptr, *B = nullptr, *C = nullptr; // host matrices
float *dA = nullptr, *dB = nullptr, *dC = nullptr;

void benchmark() {

#ifndef REPEAT_TIMES
#define REPEAT_TIMES 500
#endif
  int repeat_times = REPEAT_TIMES;
  for (int size : SIZE) {
    m = n = k = size;

    cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    for (int j = 0; j < repeat_times; j++) {
      // We don't reset dC between runs to save time
      run_kernel(m, n, k, alpha, dA, dB, beta, dC);
      cudaCheck(cudaDeviceSynchronize());
    }

    cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
  }

  // Free up CPU and GPU space
}

int main(int argc, char *argv[]) {

  A = (float *)malloc(sizeof(float) * max_size * max_size);
  B = (float *)malloc(sizeof(float) * max_size * max_size);
  C = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  METRICS_KERNEL_START
  benchmark();
  cudaDeviceSynchronize();
  EXPORT_N("gridDim_x", CEIL_DIV(SIZE_MATRIX, 32));
  EXPORT_N("gridDim_y", CEIL_DIV(SIZE_MATRIX, 32));
  EXPORT_N("gridDim_z", 1);
  EXPORT_N("blockDim_x", 32*32);
  EXPORT_N("blockDim_y", 1);
  EXPORT_N("blockDim_z", 1);
  METRICS_KERNEL_END

  free(A);
  free(B);
  free(C);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
