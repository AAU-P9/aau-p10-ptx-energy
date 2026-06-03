/*
 *  Benchmark: UBench 
 *  Author: Trasgo Research Group
 *  Source: https://trasgo.infor.uva.es/ubench/
 */
#include <atomic>
#include <chrono>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "kernel.cuh"
#include "simple_cuda_utils.h"
#include "cupti_timing.h"

// NVML headers
#include <nvml.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
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
  EXPORT_N("gridDim_x", CEIL_DIV(SIZE_MATRIX, 128));
  EXPORT_N("gridDim_y", CEIL_DIV(SIZE_MATRIX, 128));
  EXPORT_N("gridDim_z", 1);
  EXPORT_N("blockDim_x", 256);
  EXPORT_N("blockDim_y", 1);
  EXPORT_N("blockDim_z", 1);
  METRICS_KERNEL_END

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}

