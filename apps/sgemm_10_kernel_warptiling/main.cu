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


void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K10_NUM_THREADS = 128;
  // const uint K10_BN = 128;
  // const uint K10_BM = 64;
  // const uint K10_BK = 16;
  // const uint K10_WN = 64;
  // const uint K10_WM = 32;
  // const uint K10_WNITER = 1;
  // const uint K10_TN = 4;
  // const uint K10_TM = 4;
  // Settings for A6000
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // warpsubtile in warptile
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
void run_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
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
  // inlined former benchmark(); kernel loops REPEAT_TIMES internally

#ifndef REPEAT_TIMES
#define REPEAT_TIMES 500
#endif
  int repeat_times = 1; // kernel now loops REPEAT_TIMES internally
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
  cudaDeviceSynchronize();
  EXPORT_N("gridDim_x", CEIL_DIV(SIZE_MATRIX, 128));
  EXPORT_N("gridDim_y", CEIL_DIV(SIZE_MATRIX, 128));
  EXPORT_N("gridDim_z", 1);
  EXPORT_N("blockDim_x", 128);
  EXPORT_N("blockDim_y", 1);
  EXPORT_N("blockDim_z", 1);
  METRICS_KERNEL_END

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
