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
#include <thread>
#include <vector>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


// NVML headers
#include <nvml.h>

#ifndef SIZE_M
#define SIZE_M 1024
#endif

#ifndef SIZE_N
#define SIZE_N 1024
#endif

#ifndef SIZE_K
#define SIZE_K 1024
#endif

#define BLOCK_SIZE 32

#ifndef REPEAT_TIMES
#define REPEAT_TIMES 500
#endif

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

inline void randomize_matrix(float *mat, int n) {
  std::srand(static_cast<unsigned int>(0));

  for (int i = 0; i < n; i++) {
    float tmp = static_cast<float>(std::rand() % 5) +
                0.01f * static_cast<float>(std::rand() % 5);
    tmp = (std::rand() % 2 == 0) ? tmp : -tmp;
    mat[i] = tmp;
  }
}

__global__ void sgemm_naive(int _M, int _N, int _K, float alpha, const float *A,
            const float *B, float beta, float *C) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < _M && y < _N) {
    META_LOOP(iter_loop, REPEAT_TIMES, REPEAT_TIMES, false);
    for (int _rep = 0; _rep < REPEAT_TIMES; _rep++) {
      float tmp = 0.0;
      META_LOOP(main_loop, SIZE_K, SIZE_K, false);
      for (int i = 0; i < _K; ++i) {
        tmp += A[x * _K + i] * B[i * _N + y];
      }
      C[x * _N + y] = alpha * tmp + beta * C[x * _N + y];
    }
  }
}

float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

float *A = nullptr, *B = nullptr, *C = nullptr; // host matrices
float *dA = nullptr, *dB = nullptr, *dC = nullptr;

int main(int argc, char *argv[]) {
  A = (float *)malloc(sizeof(float) * SIZE_M * SIZE_K);
  B = (float *)malloc(sizeof(float) * SIZE_K * SIZE_N);
  C = (float *)malloc(sizeof(float) * SIZE_M * SIZE_N);

  cudaMalloc((void **)&dA, sizeof(float) * SIZE_M * SIZE_K);
  cudaMalloc((void **)&dB, sizeof(float) * SIZE_K * SIZE_N);
  cudaMalloc((void **)&dC, sizeof(float) * SIZE_M * SIZE_N);
  
  // Should we pr itteration instead?
  randomize_matrix(A, SIZE_M * SIZE_K);
  randomize_matrix(B, SIZE_K * SIZE_N);
  randomize_matrix(C, SIZE_M * SIZE_N);
  
  cudaMemcpy(dA, A, sizeof(float) * SIZE_M * SIZE_K,
                       cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float) * SIZE_K * SIZE_N,
                       cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, sizeof(float) * SIZE_M * SIZE_N,
                       cudaMemcpyHostToDevice);

  METRICS_KERNEL_START
  
  dim3 gridDim(CEIL_DIV(SIZE_M, BLOCK_SIZE), CEIL_DIV(SIZE_N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); 
  printf("[INFO] gridDim: (%u, %u)\n", gridDim.x, gridDim.y);
  printf("[INFO] blockDim: (%u, %u)\n", blockDim.x, blockDim.y);

  sgemm_naive<<<gridDim, blockDim>>>(SIZE_M, SIZE_N, SIZE_K, alpha, dA, dB, beta, dC);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("[ERROR] Failed to launch vector_mul kernel (error code %s)!\n", cudaGetErrorString(err));
      return -1;
  }

  cudaDeviceSynchronize();
  printf("Kernel completeds\n");

  // Export launch configuration
  EXPORT_N("gridDim_x", gridDim.x);
  EXPORT_N("gridDim_y", gridDim.y);
  EXPORT_N("gridDim_z", gridDim.z);
  EXPORT_N("blockDim_x", blockDim.x);
  EXPORT_N("blockDim_y", blockDim.y);
  EXPORT_N("blockDim_z", blockDim.z);

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
