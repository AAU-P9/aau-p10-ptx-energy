/*
 *  Article: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog
 *  Author: Simon Boehm
 *  Source: https://siboehm.com/articles/22/CUDA-MMM
 */

#include <atomic>
#include <chrono>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <thread>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include "ptx_meta.h"
#include <vector>
#include "../../include/simple_cuda_utils.h"
#include "cupti_timing.h"

// NVML headers
#include <nvml.h>

#ifndef SIZE_M
#define SIZE_M 256
#endif

#ifndef SIZE_N
#define SIZE_N 256
#endif

#ifndef SIZE_K
#define SIZE_K 256
#endif

#define SIZE_BK 8
#define SIZE_TM 8
#define SIZE_TN 8

#define SIZE_BM 128
#define SIZE_BN 128

#define TOTAL_RESULTS_BLOCKTILE (SIZE_BM * SIZE_BN)
#define NUM_THREADS_BLOCKTILE (TOTAL_RESULTS_BLOCKTILE / (SIZE_TM * SIZE_TN))
#define STRIDE_A (NUM_THREADS_BLOCKTILE / SIZE_BK)
#define STRIDE_B (NUM_THREADS_BLOCKTILE / SIZE_BN)

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// Kernels
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;

  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  META_LOOP(outer_loop, SIZE_K / BK, SIZE_K / BK, false);
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    META_LOOP(smem_load_loop, SIZE_BM / STRIDE_A, SIZE_BM / STRIDE_A, false);
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += STRIDE_A) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }

    META_LOOP(smem_load_loop, SIZE_BN / STRIDE_B, SIZE_BN / STRIDE_B, false);
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += STRIDE_B) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    META_LOOP(dot_loop, 8, 8, false);
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      META_LOOP(reg_load_loop, SIZE_TM, SIZE_TM, false);
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }

      META_LOOP(reg_load_loop, SIZE_TN, SIZE_TN, false);
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }

      META_LOOP(mac_loop, SIZE_TN, SIZE_TN, false);
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        META_LOOP(mac_loop_inner, SIZE_TN, SIZE_TN, false);
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}

// Matrix dimensions
const int M = SIZE_M; // Number of rows in A and C
const int N = SIZE_N; // Number of columns in B and C
const int K = SIZE_K; // Number of columns in A, rows in B

// Matrix multiplication parameters
const float alpha = 1.0f;
const float beta = 0.0f;

// Device pointers
float *d_A, *d_B, *d_C;

// Host matrices
std::vector<float> h_A(M *K);
std::vector<float> h_B(K *N);
std::vector<float> h_C(M *N);

// Block dimensions
dim3 threadsPerBlock(32, 32); // 32x32 = 1024 threads per block
dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

int main(int argc, char *argv[])
{
  // Initialize matrices with some test data
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < K; j++)
    {
      h_A[i * K + j] = 1.0f; // Initialize A with 1s
    }
  }

  for (int i = 0; i < K; i++)
  {
    for (int j = 0; j < N; j++)
    {
      h_B[i * N + j] = 1.0f; // Initialize B with 1s
    }
  }

  std::fill(h_C.begin(), h_C.end(), 0.0f); // Initialize C with zeros

  // Allocate device memory
  cudaCheck(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
  cudaCheck(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

  // Copy matrices from host to device
  cudaCheck(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

  // Launch CUDA workload with profiling replay loop
  // The profiler needs multiple passes to collect all metrics
  METRICS_KERNEL_START

  const uint BK = SIZE_BK;
  const uint TM = SIZE_TM;
  const uint TN = SIZE_TN;
  const uint BM = SIZE_BM;
  const uint BN = SIZE_BN;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));

  sgemm2DBlocktiling<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

  if (cudaPeekAtLastError() != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return -1;
  }

  METRICS_KERNEL_END

  // Copy result back to host
  cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Clean Up
  cudaCheck(cudaFree(d_A));
  cudaCheck(cudaFree(d_B));
  cudaCheck(cudaFree(d_C));

  return 0;
}
