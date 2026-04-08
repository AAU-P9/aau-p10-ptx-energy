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

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include "ptx_meta.h"
#include <vector>
#include "../../include/simple_cuda_utils.h"
#include "cupti_timing.h"

// NVML headers
#include <nvml.h>

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// Kernels
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {

  META_BEGIN_KERNEL(sgemm2DBlocktiling);
  META_PARAM_INT(0, M, TYPE_I32, rows_A, RANGE(1, 65536) MULTIPLE(128));
  META_PARAM_INT(1, N, TYPE_I32, cols_B, RANGE(1, 65536) MULTIPLE(128));
  META_PARAM_INT(2, K, TYPE_I32, inner_dim, RANGE(1, 65536) MULTIPLE(8));
  META_PARAM_FLOAT(3, alpha, TYPE_F32, alpha_scalar, "");
  META_PARAM_PTR(4, A, ELEM_F32, input_A, ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  META_PARAM_PTR(5, B, ELEM_F32, input_B, ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  META_PARAM_FLOAT(6, beta, TYPE_F32, beta_scalar, "");
  META_PARAM_PTR(7, C, ELEM_F32, output_C, ALIGN(ALIGN_CACHELINE) ACCESS_READWRITE ACCESS_NOALIAS);
  META_TILE(DIM_X, 256);
  META_LAUNCH(256, 1, 1, "CEIL_DIV(N,BN) CEIL_DIV(M,BM) 1");
  META_SHARED(As, ELEM_F32, BM *BK * 4);
  META_SHARED(Bs, ELEM_F32, BK *BN * 4);
  META_LAYOUT(A, LAYOUT_ROW_MAJOR, "M x K");
  META_LAYOUT(B, LAYOUT_ROW_MAJOR, "K x N");
  META_LAYOUT(C, LAYOUT_ROW_MAJOR, "M x N");
  META_ASSUME("BM == 128 && BN == 128 && BK == 8 && TM == 8 && TN == 8");
  META_ASSUME("blockDim.x == 256");

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
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  META_LOOP(outer_loop, 1, 8192, false);
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
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
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
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
const int M = 1024; // Number of rows in A and C
const int N = 1024; // Number of columns in B and C
const int K = 1024; // Number of columns in A, rows in B

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

void benchmark()
{
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
}

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
  initializeCUPTI();
  collectTimestampOffsets();
  benchmark();
  flushCUPTIBuffers();
  printKernelTiming();

  // Copy result back to host
  cudaCheck(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify result (C should be a matrix of K's since A and B are all 1's)
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      float expected = K * 1.0f * 1.0f; // Each element should be K (dot product of K 1's)
      if (fabs(h_C[i * N + j] - expected) > 1e-5)
      {
        printf("Verification failed at [%d,%d]: got %f, expected %f\n",
               i, j, h_C[i * N + j], expected);
        return 1;
      }
    }
  }
  printf("Matrix multiplication verification passed!\n");

  // Clean Up
  cudaCheck(cudaFree(d_A));
  cudaCheck(cudaFree(d_B));
  cudaCheck(cudaFree(d_C));

  return 0;
}
