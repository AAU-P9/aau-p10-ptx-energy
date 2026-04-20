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
#include "../../include/simple_cuda_utils.h"
#include "cupti_timing.h"

// NVML headers
#include <nvml.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#ifndef SIZE_M
#define SIZE_M 1024
#endif

#ifndef SIZE_N
#define SIZE_N 1024
#endif

#ifndef SIZE_K
#define SIZE_K 1024
#endif

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void KERNEL_LAUNCH_BOUNDS((BM * BN) / (TM * TN), 1)
sgemmVectorize(int M, int N, int K, float alpha, float *A,
               float *B, float beta, float *C) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

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
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  META_LOOP(outer_loop, SIZE_K / BK, SIZE_K / BK, false);
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    META_LOOP(dot_loop, SIZE_BK, SIZE_BK, false);
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      META_LOOP(reg_load_loop, SIZE_TM, SIZE_TM, false);
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      META_LOOP(reg_load_loop, SIZE_TN, SIZE_TN, false);
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }

      META_LOOP(mac_loop, SIZE_TM, SIZE_TM, false);
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
  META_LOOP(write_back_loop, SIZE_TM, SIZE_TM, false);
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    META_LOOP(write_back_loop_inner, SIZE_TN / 4, SIZE_TN / 4, false);
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }

  META_END_KERNEL(sgemmVectorize);
}

long M = SIZE_M;
long N = SIZE_N;
long K = SIZE_K;

float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

float *A = nullptr, *B = nullptr, *C = nullptr; // host matrices
float *dA = nullptr, *dB = nullptr, *dC = nullptr;

int main(int argc, char *argv[]) {
  const size_t sizeA = sizeof(float) * M * K;
  const size_t sizeB = sizeof(float) * K * N;
  const size_t sizeC = sizeof(float) * M * N;

  A = (float *)malloc(sizeA);
  B = (float *)malloc(sizeB);
  C = (float *)malloc(sizeC);

  randomize_matrix(A, M * K);
  randomize_matrix(B, K * N);
  randomize_matrix(C, M * N);

  cudaCheck(cudaMalloc((void **)&dA, sizeA));
  cudaCheck(cudaMalloc((void **)&dB, sizeB));
  cudaCheck(cudaMalloc((void **)&dC, sizeC));

  cudaCheck(cudaMemcpy(dA, A, sizeA,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeB,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeC,
                       cudaMemcpyHostToDevice));

  METRICS_KERNEL_START

  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;

  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  sgemmVectorize<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);


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
