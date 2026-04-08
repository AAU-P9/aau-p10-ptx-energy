#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "ptx_meta.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void KERNEL_LAUNCH_BOUNDS((BM * BN) / TM, 1)
sgemm1DBlocktiling(int M, int N, int K, float alpha,
                   const float *A, const float *B, float beta,
                   float *C) {

  META_BEGIN_KERNEL(sgemm1DBlocktiling);
  META_PARAM_INT(0, M, TYPE_I32, rows_A, RANGE(1, 65536) MULTIPLE(64));
  META_PARAM_INT(1, N, TYPE_I32, cols_B, RANGE(1, 65536) MULTIPLE(64));
  META_PARAM_INT(2, K, TYPE_I32, inner_dim, RANGE(1, 65536) MULTIPLE(8));
  META_PARAM_FLOAT(3, alpha, TYPE_F32, alpha_scalar, "");
  META_PARAM_PTR(4, A, ELEM_F32, input_A, ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  META_PARAM_PTR(5, B, ELEM_F32, input_B, ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  META_PARAM_FLOAT(6, beta, TYPE_F32, beta_scalar, "");
  META_PARAM_PTR(7, C, ELEM_F32, output_C, ALIGN(ALIGN_CACHELINE) ACCESS_READWRITE ACCESS_NOALIAS);
  META_TILE(DIM_X, 512);
  META_LAUNCH(512, 1, 1, "CEIL_DIV(N,BN) CEIL_DIV(M,BM) 1");
  META_SHARED(As, ELEM_F32, BM *BK * 4);
  META_SHARED(Bs, ELEM_F32, BK *BN * 4);
  META_LAYOUT(A, LAYOUT_ROW_MAJOR, "M x K");
  META_LAYOUT(B, LAYOUT_ROW_MAJOR, "K x N");
  META_LAYOUT(C, LAYOUT_ROW_MAJOR, "M x N");
  META_ASSUME("BM == 64 && BN == 64 && BK == 8 && TM == 8");
  META_ASSUME("blockDim.x == 512");

  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  META_LOOP(outer_loop, 1, 8192, false);
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
  META_LOOP(dot_loop, 8, 8, false);
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  META_LOOP(result_loop, 8, 8, false);
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }

  META_END_KERNEL(sgemm1DBlocktiling);
}