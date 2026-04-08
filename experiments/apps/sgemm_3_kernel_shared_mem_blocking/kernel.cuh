#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "ptx_meta.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void KERNEL_LAUNCH_BOUNDS(BLOCKSIZE * BLOCKSIZE, 1)
sgemm_shared_mem_block(int M, int N, int K, float alpha,
                       const float *A, const float *B,
                       float beta, float *C) {

  META_BEGIN_KERNEL(sgemm_shared_mem_block);
  META_PARAM_INT(0, M, TYPE_I32, rows_A, RANGE(1, 65536) MULTIPLE(32));
  META_PARAM_INT(1, N, TYPE_I32, cols_B, RANGE(1, 65536) MULTIPLE(32));
  META_PARAM_INT(2, K, TYPE_I32, inner_dim, RANGE(1, 65536) MULTIPLE(32));
  META_PARAM_FLOAT(3, alpha, TYPE_F32, alpha_scalar, "");
  META_PARAM_PTR(4, A, ELEM_F32, input_A, ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  META_PARAM_PTR(5, B, ELEM_F32, input_B, ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  META_PARAM_FLOAT(6, beta, TYPE_F32, beta_scalar, "");
  META_PARAM_PTR(7, C, ELEM_F32, output_C, ALIGN(ALIGN_CACHELINE) ACCESS_READWRITE ACCESS_NOALIAS);
  META_TILE(DIM_X, 1024);
  META_LAUNCH(1024, 1, 1, "CEIL_DIV(M,BLOCKSIZE) CEIL_DIV(N,BLOCKSIZE) 1");
  META_SHARED(As, ELEM_F32, BLOCKSIZE *BLOCKSIZE * 4);
  META_SHARED(Bs, ELEM_F32, BLOCKSIZE *BLOCKSIZE * 4);
  META_LAYOUT(A, LAYOUT_ROW_MAJOR, "M x K");
  META_LAYOUT(B, LAYOUT_ROW_MAJOR, "K x N");
  META_LAYOUT(C, LAYOUT_ROW_MAJOR, "M x N");
  META_ASSUME("BLOCKSIZE == 32 && blockDim.x == 1024");
  META_ASSUME("K % BLOCKSIZE == 0");

  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  META_LOOP(outer_loop, 1, 2048, false);
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    META_LOOP(dot_loop, 32, 32, false);
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];

  META_END_KERNEL(sgemm_shared_mem_block);
}