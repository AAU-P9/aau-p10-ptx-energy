#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include "ptx_meta.h"

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void KERNEL_LAUNCH_BOUNDS(1024, 1)
sgemm_naive(int M, int N, int K, float alpha, const float *A,
            const float *B, float beta, float *C) {

  META_BEGIN_KERNEL(sgemm_naive);
  META_PARAM_INT(0, M, TYPE_I32, rows_A, RANGE(1, 65536) MULTIPLE(32));
  META_PARAM_INT(1, N, TYPE_I32, cols_B, RANGE(1, 65536) MULTIPLE(32));
  META_PARAM_INT(2, K, TYPE_I32, inner_dim, RANGE(1, 65536));
  META_PARAM_FLOAT(3, alpha, TYPE_F32, alpha_scalar, "");
  META_PARAM_PTR(4, A, ELEM_F32, input_A, ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  META_PARAM_PTR(5, B, ELEM_F32, input_B, ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  META_PARAM_FLOAT(6, beta, TYPE_F32, beta_scalar, "");
  META_PARAM_PTR(7, C, ELEM_F32, output_C, ALIGN(ALIGN_CACHELINE) ACCESS_READWRITE ACCESS_NOALIAS);
  META_TILE(DIM_X, 32);
  META_TILE(DIM_Y, 32);
  META_LAUNCH(32, 32, 1, "CEIL_DIV(M,32) CEIL_DIV(N,32) 1");
  META_LAYOUT(A, LAYOUT_ROW_MAJOR, "M x K");
  META_LAYOUT(B, LAYOUT_ROW_MAJOR, "K x N");
  META_LAYOUT(C, LAYOUT_ROW_MAJOR, "M x N");
  META_ASSUME("blockDim.x == 32 && blockDim.y == 32");

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    META_LOOP(k_loop, 1, 65536, false);
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }

  META_END_KERNEL(sgemm_naive);
}