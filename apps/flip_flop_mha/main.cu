#include "cupti_timing.h"
#include "ptx_meta.h"
#include <cuda.h>
#include <cub/block/block_reduce.cuh>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Multi-head attention kernel (flip-flop / decode step).
// One block per (candidate, head). Threads compute Q·K^T dot products,
// softmax over n_steps, then weighted sum of V. Shared memory holds the
// query slice and the softmax logits.

#ifndef _GRID_DIM
// beam_size * nhead blocks total — overridden at build time if needed
#define _GRID_DIM (BEAM_SIZE * NHEAD)
#endif

#ifndef _BLOCK_DIM
#define _BLOCK_DIM 256
#endif

#ifndef BEAM_SIZE
#define BEAM_SIZE 4
#endif

#ifndef NHEAD
#define NHEAD 8
#endif

#ifndef DIM_PER_HEAD
#define DIM_PER_HEAD 64
#endif

// n_steps must be <= _BLOCK_DIM (256). Larger sequences need loop tiling.
#ifndef N_STEPS
#define N_STEPS 128
#endif

#ifndef THRESHOLD
#define THRESHOLD 20
#endif

#ifndef ITERATIONS
#define ITERATIONS 100000
#endif

#define QK_COL (NHEAD * DIM_PER_HEAD)
#define V_COL  QK_COL
#define SMEM_BYTES ((DIM_PER_HEAD + N_STEPS) * sizeof(float))

#define FINAL_MASK 0xffffffff

__inline__ __device__
float warpReduceSum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

template <int BLOCK_SIZE>
__inline__ __device__
float blockReduceSum(float val, typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage &temp_storage) {
    return cub::BlockReduce<float, BLOCK_SIZE>(temp_storage).Sum(val);
}

__inline__ __device__
float warpReduceMax(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

struct FloatMax {
    __device__ float operator()(float a, float b) const { return a > b ? a : b; }
};

template <int BLOCK_SIZE>
__inline__ __device__
float blockReduceMax(float val, typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage &temp_storage) {
    return cub::BlockReduce<float, BLOCK_SIZE>(temp_storage).Reduce(val, FloatMax{});
}

extern "C" __global__
KERNEL_LAUNCH_BOUNDS(_BLOCK_DIM, 2)
void mha(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    const int beam_size,
    const int n_steps,
    const int qk_col,
    const int v_col,
    const int nhead,
    const float scale,
    const int threshold,
    const int iterations,
    float *__restrict__ dst)
{
    KERNEL_META(mha,
        _META_FRAG_PARAM_PTR(0,  q,         f32, input,  READONLY NOALIAS ALIGN(128))
        _META_FRAG_PARAM_PTR(1,  k,         f32, input,  READONLY NOALIAS ALIGN(128))
        _META_FRAG_PARAM_PTR(2,  v,         f32, input,  READONLY NOALIAS ALIGN(128))
        _META_FRAG_PARAM_INT(3,  beam_size, i32, dim,    RANGE(1, 64))
        _META_FRAG_PARAM_INT(4,  n_steps,   i32, dim,    RANGE(1, 256))
        _META_FRAG_PARAM_INT(5,  qk_col,    i32, dim,    RANGE(1, 4096))
        _META_FRAG_PARAM_INT(6,  v_col,     i32, dim,    RANGE(1, 4096))
        _META_FRAG_PARAM_INT(7,  nhead,     i32, dim,    RANGE(1, 64))
        _META_FRAG_PARAM_FLOAT(8, scale,    f32, scalar, "")
        _META_FRAG_PARAM_INT(9,  threshold,  i32, scalar, RANGE(1, 100))
        _META_FRAG_PARAM_INT(10, iterations, i32, scalar, RANGE(1, 10000000))
        _META_FRAG_PARAM_PTR(11, dst,        f32, output, WRITEONLY NOALIAS ALIGN(128))
        _META_FRAG_LAUNCH(_BLOCK_DIM, 1, 1, "beam_size*nhead 1 1")
        _META_FRAG_SHARED(sq,     f32, DIM_PER_HEAD)
        _META_FRAG_SHARED(logits, f32, N_STEPS)
        _META_FRAG_LOOP(iter_loop,    1, ITERATIONS,   false)
        _META_FRAG_LOOP(dot_loop,     1, DIM_PER_HEAD, false)
        _META_FRAG_LOOP(weighted_sum, 1, N_STEPS,      false)
        _META_FRAG_ASSUME("n_steps <= blockDim.x")
        _META_FRAG_CUSTOM("algorithm", "multi_head_attention_decode")
    );

    const int BLOCK_SIZE = 256;
    int dim_per_head = qk_col / nhead;

    int candidate_id = blockIdx.x / nhead;
    int head_id      = blockIdx.x % nhead;

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // sq: query slice for this head; logits: softmax weights over time steps
    extern __shared__ float shared_buffer[];
    float *sq     = shared_buffer;
    float *logits = shared_buffer + dim_per_head;

    META_LOOP(iter_loop, 1, ITERATIONS, false);
    for (int iter = 0; iter < iterations; iter++) {

    // Load query vector for this candidate/head into shared memory
    int q_index = candidate_id * qk_col + head_id * dim_per_head + threadIdx.x;
    if (threadIdx.x < dim_per_head)
        sq[threadIdx.x] = q[q_index];
    __syncthreads();

    // Each thread computes the dot-product for one time step
    float dot_val = 0.f;
    if (threadIdx.x < n_steps) {
        const float *k_ptr = k + candidate_id * (qk_col * n_steps)
                               + threadIdx.x * qk_col
                               + head_id * dim_per_head;
        META_LOOP(dot_loop, 1, DIM_PER_HEAD, false);
        for (int i = 0; i < dim_per_head; i++) {
            dot_val += sq[i] * k_ptr[i];
        }
        dot_val *= scale;
    }

    // Numerically-stable softmax: subtract max, clip at -threshold, exponentiate
    __shared__ float s_max_val;
    __shared__ float s_sum;

    float local_val = (threadIdx.x < n_steps) ? dot_val : -1e20f;
    float max_val   = blockReduceMax<BLOCK_SIZE>(local_val, temp_storage);
    if (threadIdx.x == 0)
        s_max_val = max_val;
    __syncthreads();

    local_val -= s_max_val;
    if (local_val < -threshold)
        local_val = -threshold;
    float exp_val = expf(local_val);

    float sum_exp = blockReduceSum<BLOCK_SIZE>((threadIdx.x < n_steps) ? exp_val : 0.f, temp_storage);
    if (threadIdx.x == 0)
        s_sum = sum_exp;
    __syncthreads();

    if (threadIdx.x < n_steps)
        logits[threadIdx.x] = exp_val / s_sum;
    __syncthreads();

    // Weighted sum over value matrix: each thread owns one output element
    if (threadIdx.x < dim_per_head) {
        float weighted_sum = 0.f;
        int v_index = candidate_id * (v_col * n_steps) + head_id * dim_per_head + threadIdx.x;
        META_LOOP(weighted_sum, 1, N_STEPS, false);
        for (int t = 0; t < n_steps; t++)
            weighted_sum += logits[t] * v[v_index + t * v_col];
        dst[candidate_id * v_col + head_id * dim_per_head + threadIdx.x] = weighted_sum;
    }

    } // for iter
}

int main()
{
    METRICS_KERNEL_START

    const int beam_size    = BEAM_SIZE;
    const int nhead        = NHEAD;
    const int dim_per_head = DIM_PER_HEAD;
    const int n_steps      = N_STEPS;
    const int qk_col       = QK_COL;
    const int v_col        = V_COL;
    const float scale      = 1.0f / sqrtf((float)dim_per_head);
    const int threshold    = THRESHOLD;

    size_t q_elems = (size_t)beam_size * qk_col;
    size_t k_elems = (size_t)beam_size * n_steps * qk_col;
    size_t v_elems = (size_t)beam_size * n_steps * v_col;
    size_t d_elems = (size_t)beam_size * v_col;

    float *h_q   = (float *)malloc(q_elems * sizeof(float));
    float *h_k   = (float *)malloc(k_elems * sizeof(float));
    float *h_v   = (float *)malloc(v_elems * sizeof(float));

    for (size_t i = 0; i < q_elems; i++) h_q[i] = 0.1f + (float)(i % 17) * 0.01f;
    for (size_t i = 0; i < k_elems; i++) h_k[i] = 0.1f + (float)(i % 13) * 0.01f;
    for (size_t i = 0; i < v_elems; i++) h_v[i] = 0.1f + (float)(i % 11) * 0.01f;

    float *d_q, *d_k, *d_v, *d_dst;
    cudaMalloc(&d_q,   q_elems * sizeof(float));
    cudaMalloc(&d_k,   k_elems * sizeof(float));
    cudaMalloc(&d_v,   v_elems * sizeof(float));
    cudaMalloc(&d_dst, d_elems * sizeof(float));

    cudaMemcpy(d_q, h_q, q_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, k_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, v_elems * sizeof(float), cudaMemcpyHostToDevice);

    const int iterations = ITERATIONS;

    printf("[LOG] flip_flop_mha: beam=%d nhead=%d dim_per_head=%d n_steps=%d iterations=%d\n",
           beam_size, nhead, dim_per_head, n_steps, iterations);
    printf("[LOG] Launching with gridDim=%d blockDim=%d smem=%zu bytes\n",
           beam_size * nhead, _BLOCK_DIM, SMEM_BYTES);

    mha<<<beam_size * nhead, _BLOCK_DIM, SMEM_BYTES>>>(
        d_q, d_k, d_v,
        beam_size, n_steps, qk_col, v_col, nhead,
        scale, threshold, iterations,
        d_dst);
    cudaDeviceSynchronize();

    EXPORT_N("gridDim_x",  beam_size * nhead);
    EXPORT_N("gridDim_y",  1);
    EXPORT_N("gridDim_z",  1);
    EXPORT_N("blockDim_x", _BLOCK_DIM);
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);
    EXPORT_N("beam_size",  beam_size);
    EXPORT_N("nhead",      nhead);
    EXPORT_N("dim_per_head", dim_per_head);
    EXPORT_N("n_steps",    n_steps);
    EXPORT_N("iterations", iterations);
    EXPORT_N("smem_bytes", (int)SMEM_BYTES);

    METRICS_KERNEL_END

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_dst);
    free(h_q);
    free(h_k);
    free(h_v);
    disableCUPTI();
}
