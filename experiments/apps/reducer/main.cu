#include "ptx_meta.h"
#include <cuda.h>
#include <stdio.h>


#define ITERATIONS 40000000

#define BLOCK_SIZE 256

#define N_ELEMS (1024 * 256) // 262144 elements
#define NUM_BLOCKS (N_ELEMS / BLOCK_SIZE)

extern "C" {

__global__ void KERNEL_LAUNCH_BOUNDS(BLOCK_SIZE, 1)
    reduce_sum_kernel(const float *__restrict__ input,
                      float *__restrict__ output, int N) {
  KERNEL_BEGIN(reduce_sum_kernel);

  KERNEL_PARAM_PTR(0, input, ELEM_F32, input_vector,
                   ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS);
  KERNEL_PARAM_PTR(1, output, ELEM_F32, partial_sums,
                   ALIGN(ALIGN_CACHELINE) ACCESS_WRITEONLY ACCESS_NOALIAS);
  KERNEL_PARAM_INT(2, N, TYPE_I32, num_elements,
                   RANGE(1, 1048576) MULTIPLE(256));

  KERNEL_TILE(DIM_X, 256);
  KERNEL_LAUNCH_CONFIG(256, 1, 1, "N/256 1 1");
  KERNEL_LAYOUT(input, LAYOUT_LINEAR, "N");
  KERNEL_LAYOUT(output, LAYOUT_LINEAR, "num_blocks");

  KERNEL_ASSUME("N > 0 && N <= 1048576 && N % 256 == 0");
  KERNEL_ASSUME("blockDim.x == 256");
  KERNEL_CONST_REF(reduce_sum,
                   "version block_size warp_size min_blocks_per_sm max_n");

  ASSUME_DIM(N, 1, 1048576, 256);
  ASSUME_ALIGNED(input, 128);
  ASSUME_ALIGNED(output, 128);

  // ----- Shared memory -------------------------------------------------
  KERNEL_SHARED(sdata, ELEM_F32, 1024);
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * BLOCK_SIZE + tid;
  int gridSize = BLOCK_SIZE * gridDim.x;

  ASSUME_RANGE(tid, 0, BLOCK_SIZE - 1);

  // ----- Phase 1: grid-stride accumulation into shared memory ----------
  KERNEL_LOOP(grid_stride, 1, 4096, false);
  float sum = 0.0f;
  for (int i = gid; i < N; i += gridSize) {
    sum += input[i];
  }
  sdata[tid] = sum;
  __syncthreads();

  // ----- Phase 2: tree reduction in shared memory ----------------------
  // Each step halves the active threads.  We unroll the last warp.
  if (BLOCK_SIZE >= 256 && tid < 128) {
    sdata[tid] += sdata[tid + 128];
  }
  __syncthreads();
  if (BLOCK_SIZE >= 128 && tid < 64) {
    sdata[tid] += sdata[tid + 64];
  }
  __syncthreads();

  // ----- Phase 3: warp-level shuffle reduction (last 32 threads) -------
  if (tid < 32) {
    // Reduce 64 -> 32 (still need volatile or explicit load since tid+32 is in
    // same warp)
    sdata[tid] += sdata[tid + 32];
    __syncwarp();
    float val = sdata[tid];
    KERNEL_LOOP(warp_reduce, 5, 5, true);
    for (int offset = 16; offset >= 1; offset >>= 1) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (tid == 0) {
      output[blockIdx.x] = val;
    }
  }

  KERNEL_END(reduce_sum_kernel);
}
}

int main() {
  int h[4] = {0};
  int *d;

  // Initialize CUPTI profiling
  // initializeCUPTI();

  cudaMalloc(&d, 4 * sizeof(int));
  cudaMemcpy(d, h, 4 * sizeof(int), cudaMemcpyHostToDevice);

  printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

  // Get CPU/GPU offsets
  // collectTimestampOffsets();

  dim3 threads(BLOCK_SIZE);
  dim3 blocks(NUM_BLOCKS);

  const size_t inputBytes = N_ELEMS * sizeof(float);
  const size_t partialBytes = NUM_BLOCKS * sizeof(float);

  // Host allocations
  float *h_input = (float *)malloc(inputBytes);
  float *h_partial = (float *)malloc(partialBytes);

  // Fill with 1.0 so expected sum = N_ELEMS
  for (int i = 0; i < N_ELEMS; i++)
    h_input[i] = 1.0f;

  // Device allocations
  float *d_input, *d_partial;
  cudaMalloc(&d_input, inputBytes);
  cudaMalloc(&d_partial, partialBytes);
  cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);

  // Run kernel
  reduce_sum_kernel<<<blocks, threads>>>(d_input, d_partial, N_ELEMS);

  cudaDeviceSynchronize();

  // Possibly read back results (not necessary for timing, but included for
  // completeness) cudaMemcpy(h, d, 4*sizeof(int), cudaMemcpyDeviceToHost);

  // Flush all activity buffers
  // flushCUPTIBuffers();

  // printKernelTiming();

  // Clean up
  cudaFree(d);

  // disableCUPTI();
}
