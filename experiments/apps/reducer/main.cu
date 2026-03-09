#include "cupti_timing.h"
#include "ptx_meta.h"

#define ITERATIONS 40000000

#define BLOCK_SIZE 256

#define N_ELEMS (1024 * 256) // 262144 elements
#define NUM_BLOCKS (N_ELEMS / BLOCK_SIZE)

extern "C" {

__global__ void KERNEL_LAUNCH_BOUNDS(BLOCK_SIZE, 1)
    reduce_sum_kernel(const float *__restrict__ input,
                      float *__restrict__ output, int N) {
  // =======================================================================
  // PTX METADATA BLOCK
  //
  // Everything inside KERNEL_META(...) is compiled into PTX comments
  // (lines starting with "// @META:3 ...").  These comments have ZERO
  // runtime cost — the GPU ignores them entirely.  They exist so that
  // a downstream tool can parse the .ptx file and learn structured
  // facts about this kernel without running it.
  //
  // How it works under the hood:
  //   Each META_* macro expands to a C string literal like
  //     "// @META:3 PARAM 0 input ptr_f32 ...\n\t"
  //   All these strings are concatenated at compile time (standard C
  //   string literal joining) and wrapped in a single:
  //     asm volatile("// @META:3 BEGIN_KERNEL ...\n\t"
  //                  "// @META:3 PARAM ...\n\t"
  //                  ...
  //                  "// @META:3 END_KERNEL ...");
  //   The `asm volatile` tells the compiler:
  //     - "asm"      → emit this text into the assembly (PTX) output
  //     - "volatile" → don't optimize it away, even though it has no
  //                     side effects
  //   Because it's a comment ("//"), the GPU assembler and hardware
  //   skip it completely.  But grep/sed/python can read it.
  //
  // The ":3" in @META:3 is the protocol version — if we ever change the
  // tag format, we bump this number so parsers can handle both old and
  // new formats.
  // =======================================================================
  KERNEL_META(
      reduce_sum_kernel,

      // -----------------------------------------------------------------
      // PARAM — Describe each kernel parameter
      //
      // Format: META_PARAM_PTR(index, name, element_type, role, constraints)
      //         META_PARAM_INT(index, name, c_type, role, constraints)
      //
      //   index       — positional index in the kernel signature (0-based)
      //   name        — human-readable name (must match the actual param)
      //   element_type— for pointers: what the pointer points to (f32, f64,
      //   i32...) c_type      — for scalars: the C type (i32, i64, f32...) role
      //   — semantic meaning ("input_vector", "num_elements", etc.) constraints
      //   — space-separated tokens describing valid values:
      //       ALIGN(n)      — pointer is aligned to n bytes
      //       RANGE(lo,hi)  — integer value is always in [lo, hi]
      //       MULTIPLE(m)   — integer value is always a multiple of m
      //       READONLY      — kernel only reads through this pointer
      //       WRITEONLY     — kernel only writes through this pointer
      //       NOALIAS       — this pointer doesn't alias any other parameter
      //
      // PTX output example:
      //   // @META:3 PARAM 0 input ptr_f32 input_vector align=128 readonly
      //   noalias
      //
      // Use named constants from ptx_meta.h instead of raw strings:
      //   ELEM_F32, TYPE_I32          instead of  f32, i32
      //   ACCESS_READONLY, ACCESS_NOALIAS  instead of  READONLY, NOALIAS
      //   ALIGN(ALIGN_CACHELINE)      instead of  ALIGN(128)
      // -----------------------------------------------------------------
      META_PARAM_PTR(0, input, ELEM_F32, input_vector,
                     ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS)
          META_PARAM_PTR(1, output, ELEM_F32, partial_sums,
                         ALIGN(ALIGN_CACHELINE) ACCESS_WRITEONLY ACCESS_NOALIAS)
              META_PARAM_INT(2, N, TYPE_I32, num_elements,
                             RANGE(1, 1048576) MULTIPLE(256))

      // -----------------------------------------------------------------
      // TILE — Block/tile dimensions used by this kernel
      //
      // Format: META_TILE(dimension, size)
      //
      //   dimension — which axis: x, y, or z
      //   size      — number of threads along that axis
      //
      // This tells a parser "this kernel uses 256 threads in the x
      // dimension per block."  For 2D kernels (like matmul) you'd have
      // both META_TILE(x, 16) and META_TILE(y, 16).
      //
      // PTX output:
      //   // @META:3 TILE x 256
      // -----------------------------------------------------------------
      META_TILE(DIM_X, 256)

      // -----------------------------------------------------------------
      // LAUNCH — Expected launch configuration
      //
      // Format: META_LAUNCH(block_x, block_y, block_z, grid_expression)
      //
      //   block_x/y/z    — threads per block in each dimension
      //   grid_expression — human-readable string describing how the grid
      //                     size is computed from the input parameters
      //
      // This tells a parser "the host launches this kernel with 256×1×1
      // threads per block, and the grid has N/256 blocks in x."
      //
      // PTX output:
      //   // @META:3 LAUNCH 256 1 1 N/256 1 1
      // -----------------------------------------------------------------
      META_LAUNCH(256, 1, 1, "N/256 1 1")

      // -----------------------------------------------------------------
      // SHARED_MEM — Shared memory allocations inside the kernel
      //
      // Format: META_SHARED(variable_name, element_type, total_bytes)
      //
      //   variable_name — the name of the __shared__ array in the source
      //   element_type  — what it stores (f32, i32, etc.)
      //   total_bytes   — total size in bytes (256 floats × 4 bytes = 1024)
      //
      // Shared memory is fast on-chip SRAM shared by all threads in a
      // block.  A parser uses this to estimate occupancy (how many blocks
      // can run simultaneously on one SM).
      //
      // PTX output:
      //   // @META:3 SHARED_MEM sdata f32 1024
      // -----------------------------------------------------------------
      META_SHARED(sdata, ELEM_F32, 1024)

      // -----------------------------------------------------------------
      // LOOP — Trip count bounds for key loops
      //
      // Format: META_LOOP(label, min_iterations, max_iterations, is_unrolled)
      //
      //   label          — a human-readable name for the loop
      //   min_iterations — fewest times the loop body can execute
      //   max_iterations — most times the loop body can execute
      //   is_unrolled    — "true" if the compiler fully unrolls it,
      //                    "false" if it stays as a dynamic loop
      //
      // This is useful for performance modelling — a parser can estimate
      // instruction counts and memory traffic from loop bounds.
      //
      // PTX output:
      //   // @META:3 LOOP grid_stride 1 4096 false
      //   // @META:3 LOOP warp_reduce 5 5 true
      // -----------------------------------------------------------------
      META_LOOP(grid_stride, 1, 4096, false) META_LOOP(warp_reduce, 5, 5, true)

      // -----------------------------------------------------------------
      // LAYOUT — Memory layout of arrays
      //
      // Format: META_LAYOUT(array_name, order, dimensions)
      //
      //   array_name — which parameter this describes
      //   order      — memory ordering: "linear", "row_major", "col_major"
      //   dimensions — shape expression as a string
      //
      // "linear" means a flat 1D array.  For matrices you'd use
      // "row_major" with a shape like "MxN".
      //
      // PTX output:
      //   // @META:3 LAYOUT input linear N
      //   // @META:3 LAYOUT output linear num_blocks
      // -----------------------------------------------------------------
      META_LAYOUT(input, LAYOUT_LINEAR, "N")
          META_LAYOUT(output, LAYOUT_LINEAR, "num_blocks")

      // -----------------------------------------------------------------
      // ASSUME — Human-readable constraint descriptions
      //
      // Format: META_ASSUME("description string")
      //
      // These are free-form text describing invariants the kernel
      // relies on.  They document the "contract" between the host
      // launcher and the kernel — if any of these are violated,
      // the kernel may produce wrong results or crash.
      //
      // Unlike DEVICE_ASSUME() (which actually tells the compiler to
      // optimize based on the assumption), META_ASSUME is purely
      // informational — it only appears as a comment in PTX for the
      // parser to read.
      //
      // PTX output:
      //   // @META:3 ASSUME N > 0 && N <= 1048576 && N % 256 == 0
      //   // @META:3 ASSUME blockDim.x == 256
      // -----------------------------------------------------------------
      META_ASSUME("N > 0 && N <= 1048576 && N % 256 == 0")
          META_ASSUME("blockDim.x == 256")

      // -----------------------------------------------------------------
      // CONST_REF — Link to named constant fields in the PTX
      //
      // Format: META_CONST_REF(kernel_name, "field1 field2 ...")
      //
      //   kernel_name — matches the name used in KERNEL_CONST() above
      //   fields      — space-separated list of field names
      //
      // This tells the parser "look for PTX symbols matching
      // __ptxmeta_reduce_sum__<field> for each listed field."
      // Each symbol is a single int32 with a self-describing name:
      //   __ptxmeta_reduce_sum__version        = 3
      //   __ptxmeta_reduce_sum__block_size     = 256
      //   __ptxmeta_reduce_sum__warp_size      = 32
      //   __ptxmeta_reduce_sum__min_blocks_per_sm = 1
      //   __ptxmeta_reduce_sum__max_n          = 1048576
      //
      // PTX output:
      //   // @META:3 CONST __ptxmeta_reduce_sum version block_size warp_size
      //   min_blocks_per_sm max_n
      // -----------------------------------------------------------------
      META_CONST_REF(reduce_sum,
                     "version block_size warp_size min_blocks_per_sm max_n"));
  // ===== END METADATA ==================================================

  // ----- Compiler-side assumptions ----------------------------------------
  //
  // These are DIFFERENT from the META_ASSUME comments above.
  //
  // META_ASSUME("N > 0 ...") → a PTX comment for the parser to read.
  //                             The compiler ignores it.
  //
  // ASSUME_DIM(N, 1, ...) → actually tells the compiler "you may assume
  //                          N is in [1, 1048576] and is a multiple of 256."
  //                          The compiler uses this to:
  //                            - eliminate bounds checks it can prove
  //                            - use narrower integer arithmetic
  //                            - remove dead branches
  //                          But this does NOT appear in the PTX output.
  //
  // ASSUME_ALIGNED(input, 128) → tells the compiler the pointer is 128-byte
  //                               aligned, enabling wider (vectorized) loads.
  //
  // Under the hood these expand to:
  //   if (!(N >= 1)) __builtin_unreachable();
  //   if (!(N <= 1048576)) __builtin_unreachable();
  //   if (!(N % 256 == 0)) __builtin_unreachable();
  //
  // __builtin_unreachable() is a compiler intrinsic that says "this code
  // path can never be reached."  So `if (!expr) unreachable` means
  // "expr is always true" — the compiler can optimize based on that fact.
  // ---------------------------------------------------------------------
  ASSUME_DIM(N, 1, 1048576, 256);
  ASSUME_ALIGNED(input, 128);
  ASSUME_ALIGNED(output, 128);

  // ----- Shared memory -------------------------------------------------
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * BLOCK_SIZE + tid;
  int gridSize = BLOCK_SIZE * gridDim.x;

  ASSUME_RANGE(tid, 0, BLOCK_SIZE - 1);

  // ----- Phase 1: grid-stride accumulation into shared memory ----------
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
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (tid == 0) {
      output[blockIdx.x] = val;
    }
  }
}
}

int main() {
  int h[4] = {0};
  int *d;

  // Initialize CUPTI profiling
  initializeCUPTI();

  cudaMalloc(&d, 4 * sizeof(int));
  cudaMemcpy(d, h, 4 * sizeof(int), cudaMemcpyHostToDevice);

  printf("[LOG] Running kernel with %d iterations...\n", ITERATIONS);

  // Get CPU/GPU offsets
  collectTimestampOffsets();

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
  flushCUPTIBuffers();

  printKernelTiming();

  // Clean up
  cudaFree(d);

  disableCUPTI();
}
