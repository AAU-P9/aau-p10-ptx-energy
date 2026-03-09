/**
 * reduce.cu — Parallel sum-reduction kernel annotated with ptx_meta.h
 *
 * Demonstrates the typed PTX metadata framework on a non-matmul kernel.
 * All annotations appear as structured // @META:3 comments in the PTX.
 *
 * Kernel: block-level sum reduction with warp-shuffle final stage.
 *   input:  float[N]
 *   output: float[num_blocks]   (one partial sum per block)
 *
 * A second pass (or atomicAdd) would finish the full reduction;
 * this example focuses on the annotation pattern.
 */

#include "ptx_meta.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// -----------------------------------------------------------------------
// Compile-time constants
// -----------------------------------------------------------------------
#define BLOCK_SIZE 256
#define WARP_SIZE  WARP_SIZE_CONST   // from ptx_meta.h (always 32)

// -----------------------------------------------------------------------
// Constant metadata — named fields in PTX
//
// Each KERNEL_CONST() creates a separate __device__ __constant__ variable
// whose name encodes what it represents:
//
//   KERNEL_CONST(reduce_sum, block_size, 256)
//   -> PTX symbol: __ptxmeta_reduce_sum__block_size = 256
//
// A parser finds all fields for this kernel by matching:
//   __ptxmeta_reduce_sum__*
//
// No positional indices to remember — the field name IS the documentation.
// -----------------------------------------------------------------------
KERNEL_CONST(reduce_sum, version,           PTX_META_VERSION);
KERNEL_CONST(reduce_sum, block_size,        BLOCK_SIZE);
KERNEL_CONST(reduce_sum, warp_size,         WARP_SIZE);
KERNEL_CONST(reduce_sum, min_blocks_per_sm, 1);
KERNEL_CONST(reduce_sum, max_n,             1048576);

// -----------------------------------------------------------------------
// Kernel function declaration
//
// KERNEL_LAUNCH_BOUNDS(max_threads, min_blocks) expands to:
//   __launch_bounds__(256, 1)
//
// This is one of the FEW annotations that survives into PTX as a real
// directive (not just a comment):
//   .maxntid 256, 1, 1        — max threads per block
//   .minnctapersm 1           — min blocks per SM (occupancy hint)
//
// The GPU's thread scheduler uses these to allocate registers.  Fewer
// registers per thread → more blocks can run concurrently → higher
// occupancy.  Without this, the compiler picks its own register budget
// which may limit occupancy.
//
// __restrict__ on the pointer parameters tells the compiler "these
// pointers don't alias each other" (they don't point to overlapping
// memory).  In PTX this enables ld.global.nc (non-coherent loads)
// which bypass the L1 cache and go through the texture cache — often
// faster for read-only data.
// -----------------------------------------------------------------------
__global__ void KERNEL_LAUNCH_BOUNDS(BLOCK_SIZE, 1)
reduce_sum_kernel(const float* __restrict__ input,
                  float*       __restrict__ output,
                  int N)
{
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
    KERNEL_META(reduce_sum_kernel,

        // -----------------------------------------------------------------
        // PARAM — Describe each kernel parameter
        //
        // Format: META_PARAM_PTR(index, name, element_type, role, constraints)
        //         META_PARAM_INT(index, name, c_type, role, constraints)
        //
        //   index       — positional index in the kernel signature (0-based)
        //   name        — human-readable name (must match the actual param)
        //   element_type— for pointers: what the pointer points to (f32, f64, i32...)
        //   c_type      — for scalars: the C type (i32, i64, f32...)
        //   role        — semantic meaning ("input_vector", "num_elements", etc.)
        //   constraints — space-separated tokens describing valid values:
        //       ALIGN(n)      — pointer is aligned to n bytes
        //       RANGE(lo,hi)  — integer value is always in [lo, hi]
        //       MULTIPLE(m)   — integer value is always a multiple of m
        //       READONLY      — kernel only reads through this pointer
        //       WRITEONLY     — kernel only writes through this pointer
        //       NOALIAS       — this pointer doesn't alias any other parameter
        //
        // PTX output example:
        //   // @META:3 PARAM 0 input ptr_f32 input_vector align=128 readonly noalias
        //
        // Use named constants from ptx_meta.h instead of raw strings:
        //   ELEM_F32, TYPE_I32          instead of  f32, i32
        //   ACCESS_READONLY, ACCESS_NOALIAS  instead of  READONLY, NOALIAS
        //   ALIGN(ALIGN_CACHELINE)      instead of  ALIGN(128)
        // -----------------------------------------------------------------
        META_PARAM_PTR(0, input,  ELEM_F32, input_vector,  ALIGN(ALIGN_CACHELINE) ACCESS_READONLY ACCESS_NOALIAS)
        META_PARAM_PTR(1, output, ELEM_F32, partial_sums,  ALIGN(ALIGN_CACHELINE) ACCESS_WRITEONLY ACCESS_NOALIAS)
        META_PARAM_INT(2, N,      TYPE_I32, num_elements,  RANGE(1, 1048576) MULTIPLE(256))

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
        META_LOOP(grid_stride, 1, 4096, false)
        META_LOOP(warp_reduce, 5, 5, true)

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
        //   // @META:3 CONST __ptxmeta_reduce_sum version block_size warp_size min_blocks_per_sm max_n
        // -----------------------------------------------------------------
        META_CONST_REF(reduce_sum, "version block_size warp_size min_blocks_per_sm max_n")
    );
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
    if (BLOCK_SIZE >= 256 && tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    if (BLOCK_SIZE >= 128 && tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads();

    // ----- Phase 3: warp-level shuffle reduction (last 32 threads) -------
    if (tid < 32) {
        // Reduce 64 -> 32 (still need volatile or explicit load since tid+32 is in same warp)
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

// -----------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------
void reduce_sum(const float* d_input, float* d_output, float* d_partial,
                int N, int num_blocks)
{
    DEVICE_ASSUME(N > 0 && N <= 1048576 && N % 256 == 0);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(num_blocks);

    reduce_sum_kernel<<<blocks, threads>>>(d_input, d_partial, N);
    // A second kernel or host-side sum of d_partial[0..num_blocks-1]
    // would complete the reduction.  Kept simple here.
}

// -----------------------------------------------------------------------
// main — correctness test
// -----------------------------------------------------------------------
int main()
{
    const int N = 1024 * 256;   // 262144 elements
    const int num_blocks = N / BLOCK_SIZE;
    const size_t inputBytes   = N * sizeof(float);
    const size_t partialBytes = num_blocks * sizeof(float);

    // Host allocations
    float* h_input   = (float*)malloc(inputBytes);
    float* h_partial = (float*)malloc(partialBytes);

    // Fill with 1.0 so expected sum = N
    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    // Device allocations
    float *d_input, *d_partial;
    cudaMalloc(&d_input,   inputBytes);
    cudaMalloc(&d_partial, partialBytes);
    cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);

    // Launch
    reduce_sum(d_input, nullptr, d_partial, N, num_blocks);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy partial sums back
    cudaMemcpy(h_partial, d_partial, partialBytes, cudaMemcpyDeviceToHost);

    // Sum partial results on host
    double total = 0.0;
    for (int i = 0; i < num_blocks; i++)
        total += h_partial[i];

    // Verify
    double expected = (double)N;
    if (fabs(total - expected) < 1.0) {
        printf("PASS — reduce_sum: N=%d, sum=%.0f (expected %.0f)\n",
               N, total, expected);
    } else {
        printf("FAIL — reduce_sum: N=%d, sum=%.1f (expected %.0f)\n",
               N, total, expected);
    }

    cudaFree(d_input);
    cudaFree(d_partial);
    free(h_input);
    free(h_partial);
    return (fabs(total - expected) < 1.0) ? 0 : 1;
}
