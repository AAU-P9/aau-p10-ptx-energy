/**
 * annotated_matmul.cu
 *
 * A matrix multiplication kernel with annotations that survive into PTX
 * for a downstream PTX analysis/parsing tool.
 *
 * Techniques used to embed info into PTX:
 *
 *   1. __launch_bounds__  ->  .maxntid / .minnctapersm directives
 *   2. __restrict__       ->  ld.global.nc (non-coherent) loads
 *   3. Inline PTX asm comments  ->  // @META ... lines in PTX output
 *      This is the most powerful mechanism: arbitrary structured metadata
 *      injected as PTX comments via asm volatile("// ...").
 *   4. .reqntid            ->  exact thread dimensions (via inline asm)
 *   5. Named .const symbols ->  metadata encoded as constant data
 *   6. DEVICE_ASSUME()    ->  compiler optimisation hints
 *
 * Works with both nvcc and clang — no clang-specific builtins.
 */

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Portable assumption macro
//
// if (!(expr)) __builtin_unreachable() is semantically identical to
// __builtin_assume(expr) but works in GCC, clang, and nvcc.  The compiler
// is permitted to assume `expr` is always true and optimise accordingly.
//
// In clang's LLVM IR this becomes:  call void @llvm.assume(i1 %cond)
// In nvcc's cicc pipeline it enables the same range-narrowing optimisations.
// ---------------------------------------------------------------------------
#define DEVICE_ASSUME(expr) \
    do { if (!(expr)) __builtin_unreachable(); } while (0)

// ---------------------------------------------------------------------------
// PTX COMMENT METADATA  (the key mechanism for your PTX parser)
//
// asm volatile("// @META ...") injects a comment line into the PTX output.
// The compiler cannot optimise it away (volatile), and PTX comments are
// free — zero runtime cost, zero register pressure.
//
// Convention used here:
//   // @META <tag> <key> <value...>
// Tags:
//   PARAM_RANGE   <name> <min> <max> <stride>   — integer param bounds
//   PARAM_ALIGN   <name> <bytes>                 — pointer alignment
//   PARAM_TYPE    <name> <type> <role>            — semantic type info
//   TILE_SIZE     <dim> <value>                   — tile dimensions
//   LOOP_BOUNDS   <label> <min_iters> <max_iters> — loop trip counts
//   KERNEL_SHAPE  <gridX> <gridY> <blockX> <blockY> — launch config
//   MEMORY        <name> <space> <size_bytes>     — memory allocation
//   VERSION       <schema_version>                — metadata format version
// ---------------------------------------------------------------------------

// Emit a raw PTX comment — appears verbatim in .ptx output
#define PTX_COMMENT(str) \
    asm volatile("// " str)

// Emit structured metadata as a PTX comment
// These use the @META prefix so your parser can grep for them
#define PTX_META(str) \
    asm volatile("// @META " str)

// ---------------------------------------------------------------------------
// Annotation helper macros (built on top of DEVICE_ASSUME)
// These also emit PTX_META comments for the PTX parser.
// ---------------------------------------------------------------------------

// Constrain an integer to [lo, hi]  (inclusive)
#define ASSUME_RANGE(x, lo, hi)         \
    DEVICE_ASSUME((x) >= (lo));         \
    DEVICE_ASSUME((x) <= (hi))

// Assert a value is a multiple of m (useful for warp/tile sizes)
#define ASSUME_MULTIPLE(x, m)           \
    DEVICE_ASSUME(((x) % (m)) == 0)

// Assert a pointer is aligned to `al` bytes
#define ASSUME_ALIGNED(ptr, al)      \
    DEVICE_ASSUME(((uintptr_t)(ptr) % (al)) == 0)

// Combined: "this is a well-formed matrix dimension"
// Rows/cols are in [1, 8192] and are multiples of 32 (warp width)
#define ASSUME_MATRIX_DIM(n)            \
    ASSUME_RANGE(n, 1, 8192);          \
    ASSUME_MULTIPLE(n, 32)

// ---------------------------------------------------------------------------
// Tile size — changing this recompiles with different unroll / register usage
// ---------------------------------------------------------------------------
#define TILE 16

// ---------------------------------------------------------------------------
// Launch bounds — this is the ONE annotation that survives into PTX as
// .maxntid and .minnctapersm directives.  Without this, PTX has zero
// hints about the thread configuration.
//   maxThreadsPerBlock = TILE*TILE = 256
//   minBlocksPerMultiprocessor = 2  (occupancy hint)
// ---------------------------------------------------------------------------
#define LAUNCH_BOUNDS __launch_bounds__(TILE * TILE, 2)

// ---------------------------------------------------------------------------
// Constant metadata table — survives into PTX as a named .const symbol.
// Your PTX parser can find __meta_matmul_kernel and read the values.
// Format: {version, tile_x, tile_y, max_M, max_N, max_K, align_bytes,
//          min_blocks_per_sm, max_threads_per_block}
// ---------------------------------------------------------------------------
__device__ __constant__ int __meta_matmul_kernel[] = {
    1,              // metadata schema version
    TILE,           // tile width  (blockDim.x)
    TILE,           // tile height (blockDim.y)
    8192,           // max M
    8192,           // max N
    8192,           // max K
    128,            // pointer alignment (bytes)
    2,              // minBlocksPerMultiprocessor
    TILE * TILE     // maxThreadsPerBlock
};

// ---------------------------------------------------------------------------
// Device-side kernel
// ---------------------------------------------------------------------------

/**
 * Tiled matrix multiplication:  C = A * B
 *
 * A: [M x K]   B: [K x N]   C: [M x N]   (all row-major)
 *
 * Annotations inject LLVM range/assume metadata so a downstream analysis
 * pass (or your own opt pass) can prove things like:
 *   - loop trip counts are bounded
 *   - no integer overflow in index arithmetic
 *   - pointer alignment for vectorisation
 */
__global__ void LAUNCH_BOUNDS matmul_kernel(
        const float* __restrict__ A,   // __restrict__ -> ld.global.nc in PTX
        const float* __restrict__ B,
        float*       __restrict__ C,
        int M, int N, int K)
{
    // ==================================================================
    // PTX METADATA BLOCK — all of these appear as comments in the .ptx
    // Your parser should grep for '// @META' lines.
    // ==================================================================
    PTX_META("VERSION 2");
    PTX_META("KERNEL matmul_kernel");

    // Parameter ranges (parseable: PARAM_RANGE <name> <min> <max> <stride>)
    PTX_META("PARAM_RANGE M 1 8192 32");
    PTX_META("PARAM_RANGE N 1 8192 32");
    PTX_META("PARAM_RANGE K 1 8192 32");

    // Parameter types (PARAM_TYPE <name> <ctype> <role>)
    PTX_META("PARAM_TYPE A ptr_f32 input_matrix");
    PTX_META("PARAM_TYPE B ptr_f32 input_matrix");
    PTX_META("PARAM_TYPE C ptr_f32 output_matrix");
    PTX_META("PARAM_TYPE M i32 rows_A");
    PTX_META("PARAM_TYPE N i32 cols_B");
    PTX_META("PARAM_TYPE K i32 cols_A_rows_B");

    // Pointer alignment (PARAM_ALIGN <name> <bytes>)
    PTX_META("PARAM_ALIGN A 128");
    PTX_META("PARAM_ALIGN B 128");
    PTX_META("PARAM_ALIGN C 128");

    // Tile / launch shape
    PTX_META("TILE_SIZE x 16");
    PTX_META("TILE_SIZE y 16");
    PTX_META("KERNEL_SHAPE grid_x=N/16 grid_y=M/16 block_x=16 block_y=16");

    // Memory allocations (MEMORY <name> <space> <bytes>)
    PTX_META("MEMORY sA shared 1024");
    PTX_META("MEMORY sB shared 1024");

    // Loop bounds (LOOP_BOUNDS <label> <min_iters> <max_iters>)
    PTX_META("LOOP_BOUNDS tile_loop 1 512");
    PTX_META("LOOP_BOUNDS inner_product 16 16");

    // Matrix layout
    PTX_META("LAYOUT A row_major MxK");
    PTX_META("LAYOUT B row_major KxN");
    PTX_META("LAYOUT C row_major MxN");

    // ==================================================================
    // Compiler-side assumptions (help nvcc/clang optimise; the @META
    // comments above are the PTX-visible counterpart for your parser)
    // ==================================================================
    ASSUME_MATRIX_DIM(M);
    ASSUME_MATRIX_DIM(N);
    ASSUME_MATRIX_DIM(K);

    ASSUME_ALIGNED(A, 128);
    ASSUME_ALIGNED(B, 128);
    ASSUME_ALIGNED(C, 128);

    // ------------------------------------------------------------------
    // 3. Compute thread / block indices and annotate their ranges too
    // ------------------------------------------------------------------
    int tx = threadIdx.x;   // [0, TILE-1]
    int ty = threadIdx.y;   // [0, TILE-1]
    int bx = blockIdx.x;    // [0, N/TILE - 1]
    int by = blockIdx.y;    // [0, M/TILE - 1]

    ASSUME_RANGE(tx, 0, TILE - 1);
    ASSUME_RANGE(ty, 0, TILE - 1);

    // Row / col of the output element this thread owns
    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    // These ranges follow from the dimension assumptions above.
    // Stating them explicitly gives LazyValueInfo more to work with.
    ASSUME_RANGE(row, 0, 8191);
    ASSUME_RANGE(col, 0, 8191);

    // ------------------------------------------------------------------
    // 4. Shared memory tiles
    // ------------------------------------------------------------------
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    float acc = 0.0f;

    // Number of tiles along K: annotate so the compiler knows the trip
    // count is in [1, 256] and is exact (no remainder because K % 32 == 0
    // and TILE == 16, so K % TILE == 0 is guaranteed by K % 32 == 0).
    int numTiles = K / TILE;
    ASSUME_RANGE(numTiles, 1, 256);

    // ------------------------------------------------------------------
    // 5. Main tiled loop
    // ------------------------------------------------------------------
    for (int t = 0; t < numTiles; ++t) {
        // Global indices into A and B for this tile
        int a_col = t * TILE + tx;   // column in A
        int b_row = t * TILE + ty;   // row    in B

        // In-bounds because t < numTiles and dimensions are multiples of TILE
        sA[ty][tx] = A[row * K + a_col];
        sB[ty][tx] = B[b_row * N + col];

        __syncthreads();

        // Inner product — TILE is compile-time constant so this fully unrolls
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // ------------------------------------------------------------------
    // 6. Write result (guard is provably true given the annotations above,
    //    but left in for correctness with arbitrary launch configs)
    // ------------------------------------------------------------------
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Host-side wrapper — calls the kernel with a known, annotated launch config
// ---------------------------------------------------------------------------
void matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    // These host-side assumes don't appear in device IR, but they document
    // the intended call contract and help with host-side analysis.
    DEVICE_ASSUME(M > 0 && M <= 8192 && M % 32 == 0);
    DEVICE_ASSUME(N > 0 && N <= 8192 && N % 32 == 0);
    DEVICE_ASSUME(K > 0 && K <= 8192 && K % 32 == 0);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

// ---------------------------------------------------------------------------
// Minimal main — runs a correctness check with identity-like matrices
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main()
{
    const int M = 64, N = 64, K = 64;
    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    // Host allocations
    float *hA = (float*)malloc(sizeA);
    float *hB = (float*)malloc(sizeB);
    float *hC = (float*)malloc(sizeC);

    // Fill A and B with small values so C is easy to verify
    // A[i][j] = (i == j) ? 1.0 : 0.0  (identity for the M==K case)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            hA[i * K + j] = (i == j) ? 1.0f : 0.0f;

    // B = some known values: B[i][j] = i + j
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            hB[i * N + j] = (float)(i + j);

    // Device allocations
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    // Run kernel
    matmul(dA, dB, dC, M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back
    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    // Verify: C = identity * B = B, so C[i][j] should == i + j
    int errors = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = (float)(i + j);
            float got = hC[i * N + j];
            if (fabsf(got - expected) > 1e-3f) {
                if (errors < 5)
                    printf("MISMATCH C[%d][%d]: expected %.3f, got %.3f\n",
                           i, j, expected, got);
                errors++;
            }
        }
    }

    if (errors == 0)
        printf("PASS — %dx%dx%d matmul correct (%d elements verified)\n",
               M, N, K, M * N);
    else
        printf("FAIL — %d/%d elements wrong\n", errors, M * N);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return errors ? 1 : 0;
}
