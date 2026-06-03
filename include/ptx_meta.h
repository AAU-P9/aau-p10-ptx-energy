#pragma once
/**
 * ptx_meta.h — Typed PTX metadata annotation framework
 *
 * A zero-cost, portable (nvcc + clang) system for embedding structured
 * metadata into PTX output that a downstream parser can extract.
 *
 * Metadata is injected as PTX comments via `asm volatile("// ...")`.
 * Zero runtime cost, zero register pressure — comments are free in PTX.
 *
 * ===== PROTOCOL =====
 *
 * All metadata lines follow the format:
 *   // @META:<version> <TAG> <fields...>
 *
 * Current version: 3
 *
 * Tags emitted:
 *   BEGIN_KERNEL  <n>
 *   END_KERNEL    <n>
 *   PARAM         <index> <n> <type> <role> [constraints...]
 *   TILE          <dim> <size>
 *   LAUNCH        <block_x> <block_y> <block_z> <grid_expr>
 *   SHARED_MEM    <n> <elem_type> <total_bytes>
 *   LOOP          <label> <min_iters> <max_iters> <is_unrolled>
 *   LAYOUT        <n> <order> <dims_expr>
 *   ASSUME        <expr_description>
 *   CONST_TABLE   <symbol_name> <num_entries>
 *   CUSTOM        <key> <value...>
 *
 * ===== USAGE =====
 *
 * Every META_* macro is standalone — usable anywhere in device code:
 *
 *   __global__ void KERNEL_LAUNCH_BOUNDS(256, 2)
 *   my_kernel(const float* __restrict__ A, int N) {
 *
 *       META_BEGIN_KERNEL(my_kernel);
 *       META_PARAM_PTR(0, A, f32, input, ALIGN(128));
 *       META_PARAM_INT(1, N, i32, dim_rows, RANGE(1, 8192) STRIDE(32));
 *       META_TILE(x, 16);
 *       META_LAUNCH(16, 16, 1, "N/16 M/16 1");
 *       META_SHARED(smem, f32, 1024);
 *       META_LOOP(main_loop, 1, 256, false);
 *       META_LAYOUT(A, row_major, "MxK");
 *       META_ASSUME("N > 0 && N <= 8192 && N % 32 == 0");
 *       META_CUSTOM("algorithm", "tiled_gemm");
 *       META_END_KERNEL(my_kernel);
 *
 *       // ... actual kernel code ...
 *   }
 *
 * Or batch into a single asm block with KERNEL_META() + _META_FRAG_*:
 *
 *   KERNEL_META(my_kernel,
 *       _META_FRAG_PARAM_PTR(0, A, f32, input, ALIGN(128))
 *       _META_FRAG_PARAM_INT(1, N, i32, dim_rows, RANGE(1, 8192))
 *       _META_FRAG_TILE(x, 16)
 *   );
 */
#include <stdint.h>

// =========================================================================
// Protocol version — bump when changing the tag format
// =========================================================================
#define PTX_META_VERSION 3

// =========================================================================
// FIXED CONSTANTS — use these instead of raw strings/numbers
//
// These are not user-specified; they are part of the ptx_meta protocol.
// Using named constants prevents typos and makes code self-documenting.
// =========================================================================

// --- Access modes (for pointer parameter constraints) --------------------
#define ACCESS_READONLY     "readonly "
#define ACCESS_WRITEONLY    "writeonly "
#define ACCESS_READWRITE    "readwrite "
#define ACCESS_NOALIAS      "noalias "

// --- Element types (for PARAM_PTR and SHARED_MEM) ------------------------
#define ELEM_F16            f16
#define ELEM_F32            f32
#define ELEM_F64            f64
#define ELEM_I8             i8
#define ELEM_I16            i16
#define ELEM_I32            i32
#define ELEM_I64            i64
#define ELEM_U8             u8
#define ELEM_U16            u16
#define ELEM_U32            u32
#define ELEM_U64            u64

// --- Scalar types (for PARAM_INT / PARAM_FLOAT) --------------------------
#define TYPE_I32            i32
#define TYPE_I64            i64
#define TYPE_U32            u32
#define TYPE_U64            u64
#define TYPE_F32            f32
#define TYPE_F64            f64

// --- Memory layout orders (for LAYOUT) -----------------------------------
#define LAYOUT_LINEAR       linear
#define LAYOUT_ROW_MAJOR    row_major
#define LAYOUT_COL_MAJOR    col_major

// --- Tile dimensions (for TILE) ------------------------------------------
#define DIM_X               x
#define DIM_Y               y
#define DIM_Z               z

// --- Common alignment values (bytes) -------------------------------------
#define ALIGN_DEFAULT       16
#define ALIGN_WARP          128     // 32 threads × 4 bytes
#define ALIGN_CACHELINE     128     // L2 cache line on most NVIDIA GPUs

// --- NVIDIA warp size (constant across all current architectures) --------
#define WARP_SIZE_CONST     32

// =========================================================================
// Internal primitives
// =========================================================================
#define _PTX_RAW(s)       asm volatile("// " s)
#define _PTX_META(s)      asm volatile("// @META:3 " s)

// String fragment for KERNEL_META batching (no asm volatile)
#define _META_LINE(s)     "// @META:3 " s "\n\t"

// Stringify with macro expansion
#define _STR2(x) #x
#define _STR(x)  _STR2(x)

// =========================================================================
// Constraint tokens — use inside META_PARAM_* last argument
// =========================================================================
#define RANGE(lo, hi)     "range=" _STR(lo) ":" _STR(hi) " "
#define STRIDE(s)         "stride=" _STR(s) " "
#define MULTIPLE(m)       "multiple=" _STR(m) " "
#define ALIGN(a)          "align=" _STR(a) " "

// Access mode short aliases
#define READONLY          ACCESS_READONLY
#define WRITEONLY         ACCESS_WRITEONLY
#define READWRITE         ACCESS_READWRITE
#define NOALIAS           ACCESS_NOALIAS

// =========================================================================
// META_* — STANDALONE macros (the public API)
//
// Each emits its own asm volatile. Usable anywhere in device code.
// Bracket with META_BEGIN_KERNEL / META_END_KERNEL.
// =========================================================================

#define META_BEGIN_KERNEL(name)  _PTX_META("BEGIN_KERNEL " #name)
#define META_END_KERNEL(name)   _PTX_META("END_KERNEL " #name)

#define META_PARAM_PTR(idx, name, elem_type, role, constraints) \
    _PTX_META("PARAM " _STR(idx) " " _STR(name) " ptr_" _STR(elem_type) " " _STR(role) " " constraints)

#define META_PARAM_INT(idx, name, type, role, constraints) \
    _PTX_META("PARAM " _STR(idx) " " _STR(name) " " _STR(type) " " _STR(role) " " constraints)

#define META_PARAM_FLOAT(idx, name, type, role, constraints) \
    _PTX_META("PARAM " _STR(idx) " " _STR(name) " " _STR(type) " " _STR(role) " " constraints)

#define META_TILE(dim, size) \
    _PTX_META("TILE " _STR(dim) " " _STR(size))

#define META_LAUNCH(bx, by, bz, grid_expr) \
    _PTX_META("LAUNCH " _STR(bx) " " _STR(by) " " _STR(bz) " " grid_expr)

#define META_SHARED(name, elem_type, total_bytes) \
    _PTX_META("SHARED_MEM " _STR(name) " " _STR(elem_type) " " _STR(total_bytes))

// min_iters/max_iters are emitted via "n" (immediate) operands so the
// compiler evaluates template params / constexpr to REAL numbers. Stringify
// (_STR) only sees preprocessor tokens, so template params like BK would
// leak through as symbolic text. Operands require compile-time int constants.
#define META_LOOP(label, min_iters, max_iters, unrolled) \
    asm volatile("// @META:3 LOOP " _STR(label) " %0 %1 " _STR(unrolled) \
                 :: "n"((int)(min_iters)), "n"((int)(max_iters)))

#define META_LAYOUT(name, order, dims) \
    _PTX_META("LAYOUT " _STR(name) " " _STR(order) " " dims)

#define META_ASSUME(desc) \
    _PTX_META("ASSUME " desc)

#define META_CONST_TABLE_REF(name, count) \
    _PTX_META("CONST_TABLE __ptxmeta_" _STR(name) " " _STR(count))

#define META_CONST_REF(kernel, fields) \
    _PTX_META("CONST __ptxmeta_" _STR(kernel) " " fields)

#define META_CUSTOM(key, value) \
    _PTX_META("CUSTOM " key " " value)

// =========================================================================
// _META_FRAG_* — string-fragment versions for KERNEL_META() batching
//
// These produce string literals that get concatenated into the single
// KERNEL_META asm block. Internal — prefer standalone META_* above.
// =========================================================================

#define _META_FRAG_PARAM_PTR(idx, name, elem_type, role, constraints) \
    _META_LINE("PARAM " _STR(idx) " " _STR(name) " ptr_" _STR(elem_type) " " _STR(role) " " constraints)

#define _META_FRAG_PARAM_INT(idx, name, type, role, constraints) \
    _META_LINE("PARAM " _STR(idx) " " _STR(name) " " _STR(type) " " _STR(role) " " constraints)

#define _META_FRAG_PARAM_FLOAT(idx, name, type, role, constraints) \
    _META_LINE("PARAM " _STR(idx) " " _STR(name) " " _STR(type) " " _STR(role) " " constraints)

#define _META_FRAG_TILE(dim, size) \
    _META_LINE("TILE " _STR(dim) " " _STR(size))

#define _META_FRAG_LAUNCH(bx, by, bz, grid_expr) \
    _META_LINE("LAUNCH " _STR(bx) " " _STR(by) " " _STR(bz) " " grid_expr)

#define _META_FRAG_SHARED(name, elem_type, total_bytes) \
    _META_LINE("SHARED_MEM " _STR(name) " " _STR(elem_type) " " _STR(total_bytes))

#define _META_FRAG_LOOP(label, min_iters, max_iters, unrolled) \
    _META_LINE("LOOP " _STR(label) " " _STR(min_iters) " " _STR(max_iters) " " _STR(unrolled))

#define _META_FRAG_LAYOUT(name, order, dims) \
    _META_LINE("LAYOUT " _STR(name) " " _STR(order) " " dims)

#define _META_FRAG_ASSUME(desc) \
    _META_LINE("ASSUME " desc)

#define _META_FRAG_CONST_TABLE_REF(name, count) \
    _META_LINE("CONST_TABLE __ptxmeta_" _STR(name) " " _STR(count))

#define _META_FRAG_CONST_REF(kernel, fields) \
    _META_LINE("CONST __ptxmeta_" _STR(kernel) " " fields)

#define _META_FRAG_CUSTOM(key, value) \
    _META_LINE("CUSTOM " key " " value)

// =========================================================================
// KERNEL_META — batch annotations into ONE asm volatile block
//
// Uses _META_FRAG_* inside. Automatically wraps with BEGIN/END_KERNEL.
//
// Usage:
//   KERNEL_META(my_kernel,
//       _META_FRAG_PARAM_PTR(0, A, f32, input, ALIGN(128))
//       _META_FRAG_PARAM_INT(1, N, i32, rows,  RANGE(1,8192))
//       _META_FRAG_TILE(x, 16)
//   );
// =========================================================================
#define KERNEL_META(name, ...)  \
    asm volatile(               \
        "// @META:3 BEGIN_KERNEL " #name "\n\t" \
        __VA_ARGS__             \
        "// @META:3 END_KERNEL " #name  \
    )

// =========================================================================
// Portable DEVICE_ASSUME — works in nvcc (GCC) and clang
// =========================================================================
#define DEVICE_ASSUME(expr) \
    do { if (!(expr)) __builtin_unreachable(); } while (0)

// =========================================================================
// LAUNCH BOUNDS  — surfaces as .maxntid / .minnctapersm in PTX
// =========================================================================
#define KERNEL_LAUNCH_BOUNDS(max_threads, min_blocks) \
    __launch_bounds__(max_threads, min_blocks)

// =========================================================================
// CONSTANT METADATA — named fields
//
//   KERNEL_CONST(reduce_sum, block_size, 256)
//     -> __device__ __constant__ int __ptxmeta_reduce_sum__block_size = 256;
// =========================================================================
#define KERNEL_CONST(kernel, field, value) \
    __device__ __constant__ int __ptxmeta_##kernel##__##field = (value)

// =========================================================================
// LEGACY: positional array table (kept for backward compatibility)
// =========================================================================
#define _CONST_TABLE_NAME(name) __ptxmeta_##name

#define KERNEL_CONST_TABLE(name, ...) \
    __device__ __constant__ int _CONST_TABLE_NAME(name)[] = { __VA_ARGS__ }

// =========================================================================
// ASSUME helpers — typed wrappers over DEVICE_ASSUME
// =========================================================================

#define ASSUME_RANGE(x, lo, hi)    \
    DEVICE_ASSUME((x) >= (lo));    \
    DEVICE_ASSUME((x) <= (hi))

#define ASSUME_MULTIPLE(x, m) \
    DEVICE_ASSUME(((x) % (m)) == 0)

#define ASSUME_ALIGNED(ptr, al) \
    DEVICE_ASSUME(((uintptr_t)(ptr) % (al)) == 0)

#define ASSUME_DIM(x, lo, hi, m) \
    ASSUME_RANGE(x, lo, hi);     \
    ASSUME_MULTIPLE(x, m)
