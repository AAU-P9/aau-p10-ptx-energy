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
 *   BEGIN_KERNEL  <name>
 *   END_KERNEL    <name>
 *   PARAM         <index> <name> <type> <role> [constraints...]
 *   TILE          <dim> <size>
 *   LAUNCH        <block_x> <block_y> <block_z> <grid_expr>
 *   SHARED_MEM    <name> <elem_type> <total_bytes>
 *   LOOP          <label> <min_iters> <max_iters> <is_unrolled>
 *   LAYOUT        <name> <order> <dims_expr>
 *   ASSUME        <expr_description>
 *   CONST_TABLE   <symbol_name> <num_entries>
 *   CUSTOM        <key> <value...>
 *
 * ===== USAGE =====
 *
 *   #include "ptx_meta.h"
 *
 *   __global__ void KERNEL_LAUNCH_BOUNDS(256, 2)
 *   my_kernel(const float* __restrict__ A, int N) {
 *
 *       // All metadata in a SINGLE asm block — one begin/end pair in PTX
 *       KERNEL_META(my_kernel,
 *           META_PARAM_PTR(0, A, f32, input, ALIGN(128))
 *           META_PARAM_INT(1, N, i32, dim_rows, RANGE(1, 8192) STRIDE(32))
 *           META_TILE(x, 16)
 *           META_LAUNCH(16, 16, 1, "N/16 M/16 1")
 *           META_SHARED(smem, f32, 1024)
 *           META_LOOP(main_loop, 1, 256, false)
 *           META_LAYOUT(A, row_major, "MxK")
 *           META_ASSUME("N > 0 && N <= 8192 && N % 32 == 0")
 *       );
 *
 *       // ... actual kernel code ...
 *   }
 */

#include <cuda_runtime.h>
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
// These are raw tokens — the META_* macros stringize them with #.
// Usage: META_PARAM_PTR(0, A, ELEM_F32, input, ...)
//   -> expands ELEM_F32 to f32, then # stringizes to "f32"
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
// Same: raw tokens that get stringized by the macro.
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
// Core: emit a raw line into PTX.  Everything else builds on this.
//
// Two modes:
//   1. _PTX_META(s)     — standalone asm volatile (one begin/end pair each)
//   2. _META_LINE(s)    — produces a string fragment for use inside
//                         KERNEL_META(...) — ONE begin/end pair total
//
// Prefer KERNEL_META() for all annotations.  The standalone macros
// (KERNEL_BEGIN etc.) still exist for backward compat / one-offs.
// =========================================================================
#define _PTX_RAW(s)       asm volatile("// " s)
#define _PTX_META(s)      asm volatile("// @META:3 " s)

// String fragment: "// @META:3 <tag> ...\n"  (no asm volatile — just a string)
#define _META_LINE(s)     "// @META:3 " s "\n\t"

// Stringify with macro expansion: _STR(ELEM_F32) -> _STR2(f32) -> "f32"
// Plain # would give "ELEM_F32" (the macro name, not its value).
#define _STR2(x) #x
#define _STR(x)  _STR2(x)

// =========================================================================
// KERNEL_META — emit ALL annotations as ONE asm volatile block
//
// Usage:
//   KERNEL_META(my_kernel,
//       META_PARAM_PTR(0, A, f32, input, ALIGN(128))
//       META_PARAM_INT(1, N, i32, rows,  RANGE(1,8192))
//       META_TILE(x, 16)
//   );
//
// PTX output (one begin/end pair):
//   // begin inline asm
//   // @META:3 BEGIN_KERNEL my_kernel
//   // @META:3 PARAM 0 A ptr_f32 input align=128
//   // @META:3 PARAM 1 N i32 rows range=1:8192
//   // @META:3 TILE x 16
//   // @META:3 END_KERNEL my_kernel
//   // end inline asm
// =========================================================================
#define KERNEL_META(name, ...)  \
    asm volatile(               \
        "// @META:3 BEGIN_KERNEL " #name "\n\t" \
        __VA_ARGS__             \
        "// @META:3 END_KERNEL " #name  \
    )

// =========================================================================
// META_* — string-fragment versions for use inside KERNEL_META()
//
// These do NOT emit asm volatile themselves — they produce string literals
// that get concatenated into the single KERNEL_META asm block.
// =========================================================================

// Parameter: pointer
#define META_PARAM_PTR(idx, name, elem_type, role, constraints) \
    _META_LINE("PARAM " _STR(idx) " " _STR(name) " ptr_" _STR(elem_type) " " _STR(role) " " constraints)

// Parameter: integer
#define META_PARAM_INT(idx, name, type, role, constraints) \
    _META_LINE("PARAM " _STR(idx) " " _STR(name) " " _STR(type) " " _STR(role) " " constraints)

// Parameter: float
#define META_PARAM_FLOAT(idx, name, type, role, constraints) \
    _META_LINE("PARAM " _STR(idx) " " _STR(name) " " _STR(type) " " _STR(role) " " constraints)

// Tile size
#define META_TILE(dim, size) \
    _META_LINE("TILE " _STR(dim) " " _STR(size))

// Launch configuration
#define META_LAUNCH(bx, by, bz, grid_expr) \
    _META_LINE("LAUNCH " _STR(bx) " " _STR(by) " " _STR(bz) " " grid_expr)

// Shared memory
#define META_SHARED(name, elem_type, total_bytes) \
    _META_LINE("SHARED_MEM " _STR(name) " " _STR(elem_type) " " _STR(total_bytes))

// Loop metadata
#define META_LOOP(label, min_iters, max_iters, unrolled) \
    _META_LINE("LOOP " _STR(label) " " _STR(min_iters) " " _STR(max_iters) " " _STR(unrolled))

// Memory layout
#define META_LAYOUT(name, order, dims) \
    _META_LINE("LAYOUT " _STR(name) " " _STR(order) " " dims)

// Assumption (human-readable)
#define META_ASSUME(desc) \
    _META_LINE("ASSUME " desc)

// Constant table reference
#define META_CONST_TABLE_REF(name, count) \
    _META_LINE("CONST_TABLE __ptxmeta_" _STR(name) " " _STR(count))

// Custom key-value
#define META_CUSTOM(key, value) \
    _META_LINE("CUSTOM " key " " value)

// =========================================================================
// Portable DEVICE_ASSUME — works in nvcc (GCC) and clang
// =========================================================================
#define DEVICE_ASSUME(expr) \
    do { if (!(expr)) __builtin_unreachable(); } while (0)

// =========================================================================
// KERNEL BOUNDARY MARKERS (standalone — legacy, prefer KERNEL_META())
//
// Each of these emits its own asm volatile = its own begin/end pair.
// Kept for backward compatibility.  New code should use KERNEL_META().
// =========================================================================
#define KERNEL_BEGIN(name)       _PTX_META("BEGIN_KERNEL " #name)
#define KERNEL_END(name)        _PTX_META("END_KERNEL " #name)

// =========================================================================
// LAUNCH BOUNDS  — surfaces as .maxntid / .minnctapersm in PTX
// =========================================================================
#define KERNEL_LAUNCH_BOUNDS(max_threads, min_blocks) \
    __launch_bounds__(max_threads, min_blocks)

// =========================================================================
// PARAMETER ANNOTATIONS
//
// Typed macros for pointer and integer parameters.
// Constraint strings are concatenated — use RANGE(), STRIDE(), ALIGN().
//
// PTX output:
//   // @META:3 PARAM <idx> <name> <type> <role> <constraints>
//
// Examples:
//   KERNEL_PARAM_PTR(0, A, f32, input, ALIGN(128))
//   KERNEL_PARAM_INT(3, M, i32, rows,  RANGE(1,8192) STRIDE(32))
//   KERNEL_PARAM_INT(5, K, i32, inner,  RANGE(1,8192) STRIDE(32) MULTIPLE(16))
// =========================================================================

// Constraint tokens — use inside KERNEL_PARAM_* / META_PARAM_* last argument
#define RANGE(lo, hi)     "range=" _STR(lo) ":" _STR(hi) " "
#define STRIDE(s)         "stride=" _STR(s) " "
#define MULTIPLE(m)       "multiple=" _STR(m) " "
#define ALIGN(a)          "align=" _STR(a) " "

// Access mode tokens — prefer ACCESS_* constants from above, but these
// short aliases also work inside constraint strings.
#define READONLY          ACCESS_READONLY
#define WRITEONLY         ACCESS_WRITEONLY
#define READWRITE         ACCESS_READWRITE
#define NOALIAS           ACCESS_NOALIAS

// Pointer parameter
#define KERNEL_PARAM_PTR(idx, name, elem_type, role, constraints)  \
    _PTX_META("PARAM " _STR(idx) " " _STR(name) " ptr_" _STR(elem_type) " " _STR(role) " " constraints)

// Scalar integer parameter
#define KERNEL_PARAM_INT(idx, name, type, role, constraints)  \
    _PTX_META("PARAM " _STR(idx) " " _STR(name) " " _STR(type) " " _STR(role) " " constraints)

// Scalar float parameter
#define KERNEL_PARAM_FLOAT(idx, name, type, role, constraints)  \
    _PTX_META("PARAM " _STR(idx) " " _STR(name) " " _STR(type) " " _STR(role) " " constraints)

// =========================================================================
// TILE SIZE
//
// PTX: // @META:3 TILE <dim> <size>
// =========================================================================
#define KERNEL_TILE(dim, size) \
    _PTX_META("TILE " _STR(dim) " " _STR(size))

// =========================================================================
// LAUNCH CONFIGURATION
//
// PTX: // @META:3 LAUNCH <bx> <by> <bz> <grid_expr>
// grid_expr is a human-readable string like "N/16 M/16 1"
// =========================================================================
#define KERNEL_LAUNCH_CONFIG(bx, by, bz, grid_expr) \
    _PTX_META("LAUNCH " _STR(bx) " " _STR(by) " " _STR(bz) " " grid_expr)

// =========================================================================
// SHARED MEMORY
//
// PTX: // @META:3 SHARED_MEM <name> <elem_type> <total_bytes>
// =========================================================================
#define KERNEL_SHARED(name, elem_type, total_bytes) \
    _PTX_META("SHARED_MEM " _STR(name) " " _STR(elem_type) " " _STR(total_bytes))

// =========================================================================
// LOOP METADATA
//
// PTX: // @META:3 LOOP <label> <min_iters> <max_iters> <unrolled>
// =========================================================================
#define KERNEL_LOOP(label, min_iters, max_iters, unrolled) \
    _PTX_META("LOOP " _STR(label) " " _STR(min_iters) " " _STR(max_iters) " " _STR(unrolled))

// =========================================================================
// MEMORY LAYOUT
//
// PTX: // @META:3 LAYOUT <name> <order> <dims>
// =========================================================================
#define KERNEL_LAYOUT(name, order, dims) \
    _PTX_META("LAYOUT " _STR(name) " " _STR(order) " " dims)

// =========================================================================
// ASSUME (human-readable constraint description in PTX)
//
// PTX: // @META:3 ASSUME <description>
// =========================================================================
#define KERNEL_ASSUME(desc) \
    _PTX_META("ASSUME " desc)

// =========================================================================
// CUSTOM — escape hatch for anything not covered above
//
// PTX: // @META:3 CUSTOM <key> <value>
// =========================================================================
#define KERNEL_CUSTOM(key, value) \
    _PTX_META("CUSTOM " key " " value)

// =========================================================================
// CONSTANT METADATA — named fields
//
// Each field becomes its own __device__ __constant__ variable with a
// self-describing name:
//
//   KERNEL_CONST(reduce_sum, block_size, 256)
//     -> __device__ __constant__ int __ptxmeta_reduce_sum__block_size = 256;
//     -> PTX: .const .align 4 .b8 __ptxmeta_reduce_sum__block_size[4] = {0,1,0,0};
//
// A parser finds all fields for a kernel by matching the pattern:
//   __ptxmeta_<kernel>__<field_name>
//
// Benefits over the old positional array:
//   - Field names are embedded in the PTX symbol itself
//   - No need to remember index positions
//   - Each kernel can define different fields
//   - Self-documenting: the symbol name IS the documentation
//
// Usage:
//   // At file scope (outside any function):
//   KERNEL_CONST(my_kernel, version,        PTX_META_VERSION);
//   KERNEL_CONST(my_kernel, block_size,     256);
//   KERNEL_CONST(my_kernel, warp_size,      32);
//   KERNEL_CONST(my_kernel, max_n,          1048576);
//   KERNEL_CONST(my_kernel, align_bytes,    128);
//
//   // Inside the kernel body, emit a reference for the parser:
//   META_CONST_REF(my_kernel, version block_size warp_size max_n align_bytes)
// =========================================================================

// Single named constant field:
//   __device__ __constant__ int __ptxmeta_<kernel>__<field> = <value>;
#define KERNEL_CONST(kernel, field, value) \
    __device__ __constant__ int __ptxmeta_##kernel##__##field = (value)

// Emit a META comment listing all field names (for parser discovery)
// Usage inside KERNEL_META:
//   META_CONST_REF(my_kernel, "version block_size warp_size max_n")
#define META_CONST_REF(kernel, fields) \
    _META_LINE("CONST __ptxmeta_" _STR(kernel) " " fields)

// Standalone version (outside KERNEL_META):
#define KERNEL_CONST_REF(kernel, fields) \
    _PTX_META("CONST __ptxmeta_" _STR(kernel) " " fields)

// =========================================================================
// LEGACY: positional array table (kept for backward compatibility)
//
// Prefer KERNEL_CONST() for new code.
// =========================================================================
#define _CONST_TABLE_NAME(name) __ptxmeta_##name

#define KERNEL_CONST_TABLE(name, ...) \
    __device__ __constant__ int _CONST_TABLE_NAME(name)[] = { __VA_ARGS__ }

#define KERNEL_CONST_TABLE_REF(name, count) \
    _PTX_META("CONST_TABLE __ptxmeta_" _STR(name) " " _STR(count))

#define META_CONST_TABLE_REF(name, count) \
    _META_LINE("CONST_TABLE __ptxmeta_" _STR(name) " " _STR(count))

// =========================================================================
// ASSUME helpers — typed wrappers over DEVICE_ASSUME
//
// These emit both the compiler hint AND a PTX-visible ASSUME line.
// =========================================================================

// Range: value in [lo, hi], also tells the compiler
#define ASSUME_RANGE(x, lo, hi)    \
    DEVICE_ASSUME((x) >= (lo));    \
    DEVICE_ASSUME((x) <= (hi))

// Multiple: value % m == 0
#define ASSUME_MULTIPLE(x, m) \
    DEVICE_ASSUME(((x) % (m)) == 0)

// Pointer alignment
#define ASSUME_ALIGNED(ptr, al) \
    DEVICE_ASSUME(((uintptr_t)(ptr) % (al)) == 0)

// Combined integer constraint: range + multiple
#define ASSUME_DIM(x, lo, hi, m) \
    ASSUME_RANGE(x, lo, hi);     \
    ASSUME_MULTIPLE(x, m)
