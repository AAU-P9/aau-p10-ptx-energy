BUF_BYTES_PER_THREAD = 1024

INSTRUCTION_TEMPLATES = {
    "add.f32": {
        "setup": "float b = 1.5f, d = (float)tid;",
        "asm":   '"add.f32 %0, %0, %1;" : "+f"(d) : "f"(b)',
        "sink":  "((float*)sink)[tid] = d;",
    },
    "add.s32": {
        "setup": "int b = 1, d = tid;",
        "asm":   '"add.s32 %0, %0, %1;" : "+r"(d) : "r"(b)',
        "sink":  "((int*)sink)[tid] = d;",
    },
    "add.s64": {
        "setup": "long long b = 1, d = tid;",
        "asm":   '"add.s64 %0, %0, %1;" : "+l"(d) : "l"(b)',
        "sink":  "((long long*)sink)[tid] = d;",
    },
    "mul.lo.s32": {
        "setup": "int a = tid | 1, d = 1;",
        "asm":   '"mul.lo.s32 %0, %0, %1;" : "+r"(d) : "r"(a)',
        "sink":  "((int*)sink)[tid] = d;",
    },
    "mov.b32": {
        "setup": "int tmp = tid;",
        "asm":   '"mov.b32 %0, %0;" : "+r"(tmp)',
        "sink":  "((int*)sink)[tid] = tmp;",
    },
    "mov.u32": {
        "setup": "unsigned tmp = tid;",
        "asm":   '"mov.u32 %0, %0;" : "+r"(tmp)',
        "sink":  "((unsigned*)sink)[tid] = tmp;",
    },
    "cvt.s64.s32": {
        "setup": "int a = tid; long long d = 0;",
        "asm":   ('"cvt.s64.s32 %0, %1;\\n\\t"'
                  '"cvt.s32.s64 %1, %0;"'
                  ' : "+l"(d), "+r"(a)'),
        "sink":  "((long long*)sink)[tid] = d;",
    },
    "shl.b64": {
        "setup": "long long d = tid | 1; unsigned s = 1;",
        "asm":   '"shl.b64 %0, %0, %1;" : "+l"(d) : "r"(s)',
        "sink":  "((long long*)sink)[tid] = d;",
    },
    "setp.lt.s32": {
        "setup": "int b = 512, d = 0;",
        "asm":   ('".reg .pred %%p%=;\\n\\t"'
                  '"setp.lt.s32 %%p%=, %0, %1;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%p%=;"'
                  ' : "+r"(d) : "r"(b)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    "not.pred": {
        "setup": "int a = tid & 1, d = 0;",
        "asm":   ('".reg .pred %%p%=, %%q%=;\\n\\t"'
                  '"setp.ne.s32 %%p%=, %1, 0;\\n\\t"'
                  '"not.pred %%q%=, %%p%=;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%q%=;"'
                  ' : "=r"(d) : "r"(a)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    "ld.f32": {
        "setup":     "float d = 0.0f; float* p = buf + tid;",
        "asm":       '"ld.f32 %0, [%1];" : "=f"(d) : "l"(p)',
        "sink":      "((float*)sink)[tid] = d;",
        "needs_buf": True,
    },
    "st.f32": {
        "setup":     "float v = (float)tid; float* p = buf + tid;",
        "asm":       '"st.f32 [%0], %1;" :: "l"(p), "f"(v)',
        "sink":      "",
        "needs_buf": True,
    },
    "mul.f32": {
        "setup": "float b = 1.0f, d = (float)(tid + 1);",
        "asm":   '"mul.f32 %0, %0, %1;" : "+f"(d) : "f"(b)',
        "sink":  "((float*)sink)[tid] = d;",
    },
    "mov.f32": {
        "setup": "float d = (float)tid;",
        "asm":   '"mov.f32 %0, %0;" : "+f"(d)',
        "sink":  "((float*)sink)[tid] = d;",
    },
    "mov.pred": {
        "setup": "int d = tid & 1;",
        "asm":   ('".reg .pred %%p%=, %%q%=;\\n\\t"'
                  '"setp.ne.s32 %%p%=, %0, 0;\\n\\t"'
                  '"mov.pred %%q%=, %%p%=;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%q%=;"'
                  ' : "+r"(d)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    # %=  expands to a unique integer per asm instance so repeat=2 gets distinct labels
    "bra": {
        "setup": "int d = tid;",
        "asm":   ('"bra $L%=_skip;\\n\\t"'
                  '"add.s32 %0, %0, -1;\\n\\t"'
                  '"$L%=_skip:\\n\\t"'
                  '"add.s32 %0, %0, 1;"'
                  ' : "+r"(d)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    "bra.uni": {
        "setup": "int d = tid;",
        "asm":   ('"bra.uni $L%=_skip;\\n\\t"'
                  '"add.s32 %0, %0, -1;\\n\\t"'
                  '"$L%=_skip:\\n\\t"'
                  '"add.s32 %0, %0, 1;"'
                  ' : "+r"(d)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
}


# ============================================================================
# Kernel template — REPEAT copies of the asm block per loop iter
# ============================================================================

KERNEL_TEMPLATE = """
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "ptx_meta.h"
#define ITERATIONS {iters}

__global__ void ptx_kernel(void* __restrict__ sink{extra_args}) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    {setup}
    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    #pragma unroll 1
    for (int i = 0; i < ITERATIONS; ++i) {{
{asm_blocks}
    }}
    {sink}
}}

int main() {{
    void* sink;
    cudaMalloc(&sink, sizeof(long long) * {grid} * {block});
    {host_setup}
    ptx_kernel<<<{grid}, {block}>>>(sink{launch_extra});
    EXPORT_N("gridDim_x", {grid});
    EXPORT_N("gridDim_y", 1);
    EXPORT_N("gridDim_z", 1);
    EXPORT_N("blockDim_x", {block});
    EXPORT_N("blockDim_y", 1);
    EXPORT_N("blockDim_z", 1);
    cudaDeviceSynchronize();
    cudaFree(sink);
    {host_teardown}
    return 0;
}}
"""