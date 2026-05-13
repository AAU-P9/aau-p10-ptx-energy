
INSTRUCTION_TEMPLATES = {
    "add.f32": {
        "setup": """float b{VAR_SUFFIX} = 1.5f, d{VAR_SUFFIX} = (float)tid;""",
        "asm":   '"add.f32 %0, %0, %1;" : "+f"(d{VAR_SUFFIX}) : "f"(b{VAR_SUFFIX})',
        "sink":  "((float*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "add.s32": {
        "setup": "int b{VAR_SUFFIX} = 1, d{VAR_SUFFIX} = tid;",
        "asm":   '"add.s32 %0, %0, %1;" : "+r"(d{VAR_SUFFIX}) : "r"(b{VAR_SUFFIX})',
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "add.s64": {
        "setup": "long long b{VAR_SUFFIX} = 1, d{VAR_SUFFIX} = tid;",
        "asm":   '"add.s64 %0, %0, %1;" : "+l"(d{VAR_SUFFIX}) : "l"(b{VAR_SUFFIX})',
        "sink":  "((long long*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "mul.lo.s32": {
        "setup": "int a{VAR_SUFFIX} = tid | 1, d{VAR_SUFFIX} = 1;",
        "asm":   '"mul.lo.s32 %0, %0, %1;" : "+r"(d{VAR_SUFFIX}) : "r"(a{VAR_SUFFIX})',
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "mov.b32": {
        "setup": "int tmp{VAR_SUFFIX} = tid;",
        "asm":   '"mov.b32 %0, %0;" : "+r"(tmp{VAR_SUFFIX})',
        "sink":  "((int*)sink)[tid] = tmp{VAR_SUFFIX};",
    },
    "mov.u32": {
        "setup": "unsigned tmp{VAR_SUFFIX} = tid;",
        "asm":   '"mov.u32 %0, %0;" : "+r"(tmp{VAR_SUFFIX})',
        "sink":  "((unsigned*)sink)[tid] = tmp{VAR_SUFFIX};",
    },
    "cvt.s64.s32": {
        "setup": "int a{VAR_SUFFIX} = tid; long long d{VAR_SUFFIX} = 0;",
        "asm":   ('"cvt.s64.s32 %0, %1;\\n\\t"'
                  '"cvt.s32.s64 %1, %0;"'
                  ' : "+l"(d{VAR_SUFFIX}), "+r"(a{VAR_SUFFIX})'),
        "sink":  "((long long*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "shl.b64": {
        "setup": "long long d{VAR_SUFFIX} = tid | 1; unsigned s{VAR_SUFFIX} = 1;",
        "asm":   '"shl.b64 %0, %0, %1;" : "+l"(d{VAR_SUFFIX}) : "r"(s{VAR_SUFFIX})',
        "sink":  "((long long*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "setp.lt.s32": {
        "setup": "int b{VAR_SUFFIX} = 512, d{VAR_SUFFIX} = 0;",
        "asm":   ('"setp.lt.s32 %%p{PRED_IDX}, %0, %1;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%p{PRED_IDX};"'
                  ' : "+r"(d{VAR_SUFFIX}) : "r"(b{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "not.pred": {
        "setup": "int a{VAR_SUFFIX} = tid & 1, d{VAR_SUFFIX} = 0;",
        "asm":   ('"setp.ne.s32 %%p{PRED_IDX}, %1, 0;\\n\\t"'
                  '"not.pred %%q{PRED_IDX}, %%p{PRED_IDX};\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%q{PRED_IDX};"'
                  ' : "=r"(d{VAR_SUFFIX}) : "r"(a{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "ld.f32": {
        "setup":     "float d{VAR_SUFFIX} = 0.0f; float* p{VAR_SUFFIX} = buf + tid;",
        "asm":       '"ld.f32 %0, [%1];" : "=f"(d{VAR_SUFFIX}) : "l"(p{VAR_SUFFIX})',
        "sink":      "((float*)sink)[tid] = d{VAR_SUFFIX};",
        "needs_buf": True,
    },
    "st.f32": {
        "setup":     "float v{VAR_SUFFIX} = (float)tid; float* p{VAR_SUFFIX} = buf + tid;",
        "asm":       '"st.f32 [%0], %1;" :: "l"(p{VAR_SUFFIX}), "f"(v{VAR_SUFFIX})',
        "sink":      "",
        "needs_buf": True,
    },
    "mul.f32": {
        "setup": "float b{VAR_SUFFIX} = 1.0f, d{VAR_SUFFIX} = (float)(tid + 1);",
        "asm":   '"mul.f32 %0, %0, %1;" : "+f"(d{VAR_SUFFIX}) : "f"(b{VAR_SUFFIX})',
        "sink":  "((float*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "mov.f32": {
        "setup": "float d{VAR_SUFFIX} = (float)tid;",
        "asm":   '"mov.f32 %0, %0;" : "+f"(d{VAR_SUFFIX})',
        "sink":  "((float*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "mov.pred": {
        "setup": "int d{VAR_SUFFIX} = tid & 1;",
        "asm":   ('"setp.ne.s32 %%p{PRED_IDX}, %0, 0;\\n\\t"'
                  '"mov.pred %%q{PRED_IDX}, %%p{PRED_IDX};\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%q{PRED_IDX};"'
                  ' : "+r"(d{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    # %=  expands to a unique integer per asm instance so repeat=2 gets distinct labels
    "bra": {
        "setup": "int d{VAR_SUFFIX} = tid;",
        "asm":   ('"bra $L%=_skip;\\n\\t"'
                  '"add.s32 %0, %0, -1;\\n\\t"'
                  '"$L%=_skip:\\n\\t"'
                  '"add.s32 %0, %0, 1;"'
                  ' : "+r"(d{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "bra.uni": {
        "setup": "int d{VAR_SUFFIX} = tid;",
        "asm":   ('"bra.uni $L%=_skip;\\n\\t"'
                  '"add.s32 %0, %0, -1;\\n\\t"'
                  '"$L%=_skip:\\n\\t"'
                  '"add.s32 %0, %0, 1;"'
                  ' : "+r"(d{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
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
{pred_decls}
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

def build_program(
    instructions: list[str],
    iters: int,
    grid: int = 1,
    block: int = 1,
    buf_bytes_per_thread: int = 1024
) -> str:
    """Build a CUDA kernel with multiple instructions per loop iteration.
    
    Args:
        instructions: List of instruction names to include in each loop iteration
        iters: Number of loop iterations
        grid: Grid dimension x
        block: Block dimension x
        buf_bytes_per_thread: Bytes per thread for buffer allocation
    
    Returns:
        Complete kernel source code as string
    """
    # Validate all instructions exist
    for insn in instructions:
        if insn not in INSTRUCTION_TEMPLATES:
            raise ValueError(f"Instruction {insn} not found in INSTRUCTION_TEMPLATES")
    
    # Check if any instruction needs buffer
    needs_buf: bool = any(INSTRUCTION_TEMPLATES[insn].get("needs_buf", False) for insn in instructions)
    
    # Check if any instruction needs predicates
    pred_instructions = {"setp.lt.s32", "not.pred", "mov.pred"}
    needs_pred: bool = any(insn in pred_instructions for insn in instructions)
    
    # Build asm blocks for each instruction, using instruction index as suffix
    asm_blocks_list: list[str] = []
    for idx, insn in enumerate(instructions):
        spec = INSTRUCTION_TEMPLATES[insn]
        asm_code = spec['asm'].replace("{VAR_SUFFIX}", f"_{idx}").replace("{PRED_IDX}", str(idx))
        one_block: str = f"        asm volatile (\n            {asm_code}\n        );"
        asm_blocks_list.append(one_block)
    
    asm_blocks: str = "\n".join(asm_blocks_list)
    
    # Combine setup statements from all instructions
    setup: str = "\n    ".join(
        INSTRUCTION_TEMPLATES[insn]["setup"].replace("{VAR_SUFFIX}", f"_{idx}") 
        for idx, insn in enumerate(instructions)
    )
    
    # Generate predicate declarations outside the loop
    pred_decls: str = ""
    if needs_pred:
        # Declare enough predicates for all instructions that might use them
        # Each can use up to 2 predicates (p and q), so declare 2*len(instructions)
        pred_list = []
        for i in range(len(instructions) * 2):
            pred_list.append(f"%%p{i}, %%q{i}")
        pred_decls_str = ", ".join(pred_list)
        pred_decls = f'    asm volatile (".reg .pred {pred_decls_str};" ::: );\n'
    
    # Combine sink statements from all instructions
    sink: str = "\n    ".join(
        INSTRUCTION_TEMPLATES[insn]["sink"].replace("{VAR_SUFFIX}", f"_{idx}") 
        for idx, insn in enumerate(instructions)
    )

    return KERNEL_TEMPLATE.format(
        iters=iters,
        grid=grid,
        block=block,
        extra_args=", float* __restrict__ buf" if needs_buf else "",
        setup=setup,
        pred_decls=pred_decls,
        asm_blocks=asm_blocks,
        sink=sink,
        host_setup=(f"float* buf; cudaMalloc(&buf, {buf_bytes_per_thread} * {grid} * {block});"
                    if needs_buf else ""),
        launch_extra=", buf" if needs_buf else "",
        host_teardown="cudaFree(buf);" if needs_buf else "",
    )
