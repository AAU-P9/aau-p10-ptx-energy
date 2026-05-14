
INSTRUCTION_TEMPLATES = {
    "mul.lo.s32": {
        "setup": "int a{VAR_SUFFIX} = tid | 1, d{VAR_SUFFIX} = 1;",
        "asm":   '"mul.lo.s32 %0, %0, %1;" : "+r"(d{VAR_SUFFIX}) : "r"(a{VAR_SUFFIX})',
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "cvt.s64.s32": {
        "setup": "int a{VAR_SUFFIX} = tid; long long d{VAR_SUFFIX} = 0;",
        "asm":   ('"cvt.s64.s32 %0, %1;\\n\\t"'
                  '"cvt.s32.s64 %1, %0;"'
                  ' : "+l"(d{VAR_SUFFIX}), "+r"(a{VAR_SUFFIX})'),
        "sink":  "((long long*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "cvt.sat.f32.f32": {
        "setup": "float d{VAR_SUFFIX} = (float)tid * 0.001f;",
        "asm":   '"cvt.sat.f32.f32 %0, %0;" : "+f"(d{VAR_SUFFIX})',
        "sink":  "((float*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "cvt.u64.u32": {
        "setup": "unsigned a{VAR_SUFFIX} = (unsigned)tid; unsigned long long d{VAR_SUFFIX} = 0;",
        "asm":   ('"cvt.u64.u32 %0, %1;\\n\\t"'
                  '"cvt.u32.u64 %1, %0;"'
                  ' : "+l"(d{VAR_SUFFIX}), "+r"(a{VAR_SUFFIX})'),
        "sink":  "((unsigned long long*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "setp.lt.s32": {
        "setup": "int b{VAR_SUFFIX} = 512, d{VAR_SUFFIX} = 0;",
        "asm":   ('".reg .pred p{PRED_IDX};\\n\\t"'
                  '"setp.lt.s32 p{PRED_IDX}, %0, %1;\\n\\t"'
                  '"selp.s32 %0, 1, 0, p{PRED_IDX};"'
                  ' : "+r"(d{VAR_SUFFIX}) : "r"(b{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "setp.eq.s32": {
        "setup": "int b{VAR_SUFFIX} = 512, d{VAR_SUFFIX} = 0;",
        "asm":   ('".reg .pred p{PRED_IDX};\\n\\t"'
                  '"setp.eq.s32 p{PRED_IDX}, %0, %1;\\n\\t"'
                  '"selp.s32 %0, 1, 0, p{PRED_IDX};"'
                  ' : "+r"(d{VAR_SUFFIX}) : "r"(b{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "setp.gt.f32": {
        "setup": "float a{VAR_SUFFIX} = (float)(tid | 1); float thr{VAR_SUFFIX} = 0.5f; int d{VAR_SUFFIX} = 0;",
        "asm":   ('".reg .pred p{PRED_IDX};\\n\\t"'
                  '"setp.gt.f32 p{PRED_IDX}, %1, %2;\\n\\t"'
                  '"selp.s32 %0, 1, 0, p{PRED_IDX};"'
                  ' : "=r"(d{VAR_SUFFIX}) : "f"(a{VAR_SUFFIX}), "f"(thr{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "setp.le.s32": {
        "setup": "int b{VAR_SUFFIX} = 512, d{VAR_SUFFIX} = 0;",
        "asm":   ('".reg .pred p{PRED_IDX};\\n\\t"'
                  '"setp.le.s32 p{PRED_IDX}, %0, %1;\\n\\t"'
                  '"selp.s32 %0, 1, 0, p{PRED_IDX};"'
                  ' : "+r"(d{VAR_SUFFIX}) : "r"(b{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "setp.lt.f32": {
        "setup": "float a{VAR_SUFFIX} = (float)(tid | 1); float thr{VAR_SUFFIX} = 512.0f; int d{VAR_SUFFIX} = 0;",
        "asm":   ('".reg .pred p{PRED_IDX};\\n\\t"'
                  '"setp.lt.f32 p{PRED_IDX}, %1, %2;\\n\\t"'
                  '"selp.s32 %0, 1, 0, p{PRED_IDX};"'
                  ' : "=r"(d{VAR_SUFFIX}) : "f"(a{VAR_SUFFIX}), "f"(thr{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "setp.lt.u32": {
        "setup": "unsigned b{VAR_SUFFIX} = 512u, d{VAR_SUFFIX} = 0u;",
        "asm":   ('".reg .pred p{PRED_IDX};\\n\\t"'
                  '"setp.lt.u32 p{PRED_IDX}, %0, %1;\\n\\t"'
                  '"selp.u32 %0, 1, 0, p{PRED_IDX};"'
                  ' : "+r"(d{VAR_SUFFIX}) : "r"(b{VAR_SUFFIX})'),
        "sink":  "((unsigned*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "setp.ne.s64": {
        "setup": "long long b{VAR_SUFFIX} = 512LL, d{VAR_SUFFIX} = 0LL;",
        "asm":   ('".reg .pred p{PRED_IDX};\\n\\t"'
                  '"setp.ne.s64 p{PRED_IDX}, %0, %1;\\n\\t"'
                  '"selp.s64 %0, 1, 0, p{PRED_IDX};"'
                  ' : "+l"(d{VAR_SUFFIX}) : "l"(b{VAR_SUFFIX})'),
        "sink":  "((long long*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "not.pred": {
        "setup": "int a{VAR_SUFFIX} = tid & 1, d{VAR_SUFFIX} = 0;",
        "asm":   ('".reg .pred p{PRED_IDX}, q{PRED_IDX};\\n\\t"'
                  '"setp.ne.s32 p{PRED_IDX}, %1, 0;\\n\\t"'
                  '"not.pred q{PRED_IDX}, p{PRED_IDX};\\n\\t"'
                  '"selp.s32 %0, 1, 0, q{PRED_IDX};"'
                  ' : "=r"(d{VAR_SUFFIX}) : "r"(a{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "mov.pred": {
        "setup": "int d{VAR_SUFFIX} = tid & 1;",
        "asm":   ('".reg .pred p{PRED_IDX}, q{PRED_IDX};\\n\\t"'
                  '"setp.ne.s32 p{PRED_IDX}, %0, 0;\\n\\t"'
                  '"mov.pred q{PRED_IDX}, p{PRED_IDX};\\n\\t"'
                  '"selp.s32 %0, 1, 0, q{PRED_IDX};"'
                  ' : "+r"(d{VAR_SUFFIX})'),
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "fma.rn.f32": {
        "setup": "float a{VAR_SUFFIX} = (float)(tid + 1), b{VAR_SUFFIX} = 1.0f, c{VAR_SUFFIX} = 0.5f;",
        "asm":   '"fma.rn.f32 %0, %0, %1, %2;" : "+f"(a{VAR_SUFFIX}) : "f"(b{VAR_SUFFIX}), "f"(c{VAR_SUFFIX})',
        "sink":  "((float*)sink)[tid] = a{VAR_SUFFIX};",
    },
    "fma.rm.f32": {
        "setup": "float a{VAR_SUFFIX} = (float)(tid + 1), b{VAR_SUFFIX} = 1.0f, c{VAR_SUFFIX} = 0.5f;",
        "asm":   '"fma.rm.f32 %0, %0, %1, %2;" : "+f"(a{VAR_SUFFIX}) : "f"(b{VAR_SUFFIX}), "f"(c{VAR_SUFFIX})',
        "sink":  "((float*)sink)[tid] = a{VAR_SUFFIX};",
    },
    "div.rn.f32": {
        "setup": "float a{VAR_SUFFIX} = (float)(tid + 1), b{VAR_SUFFIX} = 2.0f;",
        "asm":   '"div.rn.f32 %0, %0, %1;" : "+f"(a{VAR_SUFFIX}) : "f"(b{VAR_SUFFIX})',
        "sink":  "((float*)sink)[tid] = a{VAR_SUFFIX};",
    },
    "shfl.sync.down.b32": {
        "setup": "unsigned d{VAR_SUFFIX} = (unsigned)tid;",
        "asm":   '"shfl.sync.down.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(d{VAR_SUFFIX})',
        "sink":  "((unsigned*)sink)[tid] = d{VAR_SUFFIX};",
    },
    "ex2.approx.ftz.f32": {
        "setup": "float d{VAR_SUFFIX} = (float)(tid & 7);",
        "asm":   '"ex2.approx.ftz.f32 %0, %0;" : "+f"(d{VAR_SUFFIX})',
        "sink":  "((float*)sink)[tid] = d{VAR_SUFFIX};",
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
    "bar.sync": {
        "setup": "int d{VAR_SUFFIX} = tid;",
        "asm":   '"bar.sync 0;" ::: "memory"',
        "sink":  "((int*)sink)[tid] = d{VAR_SUFFIX};",
    },
}

# Columns: ptx_suffix, c_type, asm_constraint, sink_cast, ptr_elem_type
_TYPE_INFO: dict[str, tuple[str, str, str, str, str]] = {
    "s8":  ("s8",  "int",               "r", "int",               "signed char"),
    "s16": ("s16", "short",              "h", "int",               "short"),
    "s32": ("s32", "int",               "r", "int",               "int"),
    "s64": ("s64", "long long",         "l", "long long",         "long long"),
    "u8":  ("u8",  "unsigned",          "r", "unsigned",          "unsigned char"),
    "u16": ("u16", "unsigned short",     "h", "unsigned",          "unsigned short"),
    "u32": ("u32", "unsigned",          "r", "unsigned",          "unsigned"),
    "u64": ("u64", "unsigned long long","l", "unsigned long long","unsigned long long"),
    "f32": ("f32", "float",             "f", "float",             "float"),
    "f64": ("f64", "double",            "d", "double",            "double"),
    "b16": ("b16", "unsigned short",    "h", "unsigned",          "unsigned short"),
    "b32": ("b32", "unsigned",          "r", "unsigned",          "unsigned"),
    "b64": ("b64", "unsigned long long","l", "unsigned long long","unsigned long long"),
}

def _binary_arith(op: str, t: str) -> dict:
    ptx, c, con, sink_c, _ = _TYPE_INFO[t]
    return {
        "setup": f"{c} b{{VAR_SUFFIX}} = 1, d{{VAR_SUFFIX}} = ({c})tid;",
        "asm":   f'"{op}.{ptx} %0, %0, %1;" : "+{con}"(d{{VAR_SUFFIX}}) : "{con}"(b{{VAR_SUFFIX}})',
        "sink":  f"(({sink_c}*)sink)[tid] = d{{VAR_SUFFIX}};",
    }

def _shift_arith(op: str, t: str) -> dict:
    # shift amount is always u32 regardless of data width
    ptx, c, con, sink_c, _ = _TYPE_INFO[t]
    return {
        "setup": f"{c} d{{VAR_SUFFIX}} = ({c})tid | 1; unsigned s{{VAR_SUFFIX}} = 1u;",
        "asm":   f'"{op}.{ptx} %0, %0, %1;" : "+{con}"(d{{VAR_SUFFIX}}) : "r"(s{{VAR_SUFFIX}})',
        "sink":  f"(({sink_c}*)sink)[tid] = d{{VAR_SUFFIX}};",
    }

def _unary_arith(op: str, t: str) -> dict:
    ptx, c, con, sink_c, _ = _TYPE_INFO[t]
    return {
        "setup": f"{c} d{{VAR_SUFFIX}} = ({c})(tid | 1);",
        "asm":   f'"{op}.{ptx} %0, %0;" : "+{con}"(d{{VAR_SUFFIX}})',
        "sink":  f"(({sink_c}*)sink)[tid] = d{{VAR_SUFFIX}};",
    }

def _mov(t: str) -> dict:
    ptx, c, con, sink_c, _ = _TYPE_INFO[t]
    return {
        "setup": f"{c} d{{VAR_SUFFIX}} = ({c})tid;",
        "asm":   f'"mov.{ptx} %0, %0;" : "+{con}"(d{{VAR_SUFFIX}})',
        "sink":  f"(({sink_c}*)sink)[tid] = d{{VAR_SUFFIX}};",
    }

def _load(t: str) -> dict:
    ptx, c, con, sink_c, ptr_c = _TYPE_INFO[t]
    return {
        "setup":     f"{c} d{{VAR_SUFFIX}} = 0; {ptr_c}* p{{VAR_SUFFIX}} = ({ptr_c}*)buf + tid;",
        "asm":       f'"ld.{ptx} %0, [%1];" : "={con}"(d{{VAR_SUFFIX}}) : "l"(p{{VAR_SUFFIX}})',
        "sink":      f"(({sink_c}*)sink)[tid] = d{{VAR_SUFFIX}};",
        "needs_buf": True,
    }

def _store(t: str) -> dict:
    ptx, c, con, _, ptr_c = _TYPE_INFO[t]
    return {
        "setup":     f"{c} v{{VAR_SUFFIX}} = ({c})tid; {ptr_c}* p{{VAR_SUFFIX}} = ({ptr_c}*)buf + tid;",
        "asm":       f'"st.{ptx} [%0], %1;" :: "l"(p{{VAR_SUFFIX}}), "{con}"(v{{VAR_SUFFIX}})',
        "sink":      "",
        "needs_buf": True,
    }

_S_INTS  = ["s16", "s32", "s64"]
_U_INTS  = ["u16", "u32", "u64"]
_INTS    = _S_INTS + _U_INTS
_FLOATS  = ["f32", "f64"]
_BTYPES  = ["b16", "b32", "b64"]

# Parameterized templates: family name -> (callable(type_name) -> template, valid_types)
# Usage in build_program: ("add", "f32"), ("sub", "s32"), ("ld", "u8"), etc.
INSTRUCTION_PARAM_TEMPLATES: dict[str, "tuple[Callable[[str], dict], list[str]]"] = {
    "add": (lambda t: _binary_arith("add", t), _INTS + _FLOATS),
    "sub": (lambda t: _binary_arith("sub", t), _INTS + _FLOATS),
    "mul": (lambda t: _binary_arith("mul", t), _FLOATS),           # integer mul needs .lo/.hi/.wide
    "div": (lambda t: _binary_arith("div", t), _INTS),             # float div needs rounding mode
    "rem": (lambda t: _binary_arith("rem", t), _INTS),
    "shl": (lambda t: _shift_arith("shl", t),  _BTYPES),
    "neg": (lambda t: _unary_arith("neg", t),  _S_INTS + _FLOATS),
    "mov": (_mov,                               _INTS + _FLOATS + _BTYPES),
    "ld":  (_load,                              ["u8", "s8"] + _INTS + _FLOATS),
    "st":  (_store,                             ["u8", "s8"] + _INTS + _FLOATS),
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

InstructionSpec = "str | tuple[str, int]"

def _resolve_template(spec: "str | tuple[str, str]") -> dict:
    if isinstance(spec, str):
        if spec not in INSTRUCTION_TEMPLATES:
            raise ValueError(f"Instruction {spec!r} not found in INSTRUCTION_TEMPLATES")
        return INSTRUCTION_TEMPLATES[spec]
    name, type_name = spec
    if name not in INSTRUCTION_PARAM_TEMPLATES:
        raise ValueError(f"Instruction family {name!r} not found in INSTRUCTION_PARAM_TEMPLATES")
    fn, valid_types = INSTRUCTION_PARAM_TEMPLATES[name]
    if type_name not in valid_types:
        raise ValueError(f"Type {type_name!r} not valid for {name!r}; valid: {valid_types}")
    return fn(type_name)

def build_program(
    instructions: "list[str | tuple[str, str]]",
    iters: int,
    grid: int = 1,
    block: int = 1,
    buf_bytes_per_thread: int = 1024
) -> str:
    templates = [_resolve_template(spec) for spec in instructions]

    # Check if any instruction needs buffer
    needs_buf: bool = any(t.get("needs_buf", False) for t in templates)

    # Build asm blocks for each instruction, using instruction index as suffix
    asm_blocks_list: list[str] = []
    for idx, t in enumerate(templates):
        asm_code = t['asm'].replace("{VAR_SUFFIX}", f"_{idx}").replace("{PRED_IDX}", str(idx))
        one_block: str = f"        asm volatile (\n            {asm_code}\n        );"
        asm_blocks_list.append(one_block)

    asm_blocks: str = "\n".join(asm_blocks_list)

    # Combine setup statements from all instructions
    setup: str = "\n    ".join(
        t["setup"].replace("{VAR_SUFFIX}", f"_{idx}")
        for idx, t in enumerate(templates)
    )

    # Combine sink statements from all instructions
    sink: str = "\n    ".join(
        t["sink"].replace("{VAR_SUFFIX}", f"_{idx}")
        for idx, t in enumerate(templates)
    )

    return KERNEL_TEMPLATE.format(
        iters=iters,
        grid=grid,
        block=block,
        extra_args=", float* __restrict__ buf" if needs_buf else "",
        setup=setup,
        pred_decls="",
        asm_blocks=asm_blocks,
        sink=sink,
        host_setup=(f"float* buf; cudaMalloc(&buf, {buf_bytes_per_thread} * {grid} * {block});"
                    if needs_buf else ""),
        launch_extra=", buf" if needs_buf else "",
        host_teardown="cudaFree(buf);" if needs_buf else "",
    )
