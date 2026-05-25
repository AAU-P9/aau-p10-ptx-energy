from typing import Callable

# ── type metadata ─────────────────────────────────────────────────────────────
_TYPE_INFO: dict[str, tuple[str, str, str, str, str]] = {
    "s8":  ("s8",  "int",                "r", "int",                "signed char"),
    "s16": ("s16", "short",              "h", "int",                "short"),
    "s32": ("s32", "int",                "r", "int",                "int"),
    "s64": ("s64", "long long",          "l", "long long",          "long long"),
    "u8":  ("u8",  "unsigned",           "r", "unsigned",           "unsigned char"),
    "u16": ("u16", "unsigned short",     "h", "unsigned",           "unsigned short"),
    "u32": ("u32", "unsigned",           "r", "unsigned",           "unsigned"),
    "u64": ("u64", "unsigned long long", "l", "unsigned long long", "unsigned long long"),
    "f32": ("f32", "float",              "f", "float",              "float"),
    "f64": ("f64", "double",             "d", "double",             "double"),
    "b16": ("b16", "unsigned short",     "h", "unsigned",           "unsigned short"),
    "b32": ("b32", "unsigned",           "r", "unsigned",           "unsigned"),
    "b64": ("b64", "unsigned long long", "l", "unsigned long long", "unsigned long long"),
}

_S_INTS = ["s8", "s16", "s32", "s64"]
_U_INTS = ["u8", "u16", "u32", "u64"]
_INTS   = _S_INTS + _U_INTS
_S_INTS_WIDE = ["s16", "s32", "s64"]
_U_INTS_WIDE = ["u16", "u32", "u64"]
_INTS_WIDE = _S_INTS_WIDE + _U_INTS_WIDE
_FLOATS = ["f32", "f64"]
_BTYPES = ["b16", "b32", "b64"]

# ── template callable type ────────────────────────────────────────────────────
# Every entry in INSTRUCTION_TEMPLATES has this signature.
TemplateFn = Callable[[str, int, str], dict]


def _require(type_name: str, valid: list[str], instr: str) -> None:
    if type_name not in valid:
        raise ValueError(f"{instr!r} does not support type {type_name!r}; valid: {valid}")


# ── template implementations ──────────────────────────────────────────────────

# vs: variable suffix
# pi: predicate index (for instructions that need multiple predicates, e.g. not.pred)
# t: type name (e.g. "s32", "f32")

def _binary_arith(op: str, valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, op)
        ptx, c, con, sink_c, _ = _TYPE_INFO[t]
        return {
            "setup": f"{c} b_{vs} = 1, d_{vs} = ({c})tid;",
            "asm":   f'"{op}.{ptx} %0, %0, %1;" : "+{con}"(d_{vs}) : "{con}"(b_{vs})',
            "sink":  f"(({sink_c}*)sink)[tid] = d_{vs};",
        }
    return make


def _shift_arith(op: str, valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, op)
        ptx, c, con, sink_c, _ = _TYPE_INFO[t]
        return {
            "setup": f"{c} d_{vs} = ({c})tid | 1; unsigned s_{vs} = 1u;",
            "asm":   f'"{op}.{ptx} %0, %0, %1;" : "+{con}"(d_{vs}) : "r"(s_{vs})',
            "sink":  f"(({sink_c}*)sink)[tid] = d_{vs};",
        }
    return make


def _unary_arith(op: str, valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, op)
        ptx, c, con, sink_c, _ = _TYPE_INFO[t]
        return {
            "setup": f"{c} d_{vs} = ({c})(tid | 1);",
            "asm":   f'"{op}.{ptx} %0, %0;" : "+{con}"(d_{vs})',
            "sink":  f"(({sink_c}*)sink)[tid] = d_{vs};",
        }
    return make


def _mov_fn(valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, "mov")
        ptx, c, con, sink_c, _ = _TYPE_INFO[t]
        return {
            "setup": f"{c} d_{vs} = ({c})tid;",
            "asm":   f'"mov.{ptx} %0, %0;" : "+{con}"(d_{vs})',
            "sink":  f"(({sink_c}*)sink)[tid] = d_{vs};",
        }
    return make


def _load_fn(valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, "ld")
        ptx, c, con, sink_c, ptr_c = _TYPE_INFO[t]
        return {
            "setup":     f"{c} d_{vs} = 0; {ptr_c}* p_{vs} = ({ptr_c}*)buf + tid;",
            "asm":       f'"ld.{ptx} %0, [%1];" : "={con}"(d_{vs}) : "l"(p_{vs})',
            "sink":      f"(({sink_c}*)sink)[tid] = d_{vs};",
            "needs_buf": True,
        }
    return make


def _store_fn(valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, "st")
        ptx, c, con, _, ptr_c = _TYPE_INFO[t]
        return {
            "setup":     f"{c} v_{vs} = ({c})tid; {ptr_c}* p_{vs} = ({ptr_c}*)buf + tid;",
            "asm":       f'"st.{ptx} [%0], %1;" :: "l"(p_{vs}), "{con}"(v_{vs})',
            "sink":      "",
            "needs_buf": True,
        }
    return make


# Instructions that have one fixed type use _require to enforce it and then
# pull everything they need from _TYPE_INFO — no hardcoded C types or PTX
# suffixes anywhere in the body.

def _mul_lo(vs: str, _pi: int, t: str) -> dict:
    _require(t, ["s32", "u32"], "mul.lo")
    ptx, c, con, sink_c, _ = _TYPE_INFO[t]
    return {
        "setup": f"{c} a_{vs} = tid | 1, d_{vs} = 1;",
        "asm":   f'"mul.lo.{ptx} %0, %0, %1;" : "+{con}"(d_{vs}) : "{con}"(a_{vs})',
        "sink":  f"(({sink_c}*)sink)[tid] = d_{vs};",
    }


def _mul_hi(vs: str, _pi: int, t: str) -> dict:
    _require(t, ["s32", "u32", "s64", "u64"], "mul.hi")
    ptx, c, con, sink_c, _ = _TYPE_INFO[t]
    return {
        "setup": f"{c} a_{vs} = tid | 1, d_{vs} = 1;",
        "asm":   f'"mul.hi.{ptx} %0, %0, %1;" : "+{con}"(d_{vs}) : "{con}"(a_{vs})',
        "sink":  f"(({sink_c}*)sink)[tid] = d_{vs};",
    }


def _cvt_sat_f32(vs: str, _pi: int, t: str) -> dict:
    _require(t, ["f32"], "cvt.sat.f32")
    return {
        "setup": f"float d_{vs} = (float)tid * 0.001f;",
        "asm":   f'"cvt.sat.f32.f32 %0, %0;" : "+f"(d_{vs})',
        "sink":  f"((float*)sink)[tid] = d_{vs};",
    }


def _cvt_widen(vs: str, _pi: int, t: str) -> dict:
    """Widen then immediately narrow back so the result is observable."""
    WIDEN = {"s32": "s64", "u32": "u64"}
    _require(t, list(WIDEN), "cvt.widen")
    dst_t = WIDEN[t]
    src_ptx, src_c, src_con, src_sink, _ = _TYPE_INFO[t]
    dst_ptx, dst_c, dst_con, dst_sink, _ = _TYPE_INFO[dst_t]
    return {
        "setup": f"{src_c} a_{vs} = tid; {dst_c} d_{vs} = 0;",
        "asm":   (f'"cvt.{dst_ptx}.{src_ptx} %0, %1;\\n\\t"'
                  f'"cvt.{src_ptx}.{dst_ptx} %1, %0;"'
                  f' : "+{dst_con}"(d_{vs}), "+{src_con}"(a_{vs})'),
        "sink":  f"(({dst_sink}*)sink)[tid] = d_{vs};",
    }


def _cvt_explicit(dst_t: str, valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, f"cvt.{dst_t}")
        src_ptx, src_c, src_con, src_sink, _ = _TYPE_INFO[t]
        dst_ptx, dst_c, dst_con, dst_sink, _ = _TYPE_INFO[dst_t]
        return {
            "setup": f"{src_c} a_{vs} = tid; {dst_c} d_{vs} = 0;",
            "asm":   f'"cvt.{dst_ptx}.{src_ptx} %0, %1;" : "={dst_con}"(d_{vs}) : "{src_con}"(a_{vs})',
            "sink":  f"(({dst_sink}*)sink)[tid] = d_{vs};",
        }
    return make

def _cvta(space: str, valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, f"cvta.{space}")
        ptx, c, con, sink_c, _ = _TYPE_INFO[t]
        return {
            "setup": f"{c} a_{vs} = tid; {c} d_{vs} = 0;",
            "asm":   f'"cvta.{space}.{ptx} %0, %1;" : "={con}"(d_{vs}) : "{con}"(a_{vs})',
            "sink":  f"(({sink_c}*)sink)[tid] = d_{vs};",
        }
    return make

def _setp(cmp: str, valid: list[str]) -> TemplateFn:
    def make(vs: str, pi: int, t: str) -> dict:
        _require(t, valid, f"setp.{cmp}")
        ptx, c, con, sink_c, _ = _TYPE_INFO[t]
        # selp uses the same type for integer predicates; f32 comparisons select into s32
        sel_ptx  = ptx if t not in _FLOATS else "s32"
        sel_c    = c   if t not in _FLOATS else "int"
        sel_con  = con if t not in _FLOATS else "r"
        sel_sink = sink_c if t not in _FLOATS else "int"
        return {
            "setup": f"{c} a_{vs} = ({c})(tid | 1); {c} b_{vs} = ({c})512; {sel_c} d_{vs} = 0;",
            "asm":   (f'".reg .pred p{pi};\\n\\t"'
                      f'"setp.{cmp}.{ptx} p{pi}, %1, %2;\\n\\t"'
                      f'"selp.{sel_ptx} %0, 1, 0, p{pi};"'
                      f' : "={sel_con}"(d_{vs}) : "{con}"(a_{vs}), "{con}"(b_{vs})'),
            "sink":  f"(({sel_sink}*)sink)[tid] = d_{vs};",
        }
    return make


def _not_pred(vs: str, pi: int, t: str) -> dict:
    _require(t, ["s32"], "not.pred")
    return {
        "setup": f"int a_{vs} = tid & 1, d_{vs} = 0;",
        "asm":   (f'".reg .pred p{pi}, q{pi};\\n\\t"'
                  f'"setp.ne.s32 p{pi}, %1, 0;\\n\\t"'
                  f'"not.pred q{pi}, p{pi};\\n\\t"'
                  f'"selp.s32 %0, 1, 0, q{pi};"'
                  f' : "=r"(d_{vs}) : "r"(a_{vs})'),
        "sink":  f"((int*)sink)[tid] = d_{vs};",
    }


def _mov_pred(vs: str, pi: int, t: str) -> dict:
    _require(t, ["s32"], "mov.pred")
    return {
        "setup": f"int d_{vs} = tid & 1;",
        "asm":   (f'".reg .pred p{pi}, q{pi};\\n\\t"'
                  f'"setp.ne.s32 p{pi}, %0, 0;\\n\\t"'
                  f'"mov.pred q{pi}, p{pi};\\n\\t"'
                  f'"selp.s32 %0, 1, 0, q{pi};"'
                  f' : "+r"(d_{vs})'),
        "sink":  f"((int*)sink)[tid] = d_{vs};",
    }


def _fma(rounding: str, valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, f"fma.{rounding}")
        ptx, c, con, sink_c, _ = _TYPE_INFO[t]
        return {
            "setup": f"{c} a_{vs} = ({c})(tid + 1), b_{vs} = 1.0, c_{vs} = 0.5;",
            "asm":   f'"fma.{rounding}.{ptx} %0, %0, %1, %2;" : "+{con}"(a_{vs}) : "{con}"(b_{vs}), "{con}"(c_{vs})',
            "sink":  f"(({sink_c}*)sink)[tid] = a_{vs};",
        }
    return make


def _div_rn(valid: list[str]) -> TemplateFn:
    def make(vs: str, _pi: int, t: str) -> dict:
        _require(t, valid, "div.rn")
        ptx, c, con, sink_c, _ = _TYPE_INFO[t]
        return {
            "setup": f"{c} a_{vs} = ({c})(tid + 1), b_{vs} = 2.0;",
            "asm":   f'"div.rn.{ptx} %0, %0, %1;" : "+{con}"(a_{vs}) : "{con}"(b_{vs})',
            "sink":  f"(({sink_c}*)sink)[tid] = a_{vs};",
        }
    return make


def _shfl_sync_down(vs: str, _pi: int, t: str) -> dict:
    _require(t, ["b32"], "shfl.sync.down")
    return {
        "setup": f"unsigned d_{vs} = (unsigned)tid;",
        "asm":   f'"shfl.sync.down.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(d_{vs})',
        "sink":  f"((unsigned*)sink)[tid] = d_{vs};",
    }


def _ex2_approx_ftz(vs: str, _pi: int, t: str) -> dict:
    _require(t, ["f32"], "ex2.approx.ftz")
    return {
        "setup": f"float d_{vs} = (float)(tid & 7);",
        "asm":   f'"ex2.approx.ftz.f32 %0, %0;" : "+f"(d_{vs})',
        "sink":  f"((float*)sink)[tid] = d_{vs};",
    }


def _bra(vs: str, _pi: int, t: str) -> dict:
    _require(t, ["s32"], "bra")
    return {
        "setup": f"int d_{vs} = tid;",
        "asm":   (f'"bra $L%=_skip;\\n\\t"'
                  f'"add.s32 %0, %0, -1;\\n\\t"'
                  f'"$L%=_skip:\\n\\t"'
                  f'"add.s32 %0, %0, 1;"'
                  f' : "+r"(d_{vs})'),
        "sink":  f"((int*)sink)[tid] = d_{vs};",
    }


def _bra_uni(vs: str, _pi: int, t: str) -> dict:
    _require(t, ["s32"], "bra.uni")
    return {
        "setup": f"int d_{vs} = tid;",
        "asm":   (f'"bra.uni $L%=_skip;\\n\\t"'
                  f'"add.s32 %0, %0, -1;\\n\\t"'
                  f'"$L%=_skip:\\n\\t"'
                  f'"add.s32 %0, %0, 1;"'
                  f' : "+r"(d_{vs})'),
        "sink":  f"((int*)sink)[tid] = d_{vs};",
    }


def _bar_sync(vs: str, _pi: int, t: str) -> dict:
    _require(t, ["s32"], "bar.sync")
    return {
        "setup": f"int d_{vs} = tid;",
        "asm":   '"bar.sync 0;" ::: "memory"',
        "sink":  f"((int*)sink)[tid] = d_{vs};",
    }


# ── master registry ───────────────────────────────────────────────────────────
# Every value is a TemplateFn: (var_suffix: str, pred_idx: int, type_name: str) -> dict

INSTRUCTION_TEMPLATES: dict[str, TemplateFn] = {
    "mul.lo":        _mul_lo,
    "mul.hi":        _mul_hi,
    "cvt.sat.f32":   _cvt_sat_f32,
    "cvt.widen":     _cvt_widen,
    "not.pred":      _not_pred,
    "mov.pred":      _mov_pred,
    "shfl.sync.down":_shfl_sync_down,
    "ex2.approx.ftz":_ex2_approx_ftz,
    "bra":           _bra,
    "bra.uni":       _bra_uni,
    "bar.sync":      _bar_sync,

    "cvt.s64":       _cvt_explicit("s64", ["s32"]),
    "cvt.u64":       _cvt_explicit("u64", ["u32"]),
    "cvta.shared":   _cvta("shared", ["u64"]),
    "cvta.local":    _cvta("local", ["u64"]),
    "cvta.const":    _cvta("const", ["u64"]),

    "add":   _binary_arith("add", _INTS_WIDE + _FLOATS),
    "sub":   _binary_arith("sub", _INTS_WIDE + _FLOATS),
    "mul":   _binary_arith("mul", _FLOATS),
    "div":   _binary_arith("div", _INTS_WIDE),
    "rem":   _binary_arith("rem", _INTS_WIDE),
    "shl":   _shift_arith ("shl", _BTYPES),
    "neg":   _unary_arith ("neg", _S_INTS_WIDE + _FLOATS),
    "mov":   _mov_fn(_INTS_WIDE + _FLOATS + _BTYPES),
    "ld":    _load_fn(_INTS + _FLOATS),
    "st":    _store_fn(_INTS + _FLOATS),

    "setp.lt": _setp("lt", _INTS_WIDE + _FLOATS),
    "setp.le": _setp("le", _INTS_WIDE + _FLOATS),
    "setp.gt": _setp("gt", _INTS_WIDE + _FLOATS),
    "setp.ge": _setp("ge", _INTS_WIDE + _FLOATS),
    "setp.eq": _setp("eq", _INTS_WIDE + _FLOATS),
    "setp.ne": _setp("ne", _INTS_WIDE + _FLOATS),

    "fma.rn": _fma("rn", _FLOATS),
    "fma.rm": _fma("rm", _FLOATS),
    "div.rn": _div_rn(_FLOATS),
}


# ── kernel template ───────────────────────────────────────────────────────────

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
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\\n";
        exit(1);
    }}
    cudaFree(sink);
    {host_teardown}
    return 0;
}}
"""

InstructionSpec = tuple[str, str]  # ("instruction", "type")  e.g. ("add", "f32"), ("mul.lo", "s32")

def build_program(
    instructions: list[InstructionSpec],
    iters: int,
    grid: int = 1,
    block: int = 1,
    buf_bytes_per_thread: int = 1024,
) -> str:
    templates = []
    for idx, (instr, type_name) in enumerate(instructions):
        if instr not in INSTRUCTION_TEMPLATES:
            raise ValueError(f"Unknown instruction {instr!r}")
        templates.append(INSTRUCTION_TEMPLATES[instr](str(idx), idx, type_name))

    needs_buf: bool = any(t.get("needs_buf", False) for t in templates)

    asm_blocks: str = "\n".join(
        f"        asm volatile (\n            {t['asm']}\n        );"
        for t in templates
    )
    setup: str = "\n    ".join(t["setup"] for t in templates)
    sink:  str = "\n    ".join(t["sink"]  for t in templates)

    return KERNEL_TEMPLATE.format(
        iters         = iters,
        grid          = grid,
        block         = block,
        extra_args    = ", float* __restrict__ buf" if needs_buf else "",
        setup         = setup,
        asm_blocks    = asm_blocks,
        sink          = sink,
        host_setup    = (f"float* buf; cudaMalloc(&buf, {buf_bytes_per_thread} * {grid} * {block});"
                         if needs_buf else ""),
        launch_extra  = ", buf" if needs_buf else "",
        host_teardown = "cudaFree(buf);" if needs_buf else "",
    )
