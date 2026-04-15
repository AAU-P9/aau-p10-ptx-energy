import csv
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import time

from cubindings import execute_code
from analyser import run_analyser

# ============================================================================
# Configuration
# ============================================================================
GRID = 152          # 2 blocks per SM on AD103 (76 SMs)
BLOCK = 1024
TARGET_NS = 100_000#5_000_000_000  # 10 seconds
PILOT_ITERS = 100_000
WEIGHTS_PATH = "/home/lasse/aau-p10-ptx-energy/linear-model/weights.csv"
BUF_BYTES_PER_THREAD = 1024
PILOT_CACHE_PATH = Path(__file__).parent / "pilot_cache.json"
METRICS_WARMUP_S = 1


def _load_pilot_cache() -> dict:
    if not PILOT_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(PILOT_CACHE_PATH.read_text())
        cfg = data.get("config", {})
        if cfg.get("pilot_iters") != PILOT_ITERS or cfg.get("target_ns") != TARGET_NS:
            print("  [cache] config changed, discarding pilot cache", flush=True)
            return {}
        return data.get("results", {})
    except Exception:
        return {}


def _save_pilot_cache(results: dict) -> None:
    data = {
        "config": {"pilot_iters": PILOT_ITERS, "target_ns": TARGET_NS},
        "results": results,
    }
    PILOT_CACHE_PATH.write_text(json.dumps(data, indent=2))


def _spinner():
    while True:
        for cursor in '|/-\\':
            yield cursor

def spinning_cursor(stop_event: threading.Event):
    spinner_generator = _spinner()
    while not stop_event.is_set():
        sys.stdout.write(next(spinner_generator))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
    sys.stdout.write(' \b')
    sys.stdout.flush()

@contextmanager
def _spinner_running(label: str):
    print(f"    {label} ", end="", flush=True)
    stop = threading.Event()
    t = threading.Thread(target=spinning_cursor, args=(stop,), daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()

# ============================================================================
# Instruction templates — latency variant only (loop-carried dep on `d`)
#
# Inline asm constraints: r=.b32/.s32/.u32, l=.b64/.s64/.u64, f=.f32
# "+x" = read+write, "=x" = write-only, "x" = read-only
# ============================================================================

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
        "asm":   ('".reg .pred %%p;\\n\\t"'
                  '"setp.lt.s32 %%p, %0, %1;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%p;"'
                  ' : "+r"(d) : "r"(b)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    "not.pred": {
        "setup": "int a = tid & 1, d = 0;",
        "asm":   ('".reg .pred %%p, %%q;\\n\\t"'
                  '"setp.ne.s32 %%p, %1, 0;\\n\\t"'
                  '"not.pred %%q, %%p;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%q;"'
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
    cudaDeviceSynchronize();
    cudaFree(sink);
    {host_teardown}
    return 0;
}}
"""


def build_program(insn, iters, repeat=1, grid=GRID, block=BLOCK):
    spec = INSTRUCTION_TEMPLATES[insn]
    needs_buf = spec.get("needs_buf", False)

    one_block = f"        asm volatile (\n            {spec['asm']}\n        );"
    asm_blocks = "\n".join([one_block] * repeat)

    return KERNEL_TEMPLATE.format(
        iters=iters,
        grid=grid,
        block=block,
        extra_args=", float* __restrict__ buf" if needs_buf else "",
        setup=spec["setup"],
        asm_blocks=asm_blocks,
        sink=spec["sink"],
        host_setup=(f"float* buf; cudaMalloc(&buf, {BUF_BYTES_PER_THREAD} * {grid} * {block});"
                    if needs_buf else ""),
        launch_extra=", buf" if needs_buf else "",
        host_teardown="cudaFree(buf);" if needs_buf else "",
    )


def _execute(insn, iters, repeat):
    src = build_program(insn, iters=iters, repeat=repeat)
    t0 = time.time()
    with _spinner_running(f"compile+run (repeat={repeat})"):
        r = execute_code(src, nvcc_args=[], binary_args=[], enable_metrics=True, metrics_sleep_time=METRICS_WARMUP_S)
    print(f" {time.time()-t0:.1f}s", flush=True)
    with _spinner_running("analysing"):
        analyser_result = run_analyser(r.path, Path(WEIGHTS_PATH))
    print(f" {time.time()-t0:.1f}s total", flush=True)
    return {
        "energy_j":    float(r.power_metric_result.total_energy_j),
        "duration_ns": float(r.power_metric_result.kernel_duration_gpu_ns),
        "predicted_j": float(analyser_result.predicted_energy_joules),
        "path": r.path,
    }


def run_one(insn, pilot_cache: dict):
    cache_key = insn
    if cache_key in pilot_cache:
        iters = pilot_cache[cache_key]
        print(f"  pilot cached -> {iters} iters", flush=True)
    else:
        print(f"  pilot ({PILOT_ITERS} iters)...", flush=True)
        pilot = _execute(insn, iters=PILOT_ITERS, repeat=1)
        iters = max(PILOT_ITERS, int(PILOT_ITERS * TARGET_NS / max(pilot["duration_ns"], 1.0)))
        print(f"  pilot done ({pilot['duration_ns']*1e-9:.2f}s) -> {iters} iters", flush=True)
        pilot_cache[cache_key] = iters
        _save_pilot_cache(pilot_cache)

    iters_for_subtract = max(PILOT_ITERS, iters // 2)

    print(f"  repeat=1 ({iters_for_subtract} iters)...", flush=True)
    r1 = _execute(insn, iters=iters_for_subtract, repeat=1)
    print(f"  repeat=2 ({iters_for_subtract} iters)...", flush=True)
    r2 = _execute(insn, iters=iters_for_subtract, repeat=2)

    delta_energy_j = r2["energy_j"] - r1["energy_j"]
    delta_dur_ns   = r2["duration_ns"] - r1["duration_ns"]
    delta_pred_j   = r2["predicted_j"] - r1["predicted_j"]
    print(f"  r1={r1['duration_ns']*1e-9:.2f}s  r2={r2['duration_ns']*1e-9:.2f}s  delta={delta_dur_ns*1e-9:.3f}s", flush=True)

    isolated_ops = iters_for_subtract * GRID * BLOCK

    print(f"  raw run ({iters} iters)...", flush=True)
    raw = _execute(insn, iters=iters, repeat=1)
    raw_total_ops = iters * GRID * BLOCK
    raw_dur_s = raw["duration_ns"] * 1e-9

    return {
        "iters":               iters,
        "iters_for_subtract":  iters_for_subtract,
        "raw_total_ops":       raw_total_ops,

        "raw_energy_j":        raw["energy_j"],
        "raw_predicted_j":     raw["predicted_j"],
        "raw_duration_s":      raw_dur_s,
        "raw_energy_per_op_j": raw["energy_j"] / raw_total_ops,
        "raw_power_w":         raw["energy_j"] / raw_dur_s,
        "raw_rel_error_pct":   abs(raw["energy_j"] - raw["predicted_j"]) / raw["energy_j"] * 100 if raw["energy_j"] else 0.0,

        "sub_energy_j":        delta_energy_j,
        "sub_predicted_j":     delta_pred_j,
        "sub_duration_s":      delta_dur_ns * 1e-9,
        "sub_energy_per_op_j": delta_energy_j / isolated_ops if isolated_ops else 0.0,
        "sub_power_w":         delta_energy_j / (delta_dur_ns * 1e-9) if delta_dur_ns > 0 else 0.0,
        "sub_rel_error_pct":   abs(delta_energy_j - delta_pred_j) / delta_energy_j * 100 if delta_energy_j else 0.0,

        "path": raw["path"],
    }


def main():
    pilot_cache = _load_pilot_cache()
    results = {}
    for insn in INSTRUCTION_TEMPLATES:
        print(f"\n=== {insn} ===")
        try:
            r = run_one(insn, pilot_cache)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        results[insn] = r
        print(f"  raw: {r['raw_energy_per_op_j']:.3e} J/op  {r['raw_power_w']:6.2f} W  {r['raw_duration_s']:5.2f} s  pred err {r['raw_rel_error_pct']:5.1f}%")
        print(f"  sub: {r['sub_energy_per_op_j']:.3e} J/op  {r['sub_power_w']:6.2f} W  {r['sub_duration_s']:5.2f} s  pred err {r['sub_rel_error_pct']:5.1f}%")

    print("\n\n" + "=" * 90)
    print(f"{'instruction':<16} {'raw J/op':>14} {'sub J/op':>14} {'raw W':>8} {'raw s':>7} {'sub err%':>10}")
    print("=" * 90)
    for insn, r in results.items():
        print(f"{insn:<16} "
              f"{r['raw_energy_per_op_j']:>14.3e} "
              f"{r['sub_energy_per_op_j']:>14.3e} "
              f"{r['raw_power_w']:>8.2f} "
              f"{r['raw_duration_s']:>7.2f} "
              f"{r['sub_rel_error_pct']:>10.1f}")

    out_path = Path(WEIGHTS_PATH)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel_name", "instruction", "count", "total_instructions", "power_consumption_joules"])
        writer.writeheader()
        for insn, r in results.items():
            kernel_name = insn.replace(".", "_")
            count = r["raw_total_ops"]
            writer.writerow({
                "kernel_name":              kernel_name,
                "instruction":              insn,
                "count":                    count,
                "total_instructions":       count,
                "power_consumption_joules": round(r["raw_energy_j"]),
            })
    print(f"\nwrote {len(results)} rows to {out_path}")

    return results


if __name__ == "__main__":
    results = main()
