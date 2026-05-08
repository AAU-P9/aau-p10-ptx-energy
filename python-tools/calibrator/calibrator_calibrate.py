import csv
import json
from pathlib import Path
import shutil
import sys
import threading
import time

from cubindings.cubindings import execute_code, ExecutionResult
from cubindings.cubindings_analyser import run_ptx_analyser, build_kernel_params, anayser_result_to_feature_csv
from calibrator.calibrator import build_program, INSTRUCTION_TEMPLATES

# ============================================================================
# Configuration
# ============================================================================
GRID = 152          # 2 blocks per SM on AD103 (76 SMs)
BLOCK = 1024
TARGET_NS = 5_000_000_000  # 5 seconds
PILOT_ITERS = 100_000
WEIGHTS_OUTPUT_PATH = Path("/home/p10/aau-p10-ptx-energy/linear-model/weights.csv") # Set to p10 since all users have access there
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

# ============================================================================
# Instruction templates — latency variant only (loop-carried dep on `d`)
#
# Inline asm constraints: r=.b32/.s32/.u32, l=.b64/.s64/.u64, f=.f32
# "+x" = read+write, "=x" = write-only, "x" = read-only
# ============================================================================

def _execute(insn, iters, repeat) -> ExecutionResult:
    src = build_program(insn, iters=iters, repeat=repeat, grid=GRID, block=BLOCK)
    r = execute_code(src, nvcc_args=[], binary_args=[], enable_metrics=True, metrics_sleep_time=METRICS_WARMUP_S)

    return r


def run_one(insn, pilot_cache: dict):
    cache_key = insn
    if cache_key in pilot_cache:
        iters = pilot_cache[cache_key]
        print(f"  pilot cached -> {iters} iters", flush=True)
    else:
        print(f"  pilot ({PILOT_ITERS} iters)...", flush=True)
        pilot_r = _execute(insn, iters=PILOT_ITERS, repeat=1)
        dur_ns = pilot_r.power_metric_result.kernel_duration_gpu_ns
        iters = max(PILOT_ITERS, int(PILOT_ITERS * TARGET_NS / max(dur_ns, 1.0)))
        print(f"  pilot done ({dur_ns*1e-9:.2f}s) -> {iters} iters", flush=True)
        pilot_cache[cache_key] = iters
        _save_pilot_cache(pilot_cache)

    print(f"  repeat=1 ({iters} iters)...", flush=True)
    r1 = _execute(insn, iters=iters, repeat=1)

    print(f"  repeat=2 ({iters} iters)...", flush=True)
    time.sleep(1)  # Short pause to ensure any lingering effects from the first run are minimized
    r2 = _execute(insn, iters=iters, repeat=2)

    r1_energy_per_op_j = r1.power_metric_result.total_energy_j / (iters * GRID * BLOCK)
    r2_energy_per_op_j = r2.power_metric_result.total_energy_j / (iters * GRID * BLOCK * 2) # 2x ops in repeat=2

    print(f"r1 (s): {r1.power_metric_result.kernel_duration_gpu_ns*1e-9:.2f} r1 (J/op): {r1_energy_per_op_j:.3e}", flush=True)
    print(f"r2 (s): {r2.power_metric_result.kernel_duration_gpu_ns*1e-9:.2f} r2 (J/op): {r2_energy_per_op_j:.3e}", flush=True)

    delta_energy_j = r2.power_metric_result.total_energy_j - r1.power_metric_result.total_energy_j
    err_delta_energy_per_op_j =  abs(r2_energy_per_op_j - r1_energy_per_op_j) / ((r2_energy_per_op_j + r1_energy_per_op_j) / 2) * 100

    print(f"delta energy (J): {delta_energy_j:.3e} delta energy/op (J): {delta_energy_j / (iters * GRID * BLOCK):.3e} err delta energy/op (%): {err_delta_energy_per_op_j:.2f}", flush=True)

    # Get the instruction occurrence count from the PTX analyser
    # NOTE: This should in theory be the same as the `iters * GRID * BLOCK` but we check to be sure
    ptx1 = run_ptx_analyser(
        r1.path,
        kernel_params=build_kernel_params(r1.exports),
        power_consumption_joules=r1.power_metric_result.total_energy_j,
    )

    # Write the analyser output to the data folder for the Neural Network.
    analyser_output_file = r1.path / "analyser_output.json"
    shutil.copy(analyser_output_file, Path(f"/home/rasmus/aau-p10-ptx-energy/python-tools/nn-single-occurrences/data/{insn}.json"))


    instruction_count = ptx1.instruction_occurrences.get(insn, -1)

    print(f"  raw run ({iters} iters)...", flush=True)

    return {
        "iters":                     iters,
        "instruction_count":         instruction_count,

        "r1_energy_j":               r1.power_metric_result.total_energy_j,
        "r2_energy_j":               r2.power_metric_result.total_energy_j,

        "delta_energy_j":            delta_energy_j,
        "err_delta_energy_per_op_j": err_delta_energy_per_op_j,
        
        "r1_path":                   r1.path,
        "r2_path":                   r2.path,
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

    print("\n\n" + "=" * 90)
    print(f"{'instruction':<16} {'delta J/op':>14} {'delta W':>8} {'delta s':>7} {'delta err%':>10}")
    print("=" * 90)
    for insn, r in results.items():
        print(f"{insn:<16} "
              f"{r['err_delta_energy_per_op_j']:>8.2f} "
        )

    out_path = Path(WEIGHTS_OUTPUT_PATH)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel_name", "instruction", "count", "total_instructions", "power_consumption_joules"])
        writer.writeheader()
        for insn, r in results.items():
            kernel_name = insn.replace(".", "_")
            instruction_count = r["instruction_count"]
            writer.writerow({
                "kernel_name":              kernel_name,
                "instruction":              insn,
                "count":                    instruction_count,
                "total_instructions":       instruction_count,
                "power_consumption_joules": r["delta_energy_j"],
            })
    print(f"\nwrote {len(results)} rows to {out_path}")

    return results


if __name__ == "__main__":
    results = main()
