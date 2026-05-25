import json
import shutil
from pathlib import Path

from cubindings.cubindings import execute_code
from cubindings.cubindings_analyser import build_kernel_params, run_ptx_analyser
from shared.templates import (
    INSTRUCTION_TEMPLATES,
    build_program,
)

debug_enabled = False

data_output_path = Path("/home/rasmus/aau-p10-ptx-energy/data/linear_model_microbenchmarks")
data_output_path.mkdir(parents=True, exist_ok=True)

# All possible types that might be supported
ALL_TYPES = [
    "s8", "s16", "s32", "s64",
    "u8", "u16", "u32", "u64",
    "f32", "f64",
    "b16", "b32", "b64",
]

def get_valid_types_for_instruction(instr: str) -> list[str]:
    """Try all types and return those that don't raise ValueError."""
    valid = []
    for type_name in ALL_TYPES:
        try:
            template_fn = INSTRUCTION_TEMPLATES[instr]
            template_fn("_test", 0, type_name)
            valid.append(type_name)
        except ValueError:
            pass
    return valid

def run_benchmark(instr: str, type_name: str, iters: int, grid: int, block: int, repeat: int):
            kernel_name = f"{instr}_{type_name}_rpt{repeat}_r{iters}_b{block}_g{grid}"
            analyser_output_path = data_output_path / f"{kernel_name}.json"

            # I forgot to use the cached version of execude_code...
            # So instead we check if the analyser output already exists and skip if it does
            if (analyser_output_path).exists():
                print(f"  [SKIP] {kernel_name} already exists")

                # Update the kernelName since I forgot to set it
                with analyser_output_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    payload["kernelName"] = f"{instr}.{type_name}"
                    with analyser_output_path.open("w", encoding="utf-8") as handle:
                        json.dump(payload, handle, indent=2)
                return

            instructions = [(instr, type_name) for _ in range(repeat)] 
            
            src = build_program(
                instructions=instructions,
                iters=iters,
                grid=grid,
                block=block,
            )

            r = execute_code(src, nvcc_args=[], binary_args=[], enable_metrics=True, debug_enabled=debug_enabled)

            run_ptx_analyser(
                r.path,
                kernel_params=build_kernel_params(r.exports),
                debug_enabled=debug_enabled,
                power_consumption_joules=r.power_metric_result.total_energy_j,
                kernel_duration_s=r.power_metric_result.kernel_duration_cpu_s,
                kernel_name=f"{instr}.{type_name}",
            )

            shutil.copy(r.path / "analyser_output.json", analyser_output_path)

            print(
                f"  ✓ {kernel_name:<60} | Duration: {r.power_metric_result.kernel_duration_cpu_s:.3f}s | Energy: {r.power_metric_result.total_energy_j:.3e}J"
            )
        

def run_all_instructions():
    iters = 20_000_000
    grid = 64
    block = 1024

    print(f"Generating microbenchmarks...")
    print(f"Configuration: iterations={iters}, grid={grid}, block={block}")
    print()

    for instr in sorted(INSTRUCTION_TEMPLATES.keys()):
        valid_types = get_valid_types_for_instruction(instr)
        
        if not valid_types:
            print(f"[SKIP] {instr}: no valid types")
            continue
        
        print(f"[INFO] {instr}: {len(valid_types)} valid types - {', '.join(valid_types)}")
        
        for type_name in valid_types:
            run_benchmark(instr, type_name, iters, grid, block, repeat=1)
            run_benchmark(instr, type_name, iters, grid, block, repeat=2)


if __name__ == "__main__":
    run_all_instructions()
