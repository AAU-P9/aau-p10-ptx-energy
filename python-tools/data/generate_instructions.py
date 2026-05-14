from shared.templates import build_program, INSTRUCTION_TEMPLATES
from cubindings.cubindings import execute_code
from cubindings.cubindings_analyser import run_ptx_analyser, build_kernel_params
from pathlib import Path
import random
import shutil

min_loop_iters = 5_000_000
max_loop_iters = 50_000_000
max_instructions = 6
max_block_size = 1024
max_grid_size = 152

data_output_path = Path("/home/rasmus/aau-p10-ptx-energy/data/generates")

def run_random_kernel():
    insts = random.sample(sorted(INSTRUCTION_TEMPLATES), k=4)
    block_size = 1024
    grid_size = 152
    loop_iters = 10_000_000

    print(f"Running kernel with instructions: {insts}, block size: {block_size}, grid size: {grid_size}, loop iterations: {loop_iters}")

    src = build_program(insts, iters=loop_iters, grid=grid_size, block=block_size)

    r = execute_code(src, nvcc_args=[], binary_args=[], enable_metrics=True, debug=True)
        
    run_ptx_analyser(
            r.path,
            kernel_params=build_kernel_params(r.exports),
            debug_enabled=True,
            power_consumption_joules=r.power_metric_result.total_energy_j,
            kernel_duration_s=r.power_metric_result.kernel_duration_cpu_s,
    )

    kernel_name = "_".join(insts) + f"_r{loop_iters}_b{block_size}_g{grid_size}"
    shutil.copy(r.path / "analyser_output.json", data_output_path / f"{kernel_name}.json")

    print(f"[DEBUG] Result path: {r.path}, Duration (s): {r.power_metric_result.kernel_duration_cpu_s:.3f}, Energy (J): {r.power_metric_result.total_energy_j:.3e}")


if __name__ == "__main__":
    for i in range(30):
        energy_j = run_random_kernel()