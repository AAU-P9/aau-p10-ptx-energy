import shutil
from pathlib import Path

from cubindings.cubindings import execute_code
from cubindings.cubindings_analyser import build_kernel_params, run_ptx_analyser
from shared.templates import (
    INSTRUCTION_PARAM_TEMPLATES,
    INSTRUCTION_TEMPLATES,
    build_program,
)

min_loop_iters = 45_000_000
max_loop_iters = 100_000_000
max_instructions = 6
max_block_size = 1024
max_grid_size = 152

data_output_path = Path("/home/rasmus/aau-p10-ptx-energy/data/lasse_test")
data_output_path.mkdir(parents=True, exist_ok=True)

_ALL_INSTRUCTIONS: list = list(INSTRUCTION_TEMPLATES.keys()) + [
    (family, t)
    for family, (_, valid) in INSTRUCTION_PARAM_TEMPLATES.items()
    for t in valid
]


def _inst_label(spec) -> str:
    return spec if isinstance(spec, str) else f"{spec[0]}.{spec[1]}"


def run_all_instructions():
    block_size = 1024
    grid_size = 64
    loop_iters = min_loop_iters

    for inst in _ALL_INSTRUCTIONS:
        insts = [inst]  # Wrap in list for easier handling
        labels = [_inst_label(i) for i in insts]
        print(
            f"Running kernel with instructions: {labels}, block size: {block_size}, grid size: {grid_size}, loop iterations: {loop_iters}"
        )

        src = build_program(insts, iters=loop_iters, grid=grid_size, block=block_size)

        r = execute_code(src, nvcc_args=[], binary_args=[], enable_metrics=True, debug=True)

        run_ptx_analyser(
            r.path,
            kernel_params=build_kernel_params(r.exports),
            debug_enabled=True,
            power_consumption_joules=r.power_metric_result.total_energy_j,
            kernel_duration_s=r.power_metric_result.kernel_duration_cpu_s,
        )

        kernel_name = "_".join(labels) + f"_r{loop_iters}_b{block_size}_g{grid_size}"
        shutil.copy(
            r.path / "analyser_output.json", data_output_path / f"{kernel_name}.json"
        )

        print(
            f"[DEBUG] Result path: {r.path}, Duration (s): {r.power_metric_result.kernel_duration_cpu_s:.3f}, Energy (J): {r.power_metric_result.total_energy_j:.3e}"
        )

        kernel_name = "_".join(labels) + f"_r{loop_iters}_b{block_size}_g{grid_size}"
        shutil.copy(
            r.path / "analyser_output.json", data_output_path / f"{kernel_name}.json"
        )

        print(
            f"[DEBUG] Result path: {r.path}, Duration (s): {r.power_metric_result.kernel_duration_cpu_s:.3f}, Energy (J): {r.power_metric_result.total_energy_j:.3e}"
        )


if __name__ == "__main__":
    run_all_instructions()
