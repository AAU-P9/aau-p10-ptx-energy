import shutil
from pathlib import Path
from typing import Callable

from cubindings.cubindings_analyser import (
    run_ptx_analyser,
    build_kernel_params,
)
from cubindings.cubindings_cache import execute_program_cached
from cubindings.cubindings_predictor import run_predictor

artifacts_path = Path("/home/p10/aau-p10-ptx-energy/experiments/artifacts")
data_output_path = Path("/home/rasmus/aau-p10-ptx-energy/data/kernels")
debug_enabled = False
force_rebuild = False

def run_kernel_configurations(
    kernel_name: str,
    sizes: list[int],
    program_path: Path,
    nvcc_args_builder: Callable[[int], list[str]],
    program_name: str = "main.cu",
) -> None:

    for size in sizes:
        nvcc_args = nvcc_args_builder(size)
        execution_result = execute_program_cached(
            path=program_path,
            program_name=program_name,
            nvcc_args=nvcc_args,
            artifacts_path=artifacts_path,
            debug_enabled=debug_enabled,
            force_rebuild=force_rebuild,
        )

        run_ptx_analyser(
            execution_result.path,
            program_name=program_name,
            kernel_params=build_kernel_params(execution_result.exports),
            debug_enabled=debug_enabled,
            power_consumption_joules=execution_result.power_metric_result.total_energy_j,
            kernel_name=kernel_name,
            kernel_duration_s=execution_result.power_metric_result.kernel_duration_cpu_s,
        )

        shutil.copy(
            execution_result.path / "analyser_output.json",
            data_output_path / f"{kernel_name}_{size}.json",
        )

def main() -> None:
    run_kernel_configurations(
        kernel_name="vector_add_old",
        sizes=[2048, 4096, 8192, 16384],
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/vector_add_old/src"),
        nvcc_args_builder=lambda size: [f"-DSIZE_N={size}"],
    )

if __name__ == "__main__":
    main()
