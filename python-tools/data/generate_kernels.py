import shutil
from pathlib import Path

from cubindings.cubindings_analyser import (
    run_ptx_analyser,
    build_kernel_params,
)
from cubindings.cubindings_cache import execute_program_cached
from cubindings.cubindings_predictor import run_predictor

artifacts_path = Path("/home/p10/aau-p10-ptx-energy/experiments/artifacts")
data_output_path = Path("/home/rasmus/aau-p10-ptx-energy/data/kernels")
desired_execution_time_s = 5
debug_enabled = True
force_rebuild = False

def run_kernel_configuration(
    kernel_name: str,
    program_path: Path,
    nvcc_args: list[str],
    program_name: str = "main.cu",
    force_rebuild = False,
) -> None:
        print(f"[INFO] Estimating duration for kernel '{kernel_name}'")
        execution_result = execute_program_cached(
            path=program_path,
            program_name=program_name,
            nvcc_args=nvcc_args,
            artifacts_path=artifacts_path,
            debug_enabled=debug_enabled,
            force_rebuild=force_rebuild,
        )

        print(f"[INFO] Execution time for kernel '{kernel_name}': {execution_result.power_metric_result.kernel_duration_cpu_s:.6f} seconds")

        if (execution_result.power_metric_result.kernel_duration_cpu_s < desired_execution_time_s):
            raise RuntimeError(f"Execution time for kernel '{kernel_name}' is too short ({execution_result.power_metric_result.kernel_duration_cpu_s:.6f} seconds). Please adjust the kernel configuration to increase the execution time.")

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
            data_output_path / f"{kernel_name}.json",
        )

def main() -> None:

    # Naive Kernels

    run_kernel_configuration(
        kernel_name="vector_add_n64",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/vector_add_old/src"),
        nvcc_args=[f"-DSIZE_N={1024 * 64}"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="vector_add_n128",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/vector_add_old/src"),
        nvcc_args=[f"-DSIZE_N={1024 * 128}"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="vector_add_n256",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/vector_add_old/src"),
        nvcc_args=[f"-DSIZE_N={1024 * 256}"],
        force_rebuild=force_rebuild,
    )

    # Flop Flop Kernels

    run_kernel_configuration(
        kernel_name="flip_flop_mha",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/flip_flop_mha"),
        nvcc_args=[],
        force_rebuild=force_rebuild,
    )

    # NPB Kernels

    run_kernel_configuration(
        kernel_name="npb_bt_add",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_add/src"),
        nvcc_args=[f"-DITERATIONS=1000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_1",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_1/src"),
        nvcc_args=[f"-DITERATIONS=1000000"],
        force_rebuild=force_rebuild,
    )
    
    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_2",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_2/src"),
        nvcc_args=[f"-DITERATIONS=1000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_3",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_3/src"),
        nvcc_args=[f"-DITERATIONS=1000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_4",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_4/src"),
        nvcc_args=[f"-DITERATIONS=1000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_5",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_5/src"),
        nvcc_args=[f"-DITERATIONS=1000000"],
        force_rebuild=force_rebuild,
    )

if __name__ == "__main__":
    main()
