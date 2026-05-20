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
debug_enabled = False
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
            print(f"[WARNING] Execution time for kernel '{kernel_name}' is below the desired threshold of {desired_execution_time_s} seconds. Consider adjusting the kernel configuration to increase the execution time for more accurate energy measurements.")
            print(f"[WARNING] See the output at '{execution_result.path}' for details of the execution and analysis results.")
            raise RuntimeError(f"Execution time for kernel does not meet the desired threshold of {desired_execution_time_s} seconds.")

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

    # NPB Kernels (BT Add)

    run_kernel_configuration(
        kernel_name="npb_bt_add",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_add/src"),
        nvcc_args=[f"-DITERATIONS=15000000"],
        force_rebuild=force_rebuild,
    )

    # NPB Kernels (BT Compute)

    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_1",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_1/src"),
        nvcc_args=[f"-DITERATIONS=1000000"],
        force_rebuild=force_rebuild,
    )
    
    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_2",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_2/src"),
        nvcc_args=[f"-DITERATIONS=10000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_3",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_3/src"),
        nvcc_args=[f"-DITERATIONS=200000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_4",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_4/src"),
        nvcc_args=[f"-DITERATIONS=500000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_compute_rhs_5",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_compute_rhs_5/src"),
        nvcc_args=[f"-DITERATIONS=500000"],
        force_rebuild=force_rebuild,
    )

    # NPB Kernels (Solve)

    run_kernel_configuration(
        kernel_name="npb_bt_x_solve_1",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_x_solve_1/src"),
        nvcc_args=[f"-DITERATIONS=5000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_x_solve_2",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_x_solve_2/src"),
        nvcc_args=[f"-DITERATIONS=500000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="npb_bt_x_solve_2",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_bt_x_solve_2/src"),
        nvcc_args=[f"-DITERATIONS=500000"],
        force_rebuild=force_rebuild,
    )

    # NPB Kernels (CG)

    run_kernel_configuration(
        kernel_name="npb_cg_kernel_one",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_cg_kernel_one/src"),
        nvcc_args=[f"-DITERATIONS=10000000"],
        force_rebuild=force_rebuild,
    )

    # NPB Kernels (FT)

    run_kernel_configuration(
        kernel_name="npb_ft_cffts1_kernel_1",
        program_path=Path("/home/rasmus/aau-p10-ptx-energy/apps/npb_ft_cffts1_kernel_1/src"),
        nvcc_args=[f"-DITERATIONS=500000"],
        force_rebuild=force_rebuild,
    )

if __name__ == "__main__":
    main()
