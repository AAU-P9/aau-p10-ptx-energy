import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import shutil

from cubindings.cubindings import ExecutionResult
from cubindings.cubindings_analyser import (
    AnalyserResult,
    anayser_result_to_feature_csv,
    build_kernel_params,
    run_ptx_analyser,
)
from cubindings.cubindings_cache import execute_program_cached
from cubindings.cubindings_power import PowerMetricsResult
from cubindings.cubindings_predictor import LinearModelOutput, run_predictor

benchmark_prefix = "benchmark"  # Only change this if you want to invalidate the cache
model_path = Path("/home/rasmus/aau-p10-ptx-energy/linear-model/linear-model.py")
weights_path = Path("/home/p10/aau-p10-ptx-energy/linear-model/weights.csv")
artifacts_path = Path("/home/p10/aau-p10-ptx-energy/experiments/artifacts")
debug_enabled = False

csv_results: list["CSVResult"] = []


@dataclass
class CSVResult:
    kernel_name: str
    block_dim: int
    grid_dim: int
    actual_power_joules: float
    predicted_power_joules: float
    kernel_duration_cpu_s: float = 0.0


def concat_results(
    kernel_name: str,
    power_metric_result: PowerMetricsResult,
    predictor_result: LinearModelOutput,
    analysis_result: AnalyserResult,
) -> CSVResult:
    actual_power_joules = power_metric_result.total_energy_j
    predicted_power_joules = predictor_result.total_estimated_power_joules
    kernel_duration_cpu_s = power_metric_result.kernel_duration_cpu_s
    grid_dim = analysis_result.grid_dim
    block_dim = analysis_result.block_dim

    print(f"Kernel: {kernel_name}, Grid Dim: {grid_dim}, Block Dim: {block_dim}")
    print(f"Predicted Power (Joules): {predicted_power_joules}")
    print(f"Actual Power (Joules): {actual_power_joules}")

    if actual_power_joules != 0:
        print(
            f"Error (%): {((1 - (predicted_power_joules / actual_power_joules)) * 100):.2f}"
        )
    else:
        print("Error (%): n/a")

    total_instructions = sum(estimate.count for estimate in predictor_result.estimates)
    print(f"Total Instructions: {total_instructions}")

    for estimate in predictor_result.estimates:
        percentage = (
            (estimate.count / total_instructions * 100) if total_instructions else 0.0
        )
        print(
            f"  {estimate.instruction}: {estimate.count} occurrences, "
            f"{percentage:.2f}%, {estimate.estimated_power_joules} Joules"
        )

    csv_result = CSVResult(
        kernel_name=kernel_name,
        block_dim=block_dim.x,
        grid_dim=grid_dim.x,
        actual_power_joules=actual_power_joules,
        predicted_power_joules=predicted_power_joules,
        kernel_duration_cpu_s=kernel_duration_cpu_s,
    )
    csv_results.append(csv_result)
    return csv_result


def run_benchmarks(
    kernel_name: str,
    sizes: list[int],
    program_path: Path,
    nvcc_args_builder: Callable[[int], list[str]],
    parameters_builder: Callable[[ExecutionResult, int], list[dict[str, object]]]
    | None = None,
    program_name: str = "main.cu",
    data_output_path: Path | None = None,
) -> None:
    for size in sizes:
        nvcc_args = nvcc_args_builder(size)
        execution_result = execute_program_cached(
            path=program_path,
            program_name=program_name,
            nvcc_args=nvcc_args,
            cache_key=f"{benchmark_prefix}{kernel_name}",
            artifacts_path=artifacts_path,
            debug_enabled=True,
            force_rebuild=True,
        )

        parameters = (
            parameters_builder(execution_result, size) if parameters_builder else []
        )
        kernel_params = build_kernel_params(execution_result.exports, parameters)

        analysis_result = run_ptx_analyser(
            execution_result.path,
            kernel_params,
            program_name=program_name,
            debug_enabled=True,
            power_consumption_joules=execution_result.power_metric_result.total_energy_j,
        )

        if data_output_path is not None:
            data_output_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(execution_result.path / "analyser_output.json", data_output_path / f"{kernel_name}.json")

        # Add the benchmark to the feature dataset for the Neural Network.
        # csv_path = Path("/home/rasmus/aau-p10-ptx-energy/neural-network/data.csv")
        # anayser_result_to_feature_csv(kernel_name, csv_path, execution_result.power_metric_result.total_energy_j, analysis_result)

        predictor_result = run_predictor(
            model_path=model_path,
            output_path=execution_result.path,
            weights_path=weights_path,
            debug_enabled=debug_enabled,
        )

        concat_results(
            kernel_name,
            execution_result.power_metric_result,
            predictor_result,
            analysis_result,
        )


def write_csv_results(output_path: Path) -> None:
    with output_path.open(mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "kernel_name",
            "block_dim",
            "grid_dim",
            "actual_power_joules",
            "predicted_power_joules",
            "kernel_duration_cpu_s",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for result in csv_results:
            writer.writerow(
                {
                    "kernel_name": result.kernel_name,
                    "block_dim": result.block_dim,
                    "grid_dim": result.grid_dim,
                    "actual_power_joules": result.actual_power_joules,
                    "predicted_power_joules": result.predicted_power_joules,
                    "kernel_duration_cpu_s": result.kernel_duration_cpu_s,
                }
            )


def main() -> None:
    print("Running benchmarks...")

    # TODO: Fix the following stack trace when running the benchmarks:
    # thread 'main' (4150861) panicked at src/cfg/count_instructions.rs:140:70:
    # called `Result::unwrap()` on an `Err` value: ParseIntError { kind: InvalidDigit }
    # note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
    # run_benchmarks(
    #     kernel_name="matrix_mult_relu",
    #     sizes=[1024, 2048, 4096, 8192, 16384],
    #     program_path=Path("/home/rasmus/aau-p10-ptx-energy/experiments/apps/matrix_mult_relu/src"),
    #     nvcc_args_builder=lambda size: [f"-DSIZE_M={size}", "-DSIZE_N=32", "-DSIZE_K=32"],
    #     parameters_builder=lambda execution_result, size: [],
    #     data_output_path=Path("/home/rasmus/aau-p10-ptx-energy/python-tools/nn-pairwise-occurences/kernels"),
    # )

    # run_benchmarks(
    #     kernel_name="matrix_mul",
    #     sizes=sizes,
    #     program_path=Path("/home/rasmus/aau-p10-ptx-energy/experiments/apps/matrix_mult/src"),
    #     nvcc_args_builder=lambda size: [f"-DSIZE_M={size}", "-DSIZE_N=64", "-DSIZE_K=64"],
    #     parameters_builder=lambda execution_result, size: [],
    # )

    # run_benchmarks(
    #     kernel_name="vector_add_old",
    #     sizes=sizes,
    #     program_path=Path(
    #         "/home/rasmus/aau-p10-ptx-energy/experiments/apps/vector_add_old/src"
    #     ),
    #     nvcc_args_builder=lambda size: [f"-DSIZE_N={size}"],
    #     parameters_builder=lambda execution_result, size: [],
    # )

    # pair_benchmarks = Path("/home/lasse/aau-p10-ptx-energy/experiments/apps")
    # pair_sizes = [1]
    # pair_data_output = Path("/home/lasse/aau-p10-ptx-energy/python-tools/nn-pairwise-occurences/data")

    # run_benchmarks(
    #     kernel_name="pair_add_f32_st_global_f32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_add_f32_st_global_f32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_add_s32_add_s32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_add_s32_add_s32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_add_s32_add_s32_indep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_add_s32_add_s32_indep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_add_s64_ld_f32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_add_s64_ld_f32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_cvt_shl_add_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_cvt_shl_add_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_add_f32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_add_f32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_f32_add_f32_coalesced",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_f32_add_f32_coalesced/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_f32_add_f32_strided",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_f32_add_f32_strided/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_f32_ld_f32_indep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_f32_ld_f32_indep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_f32_mul_f32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_f32_mul_f32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_f32_st_f32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_f32_st_f32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_f64_add_f64_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_f64_add_f64_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_s32_add_s32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_s32_add_s32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_ld_shared_f32_add_f32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_ld_shared_f32_add_f32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_mov_f32_add_f32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_mov_f32_add_f32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_mov_u32_add_s32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_mov_u32_add_s32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_mov_u32_add_s32_indep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_mov_u32_add_s32_indep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_mul_f32_add_f32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_mul_f32_add_f32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_mul_f32_add_f32_indep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_mul_f32_add_f32_indep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_mul_s32_add_s32_dep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_mul_s32_add_s32_dep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_mul_s32_add_s32_indep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_mul_s32_add_s32_indep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )
    # run_benchmarks(
    #     kernel_name="pair_st_f32_st_f32_indep",
    #     sizes=pair_sizes,
    #     program_path=pair_benchmarks / "pair_st_f32_st_f32_indep/src",
    #     nvcc_args_builder=lambda size: [],
    #     data_output_path=pair_data_output,
    # )

    # write_csv_results(Path("benchmark_results.csv"))


if __name__ == "__main__":
    main()
