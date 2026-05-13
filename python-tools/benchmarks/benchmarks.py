import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
from tqdm import tqdm

benchmark_prefix = "benchmark"  # Only change this if you want to invalidate the cache
model_path = Path("/home/lasse/aau-p10-ptx-energy/linear-model/linear-model.py")
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

    tqdm.write(f"Kernel: {kernel_name}, Grid Dim: {grid_dim}, Block Dim: {block_dim}")
    tqdm.write(f"Predicted Power (Joules): {predicted_power_joules}")
    tqdm.write(f"Actual Power (Joules): {actual_power_joules}")

    if actual_power_joules != 0:
        tqdm.write(
            f"Error (%): {((1 - (predicted_power_joules / actual_power_joules)) * 100):.2f}"
        )
    else:
        tqdm.write("Error (%): n/a")

    total_instructions = sum(estimate.count for estimate in predictor_result.estimates)
    tqdm.write(f"Total Instructions: {total_instructions}")

    for estimate in predictor_result.estimates:
        percentage = (
            (estimate.count / total_instructions * 100) if total_instructions else 0.0
        )
        tqdm.write(
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
    pbar: "tqdm | None" = None,
    _started: "list[bool] | None" = None,
    _start_from: str = "",
) -> None:
    if _started is not None and not _started[0]:
        if kernel_name == _start_from:
            _started[0] = True
        else:
            if pbar is not None:
                pbar.update(len(sizes))
            return
    for size in sizes:
        tag = f"{kernel_name} s={size}"
        if pbar is not None:
            pbar.set_description(tag)
            pbar.set_postfix_str("compile+run", refresh=True)
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

        if pbar is not None:
            pbar.set_postfix_str("analyse", refresh=True)
        analysis_result = run_ptx_analyser(
            execution_result.path,
            kernel_params,
            program_name=program_name,
            debug_enabled=True,
            power_consumption_joules=execution_result.power_metric_result.total_energy_j,
        )

        if data_output_path is not None:
            data_output_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                execution_result.path / "analyser_output.json",
                data_output_path / f"{kernel_name}.json",
            )

        # Add the benchmark to the feature dataset for the Neural Network.
        # csv_path = Path("/home/rasmus/aau-p10-ptx-energy/neural-network/data.csv")
        # anayser_result_to_feature_csv(kernel_name, csv_path, execution_result.power_metric_result.total_energy_j, analysis_result)

        if pbar is not None:
            pbar.set_postfix_str("predict", refresh=True)
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
        if pbar is not None:
            pbar.set_postfix_str("done", refresh=True)
            pbar.update(1)


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
    start_from = ""  # set to a kernel name to skip everything before it
    _started: list[bool] = [not start_from]
    _total = 1 + 19 + 1 + 13 + 14 + 11 + 23 + 9 + 16 + 1
    pbar = tqdm(total=_total, unit="benchmark", dynamic_ncols=True)

    run_benchmarks(
        kernel_name="flip_flop_mha",
        sizes=[
            32,
            # 64,
            # 128,
            # 256,
        ],  # N_STEPS: number of key/value time steps in the attention window
        program_path=Path(
            "/home/lasse/aau-p10-ptx-energy/experiments/apps/flip_flop_mha"
        ),
        nvcc_args_builder=lambda size: [f"-DN_STEPS={size}"],
        data_output_path=Path(
            "/home/lasse/aau-p10-ptx-energy/python-tools/nn-pairwise-occurences/kernels"
        ),
        pbar=pbar,
        _started=_started,
        _start_from=start_from,
    )

    bt_apps = Path("/home/lasse/aau-p10-ptx-energy/experiments/apps")
    # PROBLEM_SIZE sets IMAX=JMAX=KMAX, i.e. the 3-D grid side length
    # 12=class S (small), 24=class W (workstation), 64=class A
    bt_sizes = [12]  # [12, 24, 64]

    for bt_kernel in [
        "npb_bt_add",
        "npb_bt_compute_rhs_1",
        "npb_bt_compute_rhs_2",
        "npb_bt_compute_rhs_3",
        "npb_bt_compute_rhs_4",
        "npb_bt_compute_rhs_5",
        "npb_bt_compute_rhs_6",
        "npb_bt_compute_rhs_7",
        "npb_bt_compute_rhs_8",
        "npb_bt_compute_rhs_9",
        "npb_bt_x_solve_1",
        "npb_bt_x_solve_2",
        "npb_bt_x_solve_3",
        "npb_bt_y_solve_1",
        "npb_bt_y_solve_2",
        "npb_bt_y_solve_3",
        "npb_bt_z_solve_1",
        "npb_bt_z_solve_2",
        "npb_bt_z_solve_3",
    ]:
        run_benchmarks(
            kernel_name=bt_kernel,
            sizes=bt_sizes,
            program_path=bt_apps / bt_kernel / "src",
            nvcc_args_builder=lambda size: [
                f"-DPROBLEM_SIZE={size}"
            ],  # 3-D grid side length (IMAX=JMAX=KMAX)
            pbar=pbar,
            _started=_started,
            _start_from=start_from,
        )

    run_benchmarks(
        kernel_name="npb_ep_kernel",
        # M_EP is the log2 of the total sample count; NN=2^(M-16) threads are launched
        # 24=class S, 25=class W, 28=class A
        sizes=[24],  # [24, 25, 28]
        program_path=Path(
            "/home/lasse/aau-p10-ptx-energy/experiments/apps/npb_ep_kernel/src"
        ),
        nvcc_args_builder=lambda size: [
            f"-DM_EP={size}",
            "-DITERATIONS=50",
        ],  # log2 of problem size; drives grid dimension
        pbar=pbar,
        _started=_started,
        _start_from=start_from,
    )

    cg_apps = Path("/home/lasse/aau-p10-ptx-energy/experiments/apps")
    # NA is the sparse matrix dimension (number of rows/columns)
    # NZ=NA*(NONZER+1)^2 non-zeros; NONZER=7 is the sparsity parameter (fixed per NPB spec)
    # 1400=class S, 7000=class W, 14000=class A
    cg_sizes = [1400]  # [1400, 7000, 14000]
    for cg_kernel in [
        "npb_cg_kernel_one",
        "npb_cg_kernel_two",
        "npb_cg_kernel_three",
        "npb_cg_kernel_four",
        "npb_cg_kernel_five_1",
        "npb_cg_kernel_five_2",
        "npb_cg_kernel_six",
        "npb_cg_kernel_seven",
        "npb_cg_kernel_eight",
        "npb_cg_kernel_nine",
        "npb_cg_kernel_ten_1",
        "npb_cg_kernel_ten_2",
        "npb_cg_kernel_eleven",
    ]:
        run_benchmarks(
            kernel_name=cg_kernel,
            sizes=cg_sizes,
            program_path=cg_apps / cg_kernel / "src",
            nvcc_args_builder=lambda size: [
                f"-DNA={size}",
                "-DNONZER=7",
            ],  # NA=matrix dimension, NONZER=nonzeros per row (sparsity pattern)
            pbar=pbar,
            _started=_started,
            _start_from=start_from,
        )

    ft_apps = Path("/home/lasse/aau-p10-ptx-energy/experiments/apps")
    # NX×NY×NZ is the 3-D complex FFT grid
    # class 0=S (64×64×64), 1=W (128×128×32), 2=A (256×256×128)
    ft_classes = {
        0: ["-DNX=64", "-DNY=64", "-DNZ=64"],
        1: ["-DNX=128", "-DNY=128", "-DNZ=32"],
        2: ["-DNX=256", "-DNY=256", "-DNZ=128"],
    }
    for ft_kernel in [
        "npb_ft_cffts1_kernel_1",
        "npb_ft_cffts1_kernel_2",
        "npb_ft_cffts1_kernel_3",
        "npb_ft_cffts2_kernel_1",
        "npb_ft_cffts2_kernel_2",
        "npb_ft_cffts2_kernel_3",
        "npb_ft_cffts3_kernel_1",
        "npb_ft_cffts3_kernel_2",
        "npb_ft_cffts3_kernel_3",
        "npb_ft_checksum_kernel",
        "npb_ft_compute_indexmap_kernel",
        "npb_ft_compute_initial_conditions_kernel",
        "npb_ft_evolve_kernel",
        "npb_ft_init_ui_kernel",
    ]:
        run_benchmarks(
            kernel_name=ft_kernel,
            sizes=[0],  # [0, 1, 2] class index; maps to NX/NY/NZ via ft_classes
            program_path=ft_apps / ft_kernel / "src",
            nvcc_args_builder=lambda cls, _c=ft_classes: _c[
                cls
            ],  # NX=grid x-dim, NY=y-dim, NZ=z-dim
            pbar=pbar,
            _started=_started,
            _start_from=start_from,
        )

    is_apps = Path("/home/lasse/aau-p10-ptx-energy/experiments/apps")
    # IS_CLASS encodes the NPB class: 1=S (2^16 keys), 2=W (2^20 keys), 3=A (2^23 keys)
    # IS sorts integer keys; MAX_KEY = 2^(MAX_KEY_LOG_2) is the key range
    for is_kernel in [
        "npb_is_create_seq_kernel",
        "npb_is_full_verify_kernel_1",
        "npb_is_full_verify_kernel_2",
        "npb_is_full_verify_kernel_3",
        "npb_is_rank_kernel_1",
        "npb_is_rank_kernel_2",
        "npb_is_rank_kernel_3",
        "npb_is_rank_kernel_4",
        "npb_is_rank_kernel_5",
        "npb_is_rank_kernel_6",
        "npb_is_rank_kernel_7",
    ]:
        run_benchmarks(
            kernel_name=is_kernel,
            sizes=[
                1,
                # 2,
                # 3,
            ],  # IS_CLASS index: 1=S (2^16 keys), 2=W (2^20 keys), 3=A (2^23 keys)
            program_path=is_apps / is_kernel / "src",
            nvcc_args_builder=lambda cls: [
                f"-DIS_CLASS={cls}"
            ],  # sets TOTAL_KEYS, MAX_KEY, NUM_BUCKETS, CLASS char
            pbar=pbar,
            _started=_started,
            _start_from=start_from,
        )

    lu_apps = Path("/home/lasse/aau-p10-ptx-energy/experiments/apps")
    # PROBLEM_SIZE = ISIZ1=ISIZ2=ISIZ3, the cubic grid side length
    # 12=class S, 33=class W, 64=class A
    for lu_kernel in [
        "npb_lu_erhs_1",
        "npb_lu_erhs_2",
        "npb_lu_erhs_3",
        "npb_lu_erhs_4",
        "npb_lu_error_gpu_kernel",
        "npb_lu_norm_gpu_kernel",
        "npb_lu_l2norm_gpu_kernel",
        "npb_lu_pintgr_1",
        "npb_lu_pintgr_2",
        "npb_lu_pintgr_3",
        "npb_lu_pintgr_4",
        "npb_lu_rhs_1",
        "npb_lu_rhs_2",
        "npb_lu_rhs_3",
        "npb_lu_rhs_4",
        "npb_lu_setbv_1",
        "npb_lu_setbv_2",
        "npb_lu_setbv_3",
        "npb_lu_setiv_gpu_kernel",
        "npb_lu_jacld_blts_gpu_kernel",
        "npb_lu_jacu_buts_gpu_kernel",
        "npb_lu_ssor_1",
        "npb_lu_ssor_2",
    ]:
        run_benchmarks(
            kernel_name=lu_kernel,
            sizes=[
                12,
                # 33,
                # 64,
            ],  # PROBLEM_SIZE: cubic grid side length (NX=NY=NZ); 12=S, 33=W, 64=A
            program_path=lu_apps / lu_kernel / "src",
            nvcc_args_builder=lambda size: [
                f"-DPROBLEM_SIZE={size}"
            ],  # sets NX=NY=NZ=PROBLEM_SIZE
            pbar=pbar,
            _started=_started,
            _start_from=start_from,
        )

    mg_apps = Path("/home/lasse/aau-p10-ptx-energy/experiments/apps")
    # MG_PROBLEM_SIZE = finest-level cubic grid side length (power of 2)
    # 32=class S, 128=class W, 256=class A
    for mg_kernel in [
        "npb_mg_comm3_kernel_1",
        "npb_mg_comm3_kernel_2",
        "npb_mg_comm3_kernel_3",
        "npb_mg_interp_gpu_kernel",
        "npb_mg_norm2u3_gpu_kernel",
        "npb_mg_psinv_gpu_kernel",
        "npb_mg_resid_gpu_kernel",
        "npb_mg_rprj3_gpu_kernel",
        "npb_mg_zero3_gpu_kernel",
    ]:
        run_benchmarks(
            kernel_name=mg_kernel,
            sizes=[
                32,
                # 128,
                # 256,
            ],  # MG_PROBLEM_SIZE: finest-level grid side (power of 2); 32=S, 128=W, 256=A
            program_path=mg_apps / mg_kernel / "src",
            nvcc_args_builder=lambda size: [
                f"-DMG_PROBLEM_SIZE={size}"
            ],  # sets NM=size+2 (grid dim with ghost cells)
            pbar=pbar,
            _started=_started,
            _start_from=start_from,
        )

    sp_apps = Path("/home/lasse/aau-p10-ptx-energy/experiments/apps")
    # PROBLEM_SIZE = NX=NY=NZ; 12=class S, 36=class W, 64=class A
    for sp_kernel in [
        "npb_sp_add_gpu_kernel",
        "npb_sp_compute_rhs_gpu_kernel_1",
        "npb_sp_compute_rhs_gpu_kernel_2",
        "npb_sp_error_norm_gpu_kernel_1",
        "npb_sp_error_norm_gpu_kernel_2",
        "npb_sp_exact_rhs_gpu_kernel_1",
        "npb_sp_exact_rhs_gpu_kernel_2",
        "npb_sp_exact_rhs_gpu_kernel_3",
        "npb_sp_exact_rhs_gpu_kernel_4",
        "npb_sp_initialize_gpu_kernel",
        "npb_sp_rhs_norm_gpu_kernel_1",
        "npb_sp_rhs_norm_gpu_kernel_2",
        "npb_sp_txinvr_gpu_kernel",
        "npb_sp_x_solve_gpu_kernel",
        "npb_sp_y_solve_gpu_kernel",
        "npb_sp_z_solve_gpu_kernel",
    ]:
        run_benchmarks(
            kernel_name=sp_kernel,
            sizes=[12],  # [12, 36, 64] PROBLEM_SIZE: NX=NY=NZ; 12=S, 36=W, 64=A
            program_path=sp_apps / sp_kernel / "src",
            nvcc_args_builder=lambda size: [f"-DPROBLEM_SIZE={size}"],
            pbar=pbar,
            _started=_started,
            _start_from=start_from,
        )

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

    run_benchmarks(
        kernel_name="matrix_transpose",
        sizes=[1024],  # [1024, 2048, 4096, 8192, 16384]
        program_path=Path(
            "/home/lasse/aau-p10-ptx-energy/experiments/apps/matrix_transpose/src"
        ),
        nvcc_args_builder=lambda size: [f"-DSIZE_M={size}", f"-DSIZE_N={size}"],
        parameters_builder=lambda execution_result, size: [],
        data_output_path=Path(
            "/home/lasse/aau-p10-ptx-energy/python-tools/nn-single-occurrences/kernels"
        ),
        pbar=pbar,
        _started=_started,
        _start_from=start_from,
    )

    # run_benchmarks(
    #     kernel_name="matrix_mul",
    #     sizes=[1024, 2048, 4096, 8192, 16384],
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

    pbar.close()
    # write_csv_results(Path("benchmark_results.csv"))


if __name__ == "__main__":
    main()
