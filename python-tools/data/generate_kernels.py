import shutil
from pathlib import Path

from cubindings.cubindings import ExecutionResult
from cubindings.cubindings_analyser import (
    build_kernel_params,
    run_ptx_analyser,
)
from cubindings.cubindings_cache import execute_program_cached

_project_root = Path(__file__).parents[2]
artifacts_path = Path("/home/p10/aau-p10-ptx-energy/experiments/artifacts")
data_output_path = _project_root / "data/kernels"
desired_execution_time_s = 10
debug_enabled = False
force_rebuild = True

short_kernels: list[tuple[str, float]] = []
analyzed_kernels: list[tuple[str, ExecutionResult]] = []


def run_kernel_configuration(
    kernel_name: str,
    program_path: Path,
    nvcc_args: list[str],
    program_name: str = "main.cu",
    force_rebuild=False,
) -> None:
    print(f"[INFO] Estimating duration for kernel '{kernel_name}'")
    cache_execution_result = execute_program_cached(
        path=program_path,
        program_name=program_name,
        nvcc_args=nvcc_args,
        artifacts_path=artifacts_path,
        debug_enabled=debug_enabled,
        force_rebuild=force_rebuild,
    )

    execution_result = cache_execution_result.execution_result

    runtime_s = execution_result.power_metric_result.kernel_duration_cpu_s
    print(f"[RUNTIME] kernel='{kernel_name}' duration={runtime_s:.3f}s")

    if runtime_s < desired_execution_time_s:
        print(
            f"[WARNING] kernel='{kernel_name}' duration={runtime_s:.3f}s below threshold, skipping analyser"
        )
        short_kernels.append((kernel_name, runtime_s))
        return

    if cache_execution_result.cache_hit:
        print(
            f"[INFO] Cache hit for kernel '{kernel_name}', skipping analyser execution."
        )
    else:
        run_ptx_analyser(
            execution_result.path,
            program_name=program_name,
            kernel_params=build_kernel_params(execution_result.exports),
            debug_enabled=debug_enabled,
            power_consumption_joules=execution_result.power_metric_result.total_energy_j,
            kernel_name=kernel_name,
            kernel_duration_s=runtime_s,
        )

        analyzed_kernels.append((kernel_name, execution_result))

    shutil.copy(
        execution_result.path / "analyser_output.json",
        data_output_path / f"{kernel_name}.json",
    )


def main() -> None:
    apps = _project_root / "apps"

    # Naive Kernels

    run_kernel_configuration(
        kernel_name="vector_add_n64",
        program_path=apps / "vector_add_old/src",
        nvcc_args=[f"-DSIZE_N={1024 * 64}", "-DITERATIONS=20000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="vector_add_n128",
        program_path=apps / "vector_add_old/src",
        nvcc_args=[f"-DSIZE_N={1024 * 128}", "-DITERATIONS=10000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="vector_add_n256",
        program_path=apps / "vector_add_old/src",
        nvcc_args=[f"-DSIZE_N={1024 * 256}", "-DITERATIONS=10000000"],
        force_rebuild=force_rebuild,
    )

    # Flip Flop Kernels

    run_kernel_configuration(
        kernel_name="flip_flop_mha",
        program_path=apps / "flip_flop_mha",
        nvcc_args=[],
        force_rebuild=force_rebuild,
    )

    # SB (Synthetic Benchmark) Kernels
    # FLOPS_PER_LOAD sweeps arith intensity from memory-bound to compute-bound (1,4,16,64)
    # DATA_LEN = 1024*1024 elements; ITERATIONS = outer loop count

    run_kernel_configuration(
        kernel_name="sb_arith_intensity",
        program_path=apps / "sb_arith_intensity",
        program_name="arith_intensity.cu",
        nvcc_args=["-DITERATIONS=500000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_divergence_50pct",
        program_path=apps / "sb_divergence_50pct",
        program_name="divergence_50pct.cu",
        nvcc_args=["-DITERATIONS=32000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_fp32_int32_mix",
        program_path=apps / "sb_fp32_int32_mix",
        program_name="fp32_int32_mix.cu",
        nvcc_args=["-DITERATIONS=42000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_fp32_pure",
        program_path=apps / "sb_fp32_pure",
        program_name="fp32_pure.cu",
        nvcc_args=["-DITERATIONS=53000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_fp64_pure",
        program_path=apps / "sb_fp64_pure",
        program_name="fp64_pure.cu",
        nvcc_args=["-DITERATIONS=18000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_gmem_coalesced",
        program_path=apps / "sb_gmem_coalesced",
        program_name="gmem_coalesced.cu",
        nvcc_args=["-DITERATIONS=2000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_gmem_strided",
        program_path=apps / "sb_gmem_strided",
        program_name="gmem_strided.cu",
        nvcc_args=["-DITERATIONS=1750000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_int32_pure",
        program_path=apps / "sb_int32_pure",
        nvcc_args=["-DITERATIONS=56000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_reg_pressure",
        program_path=apps / "sb_reg_pressure",
        program_name="reg_pressure.cu",
        nvcc_args=["-DITERATIONS=10000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sb_smem_pressure",
        program_path=apps / "sb_smem_pressure",
        program_name="smem_pressure.cu",
        nvcc_args=["-DITERATIONS=14000000"],
        force_rebuild=force_rebuild,
    )

    # UB (UBench) Kernels
    # _N=_M=96, blockSideFIL=1024; ITERATIONS loop is inside the kernel (META_LOOP pattern)

    run_kernel_configuration(
        kernel_name="ub_0_1d_store",
        program_path=apps / "ub_0_1d_store",
        nvcc_args=["-DITERATIONS=34000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_1_2d_store",
        program_path=apps / "ub_1_2d_store",
        nvcc_args=["-DITERATIONS=3000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_2_2d_store_single",
        program_path=apps / "ub_2_2d_store_single",
        nvcc_args=["-DITERATIONS=3000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_3_1d_store_empty_loop",
        program_path=apps / "ub_3_1d_store_empty_loop",
        nvcc_args=[],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_4_2d_store_arith",
        program_path=apps / "ub_4_2d_store_arith",
        nvcc_args=["-DITERATIONS=18000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_5_diagonal_store",
        program_path=apps / "ub_5_diagonal_store",
        nvcc_args=["-DITERATIONS=3000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_6_single_thread_store",
        program_path=apps / "ub_6_single_thread_store",
        nvcc_args=["-DITERATIONS=15000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_7_scalar_store_arith",
        program_path=apps / "ub_7_scalar_store_arith",
        nvcc_args=["-DITERATIONS=18000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_8_col_read_reduce",
        program_path=apps / "ub_8_col_read_reduce",
        nvcc_args=["-DITERATIONS=350000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_9_row_read_reduce",
        program_path=apps / "ub_9_row_read_reduce",
        nvcc_args=["-DITERATIONS=350000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_10_col_read_arith",
        program_path=apps / "ub_10_col_read_arith",
        nvcc_args=["-DITERATIONS=18000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_11_row_read_arith",
        program_path=apps / "ub_11_row_read_arith",
        nvcc_args=["-DITERATIONS=18000000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="ub_12_block_read_reduce",
        program_path=apps / "ub_12_block_read_reduce",
        nvcc_args=["-DITERATIONS=35000"],
        force_rebuild=force_rebuild,
    )

    # SGEMM Kernels
    # SIZE_M/N/K = matrix dimensions; REPEAT_TIMES = host-side repeat loop; SIZE_MATRIX = sweep size

    run_kernel_configuration(
        kernel_name="sgemm_1_naive",
        program_path=apps / "sgemm_1_naive",
        nvcc_args=[
            "-DSIZE_M=1024",
            "-DSIZE_N=1024",
            "-DSIZE_K=1024",
            "-DREPEAT_TIMES=1450",
        ],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_1D_blocktiling",
        program_path=apps / "sgemm_1D_blocktiling",
        nvcc_args=["-DREPEAT_TIMES=1500000"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_2_kernel_global_mem_coalesce",
        program_path=apps / "sgemm_2_kernel_global_mem_coalesce",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=1900"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_2D_blocktiling",
        program_path=apps / "sgemm_2D_blocktiling",
        nvcc_args=[
            "-DSIZE_M=256",
            "-DSIZE_N=256",
            "-DSIZE_K=256",
            "-DITERATIONS=10000",
        ],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_3_kernel_shared_mem_blocking",
        program_path=apps / "sgemm_3_kernel_shared_mem_blocking",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=1900"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_4_kernel_1D_blocktiling",
        program_path=apps / "sgemm_4_kernel_1D_blocktiling",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=2500"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_5_kernel_2D_blocktiling",
        program_path=apps / "sgemm_5_kernel_2D_blocktiling",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=270"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_6_kernel_vectorize",
        program_path=apps / "sgemm_6_kernel_vectorize",
        nvcc_args=[
            "-DSIZE_M=1024",
            "-DSIZE_N=1024",
            "-DSIZE_K=1024",
            "-DREPEAT_TIMES=500",
        ],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_7_kernel_resolve_bank_conflicts",
        program_path=apps / "sgemm_7_kernel_resolve_bank_conflicts",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=270"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_8_kernel_bank_extra_col",
        program_path=apps / "sgemm_8_kernel_bank_extra_col",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=270"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_9_kernel_autotuned",
        program_path=apps / "sgemm_9_kernel_autotuned",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=250"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_10_kernel_warptiling",
        program_path=apps / "sgemm_10_kernel_warptiling",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=130"],
        force_rebuild=force_rebuild,
    )

    run_kernel_configuration(
        kernel_name="sgemm_11_kernel_double_buffering",
        program_path=apps / "sgemm_11_kernel_double_buffering",
        nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=100"],
        force_rebuild=force_rebuild,
    )

    # sgemm_12_kernel_double_buffering2 crashes under -O0 -G debug compilation
    # (excessive register pressure from double-buffering tile config K12_BM=128 K12_WM=64)
    # run_kernel_configuration(
    #     kernel_name="sgemm_12_kernel_double_buffering2",
    #     program_path=apps / "sgemm_12_kernel_double_buffering2",
    #     nvcc_args=["-DSIZE_MATRIX=1024", "-DREPEAT_TIMES=100"],
    #     force_rebuild=force_rebuild,
    # )

    # NPB BT Kernels
    # PROBLEM_SIZE sets IMAX=JMAX=KMAX, i.e. the 3-D grid side length
    # 12=class S (small), 24=class W (workstation), 64=class A

    # run_kernel_configuration(
    #     kernel_name="npb_bt_add",
    #     program_path=apps / "npb_bt_add/src",
    #     nvcc_args=["-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_1",
    #     program_path=apps / "npb_bt_compute_rhs_1/src",
    #     nvcc_args=["-DITERATIONS=3000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_2",
    #     program_path=apps / "npb_bt_compute_rhs_2/src",
    #     nvcc_args=["-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_3",
    #     program_path=apps / "npb_bt_compute_rhs_3/src",
    #     nvcc_args=["-DITERATIONS=600000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_4",
    #     program_path=apps / "npb_bt_compute_rhs_4/src",
    #     nvcc_args=["-DITERATIONS=1500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_5",
    #     program_path=apps / "npb_bt_compute_rhs_5/src",
    #     nvcc_args=["-DITERATIONS=1500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_6",
    #     program_path=apps / "npb_bt_compute_rhs_6/src",
    #     nvcc_args=["-DITERATIONS=3000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_7",
    #     program_path=apps / "npb_bt_compute_rhs_7/src",
    #     nvcc_args=["-DITERATIONS=3000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_8",
    #     program_path=apps / "npb_bt_compute_rhs_8/src",
    #     nvcc_args=["-DITERATIONS=3000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_compute_rhs_9",
    #     program_path=apps / "npb_bt_compute_rhs_9/src",
    #     nvcc_args=["-DITERATIONS=9000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_x_solve_1",
    #     program_path=apps / "npb_bt_x_solve_1/src",
    #     nvcc_args=["-DITERATIONS=15000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_x_solve_2",
    #     program_path=apps / "npb_bt_x_solve_2/src",
    #     nvcc_args=["-DITERATIONS=180000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_x_solve_3",
    #     program_path=apps / "npb_bt_x_solve_3/src",
    #     nvcc_args=["-DITERATIONS=12000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_y_solve_1",
    #     program_path=apps / "npb_bt_y_solve_1/src",
    #     nvcc_args=["-DITERATIONS=15000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_y_solve_2",
    #     program_path=apps / "npb_bt_y_solve_2/src",
    #     nvcc_args=["-DITERATIONS=180000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_y_solve_3",
    #     program_path=apps / "npb_bt_y_solve_3/src",
    #     nvcc_args=["-DITERATIONS=12000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_z_solve_1",
    #     program_path=apps / "npb_bt_z_solve_1/src",
    #     nvcc_args=["-DITERATIONS=15000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_z_solve_2",
    #     program_path=apps / "npb_bt_z_solve_2/src",
    #     nvcc_args=["-DITERATIONS=180000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_bt_z_solve_3",
    #     program_path=apps / "npb_bt_z_solve_3/src",
    #     nvcc_args=["-DITERATIONS=12000"],
    #     force_rebuild=force_rebuild,
    # )

    # # NPB EP Kernel
    # # M_EP is the log2 of the total sample count; NN=2^(M-16) threads are launched
    # # 24=class S, 25=class W, 28=class A

    # run_kernel_configuration(
    #     kernel_name="npb_ep_kernel",
    #     program_path=apps / "npb_ep_kernel/src",
    #     nvcc_args=["-DM_EP=24", "-DITERATIONS=30"],
    #     force_rebuild=force_rebuild,
    # )

    # # NPB CG Kernels
    # # NA is the sparse matrix dimension (number of rows/columns)
    # # NZ=NA*(NONZER+1)^2 non-zeros; NONZER=7 is the sparsity parameter (fixed per NPB spec)
    # # 1400=class S, 7000=class W, 14000=class A

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_one",
    #     program_path=apps / "npb_cg_kernel_one/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_two",
    #     program_path=apps / "npb_cg_kernel_two/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_three",
    #     program_path=apps / "npb_cg_kernel_three/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_four",
    #     program_path=apps / "npb_cg_kernel_four/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_five_1",
    #     program_path=apps / "npb_cg_kernel_five_1/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_five_2",
    #     program_path=apps / "npb_cg_kernel_five_2/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_six",
    #     program_path=apps / "npb_cg_kernel_six/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_seven",
    #     program_path=apps / "npb_cg_kernel_seven/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_eight",
    #     program_path=apps / "npb_cg_kernel_eight/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_nine",
    #     program_path=apps / "npb_cg_kernel_nine/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_ten_1",
    #     program_path=apps / "npb_cg_kernel_ten_1/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_ten_2",
    #     program_path=apps / "npb_cg_kernel_ten_2/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=30000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_cg_kernel_eleven",
    #     program_path=apps / "npb_cg_kernel_eleven/src",
    #     nvcc_args=["-DNA=1400", "-DNONZER=7", "-DITERATIONS=60000000"],
    #     force_rebuild=force_rebuild,
    # )

    # # NPB FT Kernels
    # # NX×NY×NZ is the 3-D complex FFT grid
    # # class S=64×64×64, class W=128×128×32, class A=256×256×128

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts1_kernel_1",
    #     program_path=apps / "npb_ft_cffts1_kernel_1/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=1500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts1_kernel_2",
    #     program_path=apps / "npb_ft_cffts1_kernel_2/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=1500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts1_kernel_3",
    #     program_path=apps / "npb_ft_cffts1_kernel_3/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=150000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts2_kernel_1",
    #     program_path=apps / "npb_ft_cffts2_kernel_1/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=4500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts2_kernel_2",
    #     program_path=apps / "npb_ft_cffts2_kernel_2/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=45000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts2_kernel_3",
    #     program_path=apps / "npb_ft_cffts2_kernel_3/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=4500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts3_kernel_1",
    #     program_path=apps / "npb_ft_cffts3_kernel_1/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=4500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts3_kernel_2",
    #     program_path=apps / "npb_ft_cffts3_kernel_2/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=45000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_cffts3_kernel_3",
    #     program_path=apps / "npb_ft_cffts3_kernel_3/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=4500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_checksum_kernel",
    #     program_path=apps / "npb_ft_checksum_kernel/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=3000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_compute_indexmap_kernel",
    #     program_path=apps / "npb_ft_compute_indexmap_kernel/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=3000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_compute_initial_conditions_kernel",
    #     program_path=apps / "npb_ft_compute_initial_conditions_kernel/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=5000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_evolve_kernel",
    #     program_path=apps / "npb_ft_evolve_kernel/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=3000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_ft_init_ui_kernel",
    #     program_path=apps / "npb_ft_init_ui_kernel/src",
    #     nvcc_args=["-DNX=64", "-DNY=64", "-DNZ=64", "-DITERATIONS=3000000"],
    #     force_rebuild=force_rebuild,
    # )

    # # NPB IS Kernels
    # # IS_CLASS encodes the NPB class: 1=S (2^16 keys), 2=W (2^20 keys), 3=A (2^23 keys)
    # # IS sorts integer keys; MAX_KEY = 2^(MAX_KEY_LOG_2) is the key range

    # run_kernel_configuration(
    #     kernel_name="npb_is_create_seq_kernel",
    #     program_path=apps / "npb_is_create_seq_kernel/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=35000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_full_verify_kernel_1",
    #     program_path=apps / "npb_is_full_verify_kernel_1/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=16000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_full_verify_kernel_2",
    #     program_path=apps / "npb_is_full_verify_kernel_2/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=400000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_full_verify_kernel_3",
    #     program_path=apps / "npb_is_full_verify_kernel_3/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=2000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_rank_kernel_1",
    #     program_path=apps / "npb_is_rank_kernel_1/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=6000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_rank_kernel_2",
    #     program_path=apps / "npb_is_rank_kernel_2/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=65000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_rank_kernel_3",
    #     program_path=apps / "npb_is_rank_kernel_3/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=450000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_rank_kernel_4",
    #     program_path=apps / "npb_is_rank_kernel_4/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=2000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_rank_kernel_5",
    #     program_path=apps / "npb_is_rank_kernel_5/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=4000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_rank_kernel_6",
    #     program_path=apps / "npb_is_rank_kernel_6/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=11000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_is_rank_kernel_7",
    #     program_path=apps / "npb_is_rank_kernel_7/src",
    #     nvcc_args=["-DIS_CLASS=1", "-DITERATIONS=7000000"],
    #     force_rebuild=force_rebuild,
    # )

    # # NPB LU Kernels
    # # PROBLEM_SIZE = ISIZ1=ISIZ2=ISIZ3, the cubic grid side length
    # # 12=class S, 33=class W, 64=class A

    # run_kernel_configuration(
    #     kernel_name="npb_lu_erhs_1",
    #     program_path=apps / "npb_lu_erhs_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=700000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_erhs_2",
    #     program_path=apps / "npb_lu_erhs_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=170000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_erhs_3",
    #     program_path=apps / "npb_lu_erhs_3/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=165000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_erhs_4",
    #     program_path=apps / "npb_lu_erhs_4/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=165000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_error_gpu_kernel",
    #     program_path=apps / "npb_lu_error_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=420000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_norm_gpu_kernel",
    #     program_path=apps / "npb_lu_norm_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=530000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_l2norm_gpu_kernel",
    #     program_path=apps / "npb_lu_l2norm_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=760000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_pintgr_1",
    #     program_path=apps / "npb_lu_pintgr_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=800000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_pintgr_2",
    #     program_path=apps / "npb_lu_pintgr_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=800000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_pintgr_3",
    #     program_path=apps / "npb_lu_pintgr_3/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=800000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_pintgr_4",
    #     program_path=apps / "npb_lu_pintgr_4/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=1300000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_rhs_1",
    #     program_path=apps / "npb_lu_rhs_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=2300000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_rhs_2",
    #     program_path=apps / "npb_lu_rhs_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=150000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_rhs_3",
    #     program_path=apps / "npb_lu_rhs_3/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=150000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_rhs_4",
    #     program_path=apps / "npb_lu_rhs_4/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=150000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_setbv_1",
    #     program_path=apps / "npb_lu_setbv_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=360000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_setbv_2",
    #     program_path=apps / "npb_lu_setbv_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=350000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_setbv_3",
    #     program_path=apps / "npb_lu_setbv_3/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=350000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_setiv_gpu_kernel",
    #     program_path=apps / "npb_lu_setiv_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=140000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_jacld_blts_gpu_kernel",
    #     program_path=apps / "npb_lu_jacld_blts_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=120000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_jacu_buts_gpu_kernel",
    #     program_path=apps / "npb_lu_jacu_buts_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=120000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_ssor_1",
    #     program_path=apps / "npb_lu_ssor_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=7500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_lu_ssor_2",
    #     program_path=apps / "npb_lu_ssor_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=6000000"],
    #     force_rebuild=force_rebuild,
    # )

    # # NPB MG Kernels
    # # MG_PROBLEM_SIZE = finest-level cubic grid side length (power of 2)
    # # 32=class S, 128=class W, 256=class A

    # run_kernel_configuration(
    #     kernel_name="npb_mg_comm3_kernel_1",
    #     program_path=apps / "npb_mg_comm3_kernel_1/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=16000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_mg_comm3_kernel_2",
    #     program_path=apps / "npb_mg_comm3_kernel_2/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=22000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_mg_comm3_kernel_3",
    #     program_path=apps / "npb_mg_comm3_kernel_3/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=22000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_mg_interp_gpu_kernel",
    #     program_path=apps / "npb_mg_interp_gpu_kernel/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=2500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_mg_norm2u3_gpu_kernel",
    #     program_path=apps / "npb_mg_norm2u3_gpu_kernel/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=1800000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_mg_psinv_gpu_kernel",
    #     program_path=apps / "npb_mg_psinv_gpu_kernel/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=2400000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_mg_resid_gpu_kernel",
    #     program_path=apps / "npb_mg_resid_gpu_kernel/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=2500000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_mg_rprj3_gpu_kernel",
    #     program_path=apps / "npb_mg_rprj3_gpu_kernel/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=2800000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_mg_zero3_gpu_kernel",
    #     program_path=apps / "npb_mg_zero3_gpu_kernel/src",
    #     nvcc_args=["-DMG_PROBLEM_SIZE=32", "-DITERATIONS=65000000"],
    #     force_rebuild=force_rebuild,
    # )

    # # NPB SP Kernels
    # # PROBLEM_SIZE = NX=NY=NZ; 12=class S, 36=class W, 64=class A

    # run_kernel_configuration(
    #     kernel_name="npb_sp_add_gpu_kernel",
    #     program_path=apps / "npb_sp_add_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=4000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_compute_rhs_gpu_kernel_1",
    #     program_path=apps / "npb_sp_compute_rhs_gpu_kernel_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=2100000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_compute_rhs_gpu_kernel_2",
    #     program_path=apps / "npb_sp_compute_rhs_gpu_kernel_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=160000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_error_norm_gpu_kernel_1",
    #     program_path=apps / "npb_sp_error_norm_gpu_kernel_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=75000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_error_norm_gpu_kernel_2",
    #     program_path=apps / "npb_sp_error_norm_gpu_kernel_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=460000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_exact_rhs_gpu_kernel_1",
    #     program_path=apps / "npb_sp_exact_rhs_gpu_kernel_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=6000000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_exact_rhs_gpu_kernel_2",
    #     program_path=apps / "npb_sp_exact_rhs_gpu_kernel_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=32000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_exact_rhs_gpu_kernel_3",
    #     program_path=apps / "npb_sp_exact_rhs_gpu_kernel_3/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=32000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_exact_rhs_gpu_kernel_4",
    #     program_path=apps / "npb_sp_exact_rhs_gpu_kernel_4/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=30000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_initialize_gpu_kernel",
    #     program_path=apps / "npb_sp_initialize_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=100000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_rhs_norm_gpu_kernel_1",
    #     program_path=apps / "npb_sp_rhs_norm_gpu_kernel_1/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=460000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_rhs_norm_gpu_kernel_2",
    #     program_path=apps / "npb_sp_rhs_norm_gpu_kernel_2/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=460000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_txinvr_gpu_kernel",
    #     program_path=apps / "npb_sp_txinvr_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=2300000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_x_solve_gpu_kernel",
    #     program_path=apps / "npb_sp_x_solve_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=25000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_y_solve_gpu_kernel",
    #     program_path=apps / "npb_sp_y_solve_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=25000"],
    #     force_rebuild=force_rebuild,
    # )

    # run_kernel_configuration(
    #     kernel_name="npb_sp_z_solve_gpu_kernel",
    #     program_path=apps / "npb_sp_z_solve_gpu_kernel/src",
    #     nvcc_args=["-DPROBLEM_SIZE=12", "-DITERATIONS=24000"],
    #     force_rebuild=force_rebuild,
    # )

    if short_kernels:
        print("\n[SUMMARY] Kernels below threshold:")
        for name, dur in short_kernels:
            print(f"  {name}: {dur:.3f}s")
    else:
        print("\n[SUMMARY] All kernels met the threshold.")

    if analyzed_kernels:
        print("\n[SUMMARY] Re-ran Analyzer kernels:")
        for kernel_name, execution_result in analyzed_kernels:
            print(f"  {kernel_name}, Path: {execution_result.path}")
    else:
        print("\n[SUMMARY] No kernels were re-run with the Analyzer.")


if __name__ == "__main__":
    main()
