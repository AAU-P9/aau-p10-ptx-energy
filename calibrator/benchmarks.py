from pathlib import Path
from cubindings import execute_program
from analyser import run_analyser

weights_path = Path("/home/p10/aau-p10-ptx-energy/linear-model/weights.csv")
debug = False

# vector_add_old
problem_sizes = [16384, 32768, 65536, 131072, 262144, 524288, 1048576]
for size in problem_sizes:
    grid_dim = size / 1024
    block_dim = 1024 # 32x32 threads per block

    nvcc_args = [f"-DSIZE_N={size}"]
    program_name="main.cu"
    program_path = Path("/home/rasmus/aau-p10-ptx-energy/experiments/apps/vector_add_old/src") 
    execution_result = execute_program(path=program_path, program_name=program_name, nvcc_args=nvcc_args, debug=debug)
    analysis_result = run_analyser(execution_result.path, weights_path, debug=debug, nvcc_args=nvcc_args, program_name=program_name)

    actual_power_joules = execution_result.power_metric_result.total_energy_j
    predicted_power_joules = analysis_result.predicted_energy_joules
    kernel_duration_cpu_s = execution_result.power_metric_result.kernel_duration_cpu_s
    instruction_occurrences = analysis_result.linear_model_output.estimates

    print(f"Predicted Power (Joules): {predicted_power_joules}")
    print(f"Actual Power (Joules): {actual_power_joules}")
    sum = 0.0
    print(f"Error (%): {(((predicted_power_joules/actual_power_joules) - 1) * 100):.2f}")
    for estimate in analysis_result.linear_model_output.estimates:
        print(f"  {estimate.instruction}: {estimate.count} occurrences, {estimate.estimated_power_joules} Joules")
        sum += estimate.estimated_power_joules

    print(f"Total Estimated Power (Joules): {sum}")
    print("-" * 40)

# matrix_mult
# problem_sizes = [256, 512, 1024]
# for size in problem_sizes:
#     grid_dim = (size // 32) * (256 // 32)
#     block_dim = 1024 # 32x32 threads per block

#     nvcc_args = [f"-DSIZE_M={size}", f"-DSIZE_N={size}", f"-DSIZE_K={size}"]
#     program_name="main.cu"
#     program_path = Path("/home/rasmus/aau-p10-ptx-energy/experiments/apps/matrix_mult/src") 
#     execution_result = execute_program(path=program_path, program_name=program_name, nvcc_args=nvcc_args, debug=debug)
#     analysis_result = run_analyser(execution_result.path, weights_path, debug=debug, nvcc_args=nvcc_args, program_name=program_name)

#     actual_power_joules = execution_result.power_metric_result.total_energy_j
#     predicted_power_joules = analysis_result.predicted_energy_joules
#     kernel_duration_cpu_s = execution_result.power_metric_result.kernel_duration_cpu_s
#     instruction_occurrences = analysis_result.analyser_result.instruction_occurrences

#     print(f"Predicted Power (Joules): {predicted_power_joules}")
#     print(f"Actual Power (Joules): {actual_power_joules}")
#     print(f"Error (%): {((actual_power_joules - predicted_power_joules) / actual_power_joules * 100):.2f}")
#     print("-" * 40)