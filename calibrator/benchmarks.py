from dataclasses import dataclass
from pathlib import Path
from cubindings import execute_program, extract_power_metrics, extract_exports_from_path
from analyser import run_analyser, RunAnalyserResult
from power import PowerMetricsResult 
import csv
from hashlib import md5

benchmark_prefix = "benchmark_" # Only change this if you want to invalidate the cache 
weights_path = Path("/home/p10/aau-p10-ptx-energy/linear-model/weights.csv")
artifacts_path = Path("/home/p10/aau-p10-ptx-energy/experiments/artifacts")
debug_enabled = True
cache_enabled = False

csv_results = []

@dataclass
class CachedProgramResult:
    path: Path
    power_metric_result: PowerMetricsResult

@dataclass
class CSVResult:
    kernel_name: str
    block_dim: int
    grid_dim: int
    actual_power_joules: float
    predicted_power_joules: float
    kernel_duration_cpu_s: float = 0.0
    
def concat_results(kernel_name: str, power_metric_result: PowerMetricsResult , analysis_result: RunAnalyserResult) -> CSVResult:
    actual_power_joules = power_metric_result.total_energy_j
    predicted_power_joules = analysis_result.predicted_energy_joules
    kernel_duration_cpu_s = power_metric_result.kernel_duration_cpu_s
    # instruction_occurrences = analysis_result.linear_model_output.estimates TODO: Maybe add this to CSV output
    grid_dim = analysis_result.analyser_result.grid_dim
    block_dim = analysis_result.analyser_result.block_dim

    print (f"Kernel: {kernel_name}, Grid Dim: {grid_dim}, Block Dim: {block_dim}")
    print(f"Predicted Power (Joules): {predicted_power_joules}")
    print(f"Actual Power (Joules): {actual_power_joules}")
    print(f"Error (%): {((1 - (predicted_power_joules/actual_power_joules)) * 100):.2f}")

    total_instructions = sum(estimate.count for estimate in analysis_result.linear_model_output.estimates)
    print(f"Total Instructions: {total_instructions}")

    for estimate in analysis_result.linear_model_output.estimates:
        print(f"  {estimate.instruction}: {estimate.count} occurrences, {(estimate.count / total_instructions * 100):.2f}%, {estimate.estimated_power_joules} Joules")
    
    csv_results.append(CSVResult(
        kernel_name=kernel_name,
        block_dim=block_dim,   
        grid_dim=grid_dim,
        actual_power_joules=actual_power_joules,
        predicted_power_joules=predicted_power_joules,
        kernel_duration_cpu_s=kernel_duration_cpu_s
    ))

def execute_program_cached(path: Path, program_name: str, nvcc_args: list[str] = []) -> CachedProgramResult:
    # Placeholder for the power metric result
    power_metric_result = None

    # Hash the nvcc arguments to create a unique identifier
    file_name = md5(f"{path}{''.join(nvcc_args)}".encode()).hexdigest()

    # Check if their exists an artifact of the program
    folder_name = f"{benchmark_prefix}{file_name}"
    program_artifact = artifacts_path / folder_name
    
    if cache_enabled and (program_artifact / "output.json").exists():
        print("[INFO] Found cached artifact for program, skipping execution.")

        exports = extract_exports_from_path(program_artifact)
        power_metric_result = extract_power_metrics(path=program_artifact, exports=exports)
    else:
        # Copy the benchmark to artifacts
        program_artifact.mkdir(parents=True, exist_ok=True)
        for item in path.iterdir():
            if item.is_file():
                destination = program_artifact / item.name
                if not destination.exists():
                    destination.write_bytes(item.read_bytes())
    
        print(f"[INFO] Executed program at {program_artifact} with nvcc args: {nvcc_args}")

        execution_result = execute_program(path=program_artifact, program_name=program_name, nvcc_args=nvcc_args, enable_temp=False, debug=debug_enabled)
        power_metric_result = execution_result.power_metric_result

        print(power_metric_result)


    return CachedProgramResult(path=program_artifact, power_metric_result=power_metric_result)

# vector_add_old
# problem_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
# for size in problem_sizes:

#     nvcc_args = [f"-DSIZE_N={size}"]
#     program_name="main.cu"
#     program_path = Path("/home/rasmus/aau-p10-ptx-energy/experiments/apps/vector_add_old/src") 
#     cached_program_result = execute_program_cached(path=program_path, program_name=program_name, nvcc_args=nvcc_args)
#     analysis_result = run_analyser(cached_program_result.path, weights_path, debug=debug_enabled, nvcc_args=nvcc_args, program_name=program_name)

#     concat_results("vector_add_old", cached_program_result.power_metric_result, analysis_result)

# matrix_mult
# problem_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
# for size in problem_sizes:
#     nvcc_args = [f"-DSIZE_M={size}", f"-DSIZE_N=32", f"-DSIZE_K=32"] # NOTE: Do not increase SIZE_N and SIZE_K as this will increase the runtime significantly
#     program_name = "main.cu"
#     program_path = Path("/home/rasmus/aau-p10-ptx-energy/experiments/apps/matrix_mult/src") 
#     cached_program_result = execute_program_cached(path=program_path, program_name=program_name, nvcc_args=nvcc_args)
#     analysis_result = run_analyser(cached_program_result.path, weights_path, debug=debug_enabled, nvcc_args=nvcc_args, program_name=program_name)

#     concat_results("matrix_mult", cached_program_result.power_metric_result, analysis_result)

size = 32
nvcc_args = [f"-DSIZE_M={size}", f"-DSIZE_N=32", f"-DSIZE_K=32"] # NOTE: Do not increase SIZE_N and SIZE_K as this will increase the runtime significantly
program_name = "main.cu"
program_path = Path("/home/rasmus/aau-p10-ptx-energy/experiments/apps/matrix_mult/src") 
cached_program_result = execute_program_cached(path=program_path, program_name=program_name, nvcc_args=nvcc_args)
# analysis_result = run_analyser(cached_program_result.path, weights_path, debug=debug_enabled, nvcc_args=nvcc_args, program_name=program_name)

# concat_results("matrix_mult", cached_program_result.power_metric_result, analysis_result)


# Write the CSV
with open('benchmark_results.csv', mode='w', newline='') as csv_file:
    fieldnames = ['kernel_name', 'block_dim', 'grid_dim', 'actual_power_joules', 'predicted_power_joules', 'kernel_duration_cpu_s']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for result in csv_results:
        writer.writerow({
            'kernel_name': result.kernel_name,
            'block_dim': result.block_dim,
            'grid_dim': result.grid_dim,
            'actual_power_joules': result.actual_power_joules,
            'predicted_power_joules': result.predicted_power_joules,
            'kernel_duration_cpu_s': result.kernel_duration_cpu_s
        })
