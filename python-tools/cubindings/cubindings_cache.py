from pathlib import Path
from .cubindings import extract_exports_from_path, extract_power_metrics, execute_program, ExecutionResult
from hashlib import md5

def execute_program_cached(
    path: Path,
    program_name: str,
    nvcc_args: list[str] = [],
    cache_key: str = "madsen",
    artifacts_path: Path = Path("/home/p10/aau-p10-ptx-energy/experiments/artifacts"),
    debug_enabled: bool = False,
    force_rebuild: bool = False
) -> ExecutionResult:
    # Hash the nvcc arguments to create a unique identifier
    file_name = md5(f"{path}{''.join(nvcc_args)}".encode()).hexdigest()

    # Check if their exists an artifact of the program
    folder_name = f"{cache_key}{file_name}"
    program_artifact = artifacts_path / folder_name
    
    if not force_rebuild and (program_artifact / "output.json").exists():
        if debug_enabled:
            print("[INFO] Found cached artifact for program, skipping execution.")

        exports = extract_exports_from_path(program_artifact)
        power_metric_result = extract_power_metrics(path=program_artifact, exports=exports)

        return ExecutionResult(
            output="",
            error="",
            exports=exports,
            returncode=0,
            path=program_artifact,
            power_metric_result=power_metric_result
        )
    else:
        # Clear the artifact folder if it exists to ensure a clean state
        if program_artifact.exists():
            for item in program_artifact.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    # Recursively delete directories if needed
                    for sub_item in item.iterdir():
                        if sub_item.is_file():
                            sub_item.unlink()
                        elif sub_item.is_dir():
                            # Handle nested directories if necessary
                            pass
                    item.rmdir()
            program_artifact.rmdir()

        # Copy the benchmark to artifacts
        program_artifact.mkdir(parents=True, exist_ok=True)
        for item in path.iterdir():
            if item.is_file():
                destination = program_artifact / item.name
                if not destination.exists():
                    destination.write_bytes(item.read_bytes())
    
        execution_result = execute_program(path=program_artifact, program_name=program_name, nvcc_args=nvcc_args, enable_temp=False, debug_enabled=debug_enabled)
        return execution_result        
