import subprocess
import os
import signal
import time
import sys
from pathlib import Path
import shutil
import re
from dataclasses import dataclass
from power import extract_power_metrics, PowerMetricsResult
import json

@dataclass
class ExecutionResult:
    output: str
    error: str
    exports: dict[str, str]
    returncode: int
    path: Path
    power_metric_result: PowerMetricsResult = None


def _terminate_process_group(process: subprocess.Popen | None) -> None:
    if process is None:
        return

    if process.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()
    
def executeProgram(
    program_str: str,
    path: Path = None,
    nvcc_args: list[str] = [], binary_args: list[str] = [],
    enable_metrics: bool = False,
    metrics_sleep_time: int = 5
) -> ExecutionResult:   
    # Initialize power_metric_result to None by default
    power_metric_result = None

    # Fallback temporary directory if the specified path cannot be created
    if path is None:
        path = Path(f"/tmp/{time.time()}")
        path.mkdir(parents=True, exist_ok=True)

    # Append the metrics code to the program string if metrics are enabled
    if enable_metrics:
        pattern = r'(int\s+main\s*\([^)]*\)\s*\{)(.*?)(return\s+0\s*;\s*\})'
        program_str = re.sub(
            pattern,
            r'\1\nMETRICS_KERNEL_START\n\2\nMETRICS_KERNEL_END\n\3',
            program_str,
            flags=re.DOTALL
        )

        program_str = "#include \"cupti_timing.h\"\n" + program_str

    # Write the program string to a temporary file
    program_file = path / "program.cu"
    with program_file.open("w") as f:
        f.write(program_str)

    # Absolute path to the include directory for cubindings.h
    nvcc_cmd = ["nvcc", "-Xptxas", "-g", "-G", "-O0", "-arch=sm_89", "-lcupti"]
    include_path = Path(__file__).parents[1] / "include"
    nvcc_cmd.append(f"-I{str(include_path)}")
    nvcc_cmd.extend(list(nvcc_args))
    nvcc_cmd.append(str(program_file))

    nvcc_process = subprocess.run(
        nvcc_cmd,
        cwd=path, # Set the current working directory to the temporary directory
        text=True,
        check=False,
    )
    
    # Start monitoring processes if metrics are enabled
    monitor_process = None
    pmd2_process = None
    if enable_metrics:
        # Start nvidia-smi monitoring on a separate process
        monitor_log = path / "nvidia-smi.csv"
        monitor_process = None
        
        # Start pmd2-cli monitoring on a separate process
        pmd2_log = path / "pmd2.csv"
        pmd2_process = None
        if shutil.which("nvidia-smi") is not None:
            try:
                with open(monitor_log, "w") as monitor_file:
                    monitor_process = subprocess.Popen(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,power.draw,power.limit,temperature.gpu,fan.speed",
                            "--format=csv",
                            "-lms",
                            "5",
                        ],
                        stdout=monitor_file,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
            except Exception as e:
                print(f"Warning: Failed to start nvidia-smi monitoring: {e}", file=sys.stderr)
                monitor_process = None
        
        if shutil.which("pmd2-cli") is not None:
            try:
                with open(pmd2_log, "w") as pmd2_file:
                    pmd2_process = subprocess.Popen(
                        [
                            "pmd2-cli",
                            "-p",
                            "/dev/ttyACM0",
                            "-c",
                            "-i",
                            "100",
                            "monitor"
                        ],
                        stdout=pmd2_file,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
            except Exception as e:
                print(f"Warning: Failed to start pmd2-cli monitoring: {e}", file=sys.stderr)
                pmd2_process = None

            # Sleep for the specified duration to ensure subprocesses have time to collect data
            time.sleep(metrics_sleep_time)
    
    bin_cmd = [str(path / "a.out")]
    bin_cmd.extend(binary_args)

    execution_process = None
    exports = {}

    try:
        # Run the compiled program and capture its output
        execution_process = subprocess.run(
            bin_cmd,
            cwd=path, # Set the current working directory to the temporary directory
            text=True,
            check=False,
            stderr=subprocess.STDOUT,
        )

        # Read the output.json file to get the exported variables
        output_json = path / "output.json"
        if output_json.exists():
            with output_json.open("r") as f:
                exports = json.load(f)
    finally:
        if enable_metrics:
            _terminate_process_group(monitor_process)
            _terminate_process_group(pmd2_process)
            power_metric_result = extract_power_metrics(path, exports)


    # Return the execution result
    return ExecutionResult(
        output=(execution_process.stdout if execution_process is not None else ""),
        error=(execution_process.stderr if execution_process is not None else ""),
        exports=exports,
        returncode=(execution_process.returncode if execution_process is not None else -1),
        power_metric_result=power_metric_result,
        path=path
    )