import subprocess
import json
import sys
from pathlib import Path
import shutil
from dataclasses import dataclass
import time
import re
import shlex

@dataclass
class Dim3:
    x: int
    y: int
    z: int

    @staticmethod
    def from_dict(data: dict) -> "Dim3":
        return Dim3(
            x=int(data.get("x", 1)),
            y=int(data.get("y", 1)),
            z=int(data.get("z", 1)),
        )


@dataclass
class AnalyserResult:
    kernel_name: str | None
    power_consumption_joules: float | None
    grid_dim: Dim3
    block_dim: Dim3
    parameters: list
    total_instructions: int
    instruction_occurrences: dict[str, int]

    @staticmethod
    def from_dict(data: dict) -> "AnalyserResult":
        power = data.get("powerConsumptionJoules")
        return AnalyserResult(
            kernel_name=data.get("kernelName"),
            power_consumption_joules=(float(power) if power is not None else None),
            grid_dim=Dim3.from_dict(data.get("gridDim", {})),
            block_dim=Dim3.from_dict(data.get("blockDim", {})),
            parameters=data.get("parameters", []),
            total_instructions=int(data.get("totalInstructions", 0)),
            instruction_occurrences={
                str(op): int(count)
                for op, count in data.get("instructionOccurrences", {}).items()
            },
        )

@dataclass
class LinearModelResult:
    analyser_result: AnalyserResult
    predicted_energy_joules: float

def run_analyser(
    output_path: Path,
    weights_path: Path,
    nvcc_args: list[str] = [],
    prediction: bool = True,
    program_name="program.cu",
    binary_args: list[str] = [],
    debug: bool = False
) -> LinearModelResult:    
    pipe = sys.stdout if debug else subprocess.PIPE

    analyser_result = None
    total_energy_j = -1
    
    if shutil.which("injector") is not None:
        try:
            include_path = Path(__file__).parents[1] / "include"
            lli_args = " ".join(shlex.quote(str(arg)) for arg in binary_args)
            lli_suffix = f" {lli_args}" if lli_args else ""

            cmd = [f"cat {output_path / program_name} | injector > {output_path / 'injected_kernel.cu'}; clang++ -DUSE_LLI -S -emit-llvm --cuda-host-only -I{str(include_path)} {" ".join(shlex.quote(str(arg)) for arg in nvcc_args)} {output_path / 'injected_kernel.cu'} --no-cuda-version-check; lli {output_path / 'injected_kernel.ll'}{lli_suffix}"]
            subprocess.run(
                cmd,
                stdout=pipe,
                stderr=pipe,
                cwd=output_path,
                shell=True,
                check=False,
                text=True
            )


        except Exception as e:
            print(f"[Warning] Failed to run injector: {e}", file=sys.stderr)
    else:
        print("[Warning] injector not found in PATH, skipping kernel injection", file=sys.stderr)
    
    if shutil.which("ptx-analyser") is not None:
        try:
            # Run analyser and print full output so failures are visible.
            analyser_process_result = subprocess.run(
                [
                    "ptx-analyser",
                    "analyze-cfg",
                    "--kernel-params-file",
                    str(output_path / "kernel_params.json"),
                    "--output-json-path",
                    str(output_path / "analyser_output.json"),
                    str(output_path / program_name.replace(".cu", ".ptx")),
                ],
                stderr=pipe,
                stdout=pipe,
                text=True,
                check=False,
                cwd=output_path,
            )

            time.sleep(1) # Sleep briefly to ensure the output file is written before we try to read it

            # Load the analyser result from the output JSON file
            analyser_output_json = output_path / "analyser_output.json"
            if analyser_output_json.exists():
                with analyser_output_json.open("r") as f:
                    analyser_result = AnalyserResult.from_dict(json.loads(f.read()))

        except Exception as e:
            print(f"[Warning] Failed to start ptx-analyser: {e}", file=sys.stderr)
    else:
        print("[Warning] ptx-analyser not found in PATH", file=sys.stderr)

    if prediction:
        parent_dir = Path(__file__).parents[1]
        [f"uv run linear-model/linear-model.py --weights-path {weights_path} --input-path {output_path / 'analyser_output.json'}"],
        try:
            linear_model_result = subprocess.run(
                stdout=pipe,
                stderr=pipe,
                cwd=parent_dir,
                shell=True,
                check=False,
                text=True
            )
        
            total_energy_match = re.search(r"\[TOTAL\]\s+([\d\.eE+-]+)", linear_model_result.stdout)
            if total_energy_match:
                total_energy_j = float(total_energy_match.group(1))
            else:
                print("Warning: Could not find total energy in linear model output", file=sys.stderr)

        except Exception as e:
            print(f"[Warning] Failed to run linear model: {e}", file=sys.stderr)

    return LinearModelResult(analyser_result=analyser_result, predicted_energy_joules=total_energy_j)