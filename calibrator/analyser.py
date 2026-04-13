import subprocess
import json
import sys
from pathlib import Path
import shutil
from dataclasses import dataclass

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

def run_analyser(path: Path) -> AnalyserResult | None:    
    if shutil.which("injector") is not None:
        try:
            include_path = Path(__file__).parents[1] / "include"

            print(include_path)

            subprocess.run(
                [f"cat {path / 'program.cu'} | injector > {path / 'injected_kernel.cu'} && clang++ -DUSE_LLI -S -emit-llvm --cuda-host-only -I{str(include_path)} {path / 'injected_kernel.cu'} --no-cuda-version-check && lli {path / 'injected_kernel.ll'}"],
                stdout=sys.stdout,
                stderr=sys.stdout,
                cwd=path,
                shell=True,
                check=False,
                text=True
            )
        except Exception as e:
            print(f"[Warning] Failed to run injector: {e}", file=sys.stderr)
    else:
        print("[Warning] injector not found in PATH, skipping kernel injection", file=sys.stderr)
    
    try:
        with (path / "analyser_output.json").open("r") as f:
            return AnalyserResult.from_dict(json.load(f))
    except Exception as e:
        print(f"[Warning] Failed to read analyser output: {e}", file=sys.stderr)
        return None

    if shutil.which("ptx-analyser") is not None:
        try:
            # Run analyser and print full output so failures are visible.
            analyser_process_result = subprocess.run(
                [
                    "ptx-analyser",
                    "analyze-cfg",
                    "--kernel-params",
                    analyser_kernel_params,
                    "--output-json-path",
                    str(path / "analyser_output.json"),
                    str(path / "program.ptx")
                ],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                check=False,
                cwd=path,
            )

            time.sleep(1) # Sleep briefly to ensure the output file is written before we try to read it

            # Load the analyser result from the output JSON file
            analyser_output_json = path / "analyser_output.json"
            if analyser_output_json.exists():
                with analyser_output_json.open("r") as f:
                    analyser_result = AnalyserResult.from_dict(json.loads(f.read()))

        except Exception as e:
            print(f"[Warning] Failed to start ptx-analyser: {e}", file=sys.stderr)
    else:
        print("[Warning] ptx-analyser not found in PATH", file=sys.stderr)