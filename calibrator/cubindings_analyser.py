import subprocess
import json
import sys
from pathlib import Path
import shutil
from dataclasses import dataclass
import time

from calibrator.cubindings_types import ExportJSONResponse

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

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass
class AnalyserResult:
    kernel_name: str | None
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
            grid_dim=Dim3.from_dict(data.get("gridDim", {})),
            block_dim=Dim3.from_dict(data.get("blockDim", {})),
            parameters=data.get("parameters", []),
            total_instructions=int(data.get("totalInstructions", 0)),
            instruction_occurrences={
                str(op): int(count)
                for op, count in data.get("instructionOccurrences", {}).items()
            },
        )

    def to_dict(self) -> dict:
        return {
            "kernelName": self.kernel_name,
            "gridDim": self.grid_dim.to_dict(),
            "blockDim": self.block_dim.to_dict(),
            "parameters": self.parameters,
            "totalInstructions": self.total_instructions,
            "instructionOccurrences": self.instruction_occurrences,
        }

def build_kernel_params(exports: ExportJSONResponse, parameters: list[dict[str, object]] = []) -> str:
    return json.dumps(
        {
            "gridDim": {
                "x": exports["gridDim_x"],
                "y": exports["gridDim_y"],
                "z": exports["gridDim_z"],
            },
            "blockDim": {
                "x": exports["blockDim_x"],
                "y": exports["blockDim_y"],
                "z": exports["blockDim_z"],
            },
            "parameters": parameters,
        }
    )

def run_ptx_analyser(
    output_path: Path,
    kernel_params: str,
    program_name="program.cu",
    debug_enabled: bool = False
) -> AnalyserResult:
    pipe = sys.stdout if debug_enabled else subprocess.PIPE

    analyser_result = None
    
    if shutil.which("ptx-analyser") is not None:
        try:
            cmd = [
                    "ptx-analyser",
                    "analyze-cfg",
                    "--kernel-params",
                    kernel_params,
                    "--output-json-path",
                    str(output_path / "analyser_output.json"),
                    str(output_path / program_name.replace(".cu", ".ptx")),
                ]
            
            # Run analyser and print full output so failures are visible.
            subprocess.run(
                cmd,
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
                    return analyser_result
            else:
                print(f"[Warning]: Analyser output file not found at {analyser_output_json}", file=sys.stderr)

        except Exception as e:
            print(f"[Warning] Failed to start ptx-analyser: {e}", file=sys.stderr)
    else:
        print("[Warning] ptx-analyser not found in PATH", file=sys.stderr)