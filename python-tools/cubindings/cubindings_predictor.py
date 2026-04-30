
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import subprocess
import re
import json


@dataclass
class LinearModelEstimate:
    kernel_name: str
    instruction: str
    count: float
    avg_power_per_occurrence_joules: float
    estimated_power_joules: float
    used_fallback: bool

    @staticmethod
    def from_dict(data: dict) -> "LinearModelEstimate":
        return LinearModelEstimate(
            kernel_name=str(data.get("kernel_name", "")),
            instruction=str(data.get("instruction", "")),
            count=float(data.get("count", 0.0)),
            avg_power_per_occurrence_joules=float(data.get("avg_power_per_occurrence_joules", 0.0)),
            estimated_power_joules=float(data.get("estimated_power_joules", 0.0)),
            used_fallback=bool(data.get("used_fallback", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LinearModelOutput:
    weights_path: str
    input_path: str
    output_path: str
    fallback_power_joules: float
    total_estimated_power_joules: float
    estimates: list[LinearModelEstimate]

    @staticmethod
    def from_dict(data: dict) -> "LinearModelOutput":
        return LinearModelOutput(
            weights_path=str(data.get("weights_path", "")),
            input_path=str(data.get("input_path", "")),
            output_path=str(data.get("output_path", "")),
            fallback_power_joules=float(data.get("fallback_power_joules", 0.0)),
            total_estimated_power_joules=float(data.get("total_estimated_power_joules", 0.0)),
            estimates=[LinearModelEstimate.from_dict(item) for item in data.get("estimates", []) if isinstance(item, dict)],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "weights_path": self.weights_path,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "fallback_power_joules": self.fallback_power_joules,
            "total_estimated_power_joules": self.total_estimated_power_joules,
            "estimates": [estimate.to_dict() for estimate in self.estimates],
        }

def run_predictor(model_path: Path, output_path: Path, weights_path: Path, debug_enabled: bool = False) -> LinearModelOutput:
        pipe = sys.stdout if debug_enabled else subprocess.PIPE
        
        parent_dir = Path(__file__).parents[1]
        linear_model_output_path = output_path / "linear_model_output.json"

        cmd = [
            (
                f"uv run {model_path} "
                f"--weights-path {weights_path} "
                f"--input-path {output_path / 'analyser_output.json'} "
                f"--output-path {linear_model_output_path}"
            )
        ]
        try:
            linear_model_result = subprocess.run(
                cmd,
                stdout=pipe,
                stderr=pipe,
                cwd=parent_dir,
                shell=True,
                check=False,
                text=True
            )
        
            if linear_model_output_path.exists():
                with linear_model_output_path.open("r", encoding="utf-8") as handle:
                    linear_model_output = LinearModelOutput.from_dict(json.loads(handle.read()))
                    return linear_model_output

        except Exception as e:
            print(f"[Warning] Failed to run linear model: {e}", file=sys.stderr)
