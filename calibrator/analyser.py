
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

