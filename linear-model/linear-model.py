#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path


def load_instruction_power_map(weights_path: Path) -> dict[str, float]:
	"""Build average per-occurrence power (joules) for each instruction."""
	samples: dict[str, list[float]] = defaultdict(list)

	with weights_path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle, skipinitialspace=True)
		for row in reader:
			instruction = (row.get("instruction") or "").strip()
			count_raw = row.get("count")
			power_raw = row.get("power_consumption_joules")

			if not instruction or count_raw is None or power_raw is None:
				continue

			try:
				count = float(count_raw)
				power_joules = float(power_raw)
			except ValueError:
				continue

			if count <= 0:
				continue

			per_occurrence_power = power_joules / count
			samples[instruction].append(per_occurrence_power)

	return {
		instruction: sum(values) / len(values)
		for instruction, values in samples.items()
		if values
	}


def estimate_input_instruction_power(
	input_path: Path,
	instruction_power_map: dict[str, float],
) -> list[dict[str, str | float]]:
	estimates: list[dict[str, str | float]] = []

	with input_path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle, skipinitialspace=True)
		for row in reader:
			kernel_name = (row.get("kernel_name") or "").strip()
			instruction = (row.get("instruction") or "").strip()
			count_raw = row.get("count")

			if not instruction or count_raw is None:
				continue

			try:
				count = float(count_raw)
			except ValueError:
				continue

			avg_per_occurrence = instruction_power_map.get(instruction, 0.0)
			estimated_power_joules = count * avg_per_occurrence

			estimates.append(
				{
					"kernel_name": kernel_name,
					"instruction": instruction,
					"count": count,
					"avg_power_per_occurrence_joules": avg_per_occurrence,
					"estimated_power_joules": estimated_power_joules,
				}
			)

	return estimates


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	weights_path = base_dir / "weights.csv"
	input_path = base_dir / "input.csv"

	instruction_power_map = load_instruction_power_map(weights_path)

	print("Instruction average power map (joules per occurrence):")
	for instruction in sorted(instruction_power_map):
		print(f"{instruction},{instruction_power_map[instruction]:.12e}")

	print("\nInput estimates:")
	estimates = estimate_input_instruction_power(input_path, instruction_power_map)
	total_estimated = 0.0
	for entry in estimates:
		total_estimated += float(entry["estimated_power_joules"])
		print(
			f"{entry['kernel_name']},{entry['instruction']},"
			f"count={entry['count']:.0f},"
			f"avg={entry['avg_power_per_occurrence_joules']:.12e},"
			f"estimated={entry['estimated_power_joules']:.12e}"
		)

	print(f"\nTotal estimated power for input rows: {total_estimated:.12e} J")


if __name__ == "__main__":
	main()
