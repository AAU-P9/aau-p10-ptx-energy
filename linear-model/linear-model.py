#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path
from decimal import Decimal

def load_instruction_power_map(weights_path: Path) -> dict[str, float]:
	"""Build average per-occurrence power (joules) for each instruction."""
	samples: dict[str, list[float]] = defaultdict(list)

	_sum = 0.0

	with weights_path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle, skipinitialspace=True)
		for row in reader:
			instruction = (row.get("instruction") or "").strip()
			count_raw = row.get("count")
			total_instructions = row.get("total_instructions")
			power_raw = row.get("power_consumption_joules")

			if not instruction or count_raw is None or power_raw is None:
				continue

			try:
				count = float(count_raw)
				total_instructions = float(total_instructions)
				power_joules = float(power_raw)
			except ValueError:
				continue

			if count <= 0:
				continue


			per_occurrence_power = ((count / total_instructions) * power_joules) / count

			_sum += per_occurrence_power * count

			print(instruction, count, total_instructions, power_joules, per_occurrence_power)
			
			samples[instruction].append(per_occurrence_power)
	print("TOTAL SUM:", _sum)

	return {
		instruction: sum(values) / len(values)
		for instruction, values in samples.items()
		if values
	}


def estimate_input_instruction_power(
	input_path: Path,
	instruction_power_map: dict[str, float],
	fallback_power: float,
) -> list[dict[str, str | float]]:
	estimates: list[dict[str, str | float]] = []

	with input_path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.reader(handle, skipinitialspace=True)
		rows = list(reader)

		if not rows:
			return estimates

		first_row = [cell.strip() for cell in rows[0]]
		header = {name.lower(): idx for idx, name in enumerate(first_row)}
		has_header = all(key in header for key in ("kernel_name", "instruction", "count"))

		data_rows = rows[1:] if has_header else rows
		kernel_idx = header.get("kernel_name", 0)
		instruction_idx = header.get("instruction", 1)
		count_idx = header.get("count", 2)

		for row in data_rows:
			if len(row) <= max(kernel_idx, instruction_idx, count_idx):
				continue

			kernel_name = row[kernel_idx].strip()
			instruction = row[instruction_idx].strip()
			count_raw = row[count_idx].strip()

			if not instruction or not count_raw:
				continue

			try:
				count = float(count_raw)
			except ValueError:
				continue

			used_fallback = instruction not in instruction_power_map
			avg_per_occurrence = instruction_power_map.get(instruction, fallback_power)
			estimated_power_joules = count * avg_per_occurrence

			estimates.append(
				{
					"kernel_name": kernel_name,
					"instruction": instruction,
					"count": count,
					"avg_power_per_occurrence_joules": avg_per_occurrence,
					"estimated_power_joules": estimated_power_joules,
					"used_fallback": used_fallback,
				}
			)

	return estimates


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	weights_path = base_dir / "weights.csv"
	input_path = base_dir / "input.csv"

	instruction_power_map = load_instruction_power_map(weights_path)
	fallback_power = (
		sum(instruction_power_map.values()) / len(instruction_power_map)
		if instruction_power_map
		else 0.0
	)

	print("Instruction average power map (joules per occurrence):")
	for instruction in sorted(instruction_power_map):
		print(f"{instruction},{instruction_power_map[instruction]:.12f}")
	print(f"FALLBACK_AVG,{fallback_power:.12f}")

	print("\nInput estimates:")
	estimates = estimate_input_instruction_power(
		input_path,
		instruction_power_map,
		fallback_power,
	)
	total_estimated = 0.0
	for entry in estimates:
		total_estimated += float(entry["estimated_power_joules"])
		fallback_tag = " (fallback)" if entry.get("used_fallback") else ""
		print(
			f"{entry['kernel_name']},{entry['instruction']},"
			f"count={entry['count']:.0f},"
			f"avg={entry['avg_power_per_occurrence_joules']:.12f},"
			f"estimated={entry['estimated_power_joules']:.12f}"
			f"{fallback_tag}"
		)

	print(f"\nTotal estimated power for input rows: {total_estimated:.12f} J")


if __name__ == "__main__":
	main()
