#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

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

			print(f"[DEBUG] Processing row: instruction='{instruction}', count='{count_raw}', total_instructions='{total_instructions}', power_consumption_joules='{power_raw}'")

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
			
			samples[instruction].append(per_occurrence_power)

	# Print span of the samples for each instructions
	for instruction, values in samples.items():
		if values:
			min_val = min(values)
			max_val = max(values)
			percent_diff = (max_val - min_val) / min_val * 100 if min_val > 0 else float('inf')
			print(f"[INFO] '{instruction}' samples {len(values)}, span: {min_val:.12f} J - {max_val:.12f} J" f" (diff: {percent_diff:.2f}%)")

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

	with input_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	analysis_items = payload if isinstance(payload, list) else [payload]

	for index, item in enumerate(analysis_items):
		if not isinstance(item, dict):
			continue

		kernel_name = str(item.get("kernelName") or input_path.stem or f"kernel_{index}")
		instruction_occurrences = item.get("instructionOccurrences")

		if not isinstance(instruction_occurrences, dict):
			continue

		for instruction, count_raw in instruction_occurrences.items():
			instruction_name = str(instruction).strip()
			if not instruction_name:
				continue

			try:
				count = float(count_raw)
			except (TypeError, ValueError):
				continue

			if count <= 0:
				continue

			used_fallback = instruction_name not in instruction_power_map
			avg_per_occurrence = instruction_power_map.get(instruction_name, fallback_power)
			estimated_power_joules = count * avg_per_occurrence

			estimates.append(
				{
					"kernel_name": kernel_name,
					"instruction": instruction_name,
					"count": count,
					"avg_power_per_occurrence_joules": avg_per_occurrence,
					"estimated_power_joules": estimated_power_joules,
					"used_fallback": used_fallback,
				}
			)

	return estimates


def parse_args() -> argparse.Namespace:
	base_dir = Path(__file__).resolve().parent
	project_root = base_dir.parent

	parser = argparse.ArgumentParser(
		description="Estimate instruction-level power from PTX analysis JSON.",
	)
	parser.add_argument(
		"--weights-path",
		type=Path,
		default=base_dir / "weights.csv",
		help="Path to weights CSV file (default: linear-model/weights.csv)",
	)
	parser.add_argument(
		"--input-path",
		type=Path,
		default=project_root / "ptx_analysis.json",
		help="Path to PTX analysis JSON file (default: ptx_analysis.json)",
	)

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	weights_path = args.weights_path
	input_path = args.input_path

	if not weights_path.is_file():
		raise FileNotFoundError(f"Weights CSV not found: {weights_path}")

	if not input_path.is_file():
		raise FileNotFoundError(f"Input JSON not found: {input_path}")

	instruction_power_map = load_instruction_power_map(weights_path)
	fallback_power = (
		sum(instruction_power_map.values()) / len(instruction_power_map)
		if instruction_power_map
		else 0.0
	)
	
	print(f"[INFO] Fallback power: {fallback_power:.12f} J")

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
			"[ESTIMATE] "
			f"{entry['kernel_name']},{entry['instruction']},"
			f"count={entry['count']:.0f},"
			f"avg={entry['avg_power_per_occurrence_joules']:.12f},"
			f"estimated={entry['estimated_power_joules']:.12f}"
			f"{fallback_tag}"
		)

	print(f"[TOTAL] {total_estimated:.12f} J")


if __name__ == "__main__":
	main()
