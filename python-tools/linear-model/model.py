#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

def load_instruction_power_map(weights_input_path: Path) -> dict[str, float]:
    """Build average per-occurrence power (joules) for each instruction by comparing rpt1 and rpt2."""
    instruction_weights = {}

    if not weights_input_path.is_dir():
        print(f"[WARNING] {weights_input_path} is not a directory.")
        return instruction_weights

    # Find all rpt1 files
    rpt1_files = list(weights_input_path.glob("*_rpt1_*.json"))
    
    for f1 in rpt1_files:
        filename = f1.name
        f2 = f1.with_name(filename.replace("_rpt1_", "_rpt2_"))
        
        if not f2.is_file():
            continue
            
        try:
            with f1.open("r", encoding="utf-8") as handle:
                r1 = json.load(handle)
            with f2.open("r", encoding="utf-8") as handle:
                r2 = json.load(handle)

            total_blocks = r1.get("gridDim").get("x", 0) * r1.get("gridDim").get("y", 0) * r1.get("gridDim").get("z", 0) * r1.get("blockDim").get("x", 0) * r1.get("blockDim").get("y", 0) * r1.get("blockDim").get("z", 0)
            inst_occurences = r1.get("instructionOccurrences", {}).get(r1.get("kernelName", ""), None)

            if inst_occurences is None:
                # print(f"[WARNING] No instruction occurrences found for kernel '{r1.get('kernelName', '')}' in file {f1}")
                continue

            # The number of executions of the instruction is the number of blocks times the number of occurrences of the instruction per block
            inst_executions = total_blocks * inst_occurences

            power_diff = r2.get("powerConsumptionJoules", 0.0) - r1.get("powerConsumptionJoules", 0.0)
            weight = power_diff / inst_executions

            err_pct = r1.get("powerConsumptionJoules", 0.0) / (weight * inst_executions) - 1.0

            instruction_weights[r1.get("kernelName", "")] = weight

            print(f"[DEBUG] {filename}: power_diff={power_diff:.6f} J, inst_executions={inst_executions}, weight={weight:.12f} J/occurrence, err_pct={err_pct:.2%}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] Failed to process files {f1} and {f2}: {e}")
            continue

    for instruction, weight in instruction_weights.items():
        print(f"[INFO] '{instruction}' mapped weight: {weight:.12f} J/occurrence")

    return instruction_weights


def estimate_kernel_energy(
    item: dict,
    instruction_power_map: dict[str, float],
    fallback_power: float,
) -> float:
    kernel_name = item.get("kernelName") or "Unknown"
    total_blocks = item.get("gridDim", {}).get("x", 0) * item.get("gridDim", {}).get("y", 0) * item.get("gridDim", {}).get("z", 0) * item.get("blockDim", {}).get("x", 0) * item.get("blockDim", {}).get("y", 0) * item.get("blockDim", {}).get("z", 0)
    instruction_occurrences = item.get("instructionOccurrences")

    
    total_energy_j = 0.0
    for instruction, count_raw in instruction_occurrences.items():
        instruction_name = str(instruction).strip()
        if not instruction_name:
            continue

        try:
            count = float(count_raw) * total_blocks
        except (TypeError, ValueError):
            raise ValueError(f"Invalid count for instruction '{instruction_name}' in kernel '{kernel_name}': {count_raw} (must be a number)")

        if count <= 0:
            raise ValueError(f"Invalid count for instruction '{instruction_name}' in kernel '{kernel_name}': {count} (must be positive)")

        avg_per_occurrence = instruction_power_map.get(instruction_name, -1)
        if avg_per_occurrence < 0:
            # print(f"[WARNING] No weight found for instruction '{instruction_name}' in kernel '{kernel_name}'. Using fallback power {fallback_power:.12f} J/occurrence.")
            continue

        estimated_power_joules = count * avg_per_occurrence

        total_energy_j += estimated_power_joules

    return total_energy_j


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate instruction-level power from PTX analysis JSON.",
    )
    parser.add_argument(
        "--weights-input-path",
        type=Path,
        required=True,
        help="Path to directory containing reference weights JSON files (e.g. data/abc)",
    )
    parser.add_argument(
        "--kernels-input-path",
        type=Path,
        required=True,
        help="Path to kernels JSON file or directory of JSON files to be analyzed",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to write JSON output",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_input_path = args.weights_input_path
    kernels_input_path = args.kernels_input_path

    if not weights_input_path.is_dir():
        raise FileNotFoundError(f"Weights input directory not found: {weights_input_path}")

    instruction_power_map = load_instruction_power_map(weights_input_path)

    fallback_power = (
        sum(instruction_power_map.values()) / len(instruction_power_map)
        if instruction_power_map
        else 0.0
    )
    
    print(f"[INFO] Fallback power: {fallback_power:.12f} J")

    kernel_files = []
    if kernels_input_path.is_dir():
        kernel_files = list(kernels_input_path.glob("*.json"))
    elif kernels_input_path.is_file():
        kernel_files = [kernels_input_path]
    else:
        raise FileNotFoundError(f"Kernels input path not found: {kernels_input_path}")

    for input_path in kernel_files:
        with input_path.open("r", encoding="utf-8") as handle:  
            payload = json.load(handle)

            actual_energy = payload.get("powerConsumptionJoules", None)
            estimated_energy = estimate_kernel_energy(
                payload,
                instruction_power_map,
                fallback_power,
            )
            
            err = ((estimated_energy - actual_energy) / actual_energy) 

            print(f"[RESULT] {input_path.name}: Estimated Energy = {estimated_energy:.6f} J, Actual Energy = {actual_energy:.6f} J, Error = {err:.2%}")

if __name__ == "__main__":
    main()
