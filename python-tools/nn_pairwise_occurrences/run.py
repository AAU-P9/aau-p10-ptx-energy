from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .model import KERNEL_DATASETS, SCALE_CONSTANT, STATS_PATH, WEIGHTS_PATH, load_preproc_stats, load_weights, make_model, normalize_data

def main():
    p = argparse.ArgumentParser(description="Run model on a dataset of kernel JSON files")
    p.add_argument("--data-dir", type=Path, default=None, help="Directory with kernel JSON files (overrides default)")
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH, help="Weights file path")
    p.add_argument("--stats", type=Path, default=STATS_PATH, help="Preprocessing stats path")
    p.add_argument("--csv-output-path", type=Path, default=None, help="Path to write CSV output")
    args = p.parse_args()

    stats_path = str(args.stats)
    feature_indices = load_preproc_stats(stats_path)

    model, _, __ = make_model(len(feature_indices))
    load_weights(model, str(args.weights))

    data_dir = args.data_dir or KERNEL_DATASETS[0]
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.json"))

    missing_features: list[tuple[str, str]] = []
    predictions = []
    for f in files:
        with f.open("r") as fh:
            data = json.load(fh)
        
        fv, _ = normalize_data(data, feature_indices, missing_features)
        fv = np.array(fv, dtype=np.float32)
        
        total_blocks = (
            data.get("gridDim", {}).get("x", 1) * 
            data.get("gridDim", {}).get("y", 1) * 
            data.get("gridDim", {}).get("z", 1) * 
            data.get("blockDim", {}).get("x", 1) * 
            data.get("blockDim", {}).get("y", 1) * 
            data.get("blockDim", {}).get("z", 1)
        )
        total_instructions = data.get("totalInstructions", 1)
        # compute total threads and total_pairs for scaling back prediction
        total_threads = total_blocks * total_instructions
        dep = data.get("dependentPairs", {}) or {}
        indep = data.get("independentPairs", {}) or {}
        total_pairs = sum(int(v) for v in dep.values()) + sum(int(v) for v in indep.values()) or 1

        pred_normalized = model.predict(fv.reshape(1, -1), verbose=0)[0][0]
        # Divide out the neural network's scaling constant at the end
        pred = (pred_normalized / SCALE_CONSTANT) * total_threads * total_pairs
        
        kernel_name = data.get("kernelName", f.stem)
        predictions.append((kernel_name, pred, data.get("powerConsumptionJoules"), int((fv != 0).sum())))

    print("\nPredicted power consumption for kernels:")
    for name, pred, actual, num_nonzero in predictions:
        print(f"  - {name}: Predicted = {pred:.6f} J, Actual = {actual:.6f} J, Features = {num_nonzero}/{len(feature_indices)}")

    if predictions:
        actuals = np.asarray([actual for _, _, actual, _ in predictions], dtype=np.float64)
        preds = np.asarray([pred for _, pred, _, _ in predictions], dtype=np.float64)
        abs_errors = np.abs(preds - actuals)
        mae = float(np.mean(abs_errors))
        mape = float(np.mean(abs_errors / np.clip(actuals, 1e-9, None)) * 100.0)
        print(f"\nAggregate error: MAE = {mae:.6f} J, MAPE = {mape:.2f}%")

    if args.csv_output_path:
        with args.csv_output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["kernelName", "powerConsumptionJoules", "predictedPowerConsumptionJoules"])
            writer.writeheader()
            for name, pred, actual, _ in predictions:
                writer.writerow({
                    "kernelName": name,
                    "powerConsumptionJoules": actual,
                    "predictedPowerConsumptionJoules": pred
                })

    if missing_features:
        print("\n[WARNING] Missing feature identifiers encountered from pairs (dependentPairs/independentPairs):")
        for (kernel, instr) in sorted(missing_features):
            print(f"  - {instr}: {kernel}")


if __name__ == "__main__":
    main()
