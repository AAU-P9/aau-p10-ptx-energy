from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .model import STATS_PATH, WEIGHTS_PATH, make_model, load_preproc_stats, load_weights, normalize_data, KERNEL_DATASETS


def main():
    p = argparse.ArgumentParser(description="Run model on a dataset of kernel JSON files")
    p.add_argument("--data-dir", type=Path, default=None, help="Directory with kernel JSON files (overrides default)")
    p.add_argument("--weights", type=Path, default=WEIGHTS_PATH, help="Weights file path")
    p.add_argument("--stats", type=Path, default=STATS_PATH, help="Preprocessing stats path")
    args = p.parse_args()

    stats_path = str(args.stats)
    instruction_indices = load_preproc_stats(stats_path)

    model, _, __ = make_model(len(instruction_indices))
    load_weights(model, str(args.weights))

    data_dir = args.data_dir or KERNEL_DATASETS[0]
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.json"))

    missing = set()
    predictions = []
    for f in files:
        with f.open("r") as fh:
            data = json.load(fh)
        
        fv, _ = normalize_data(data, instruction_indices, missing)
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
        scale_factor = total_blocks * total_instructions

        pred_normalized = model.predict(fv.reshape(1, -1), verbose=0)[0][0]
        # Divide out the neural network's scaling constant at the end
        pred = (pred_normalized / 1e11) * scale_factor
        
        predictions.append((f.stem, pred, data.get("powerConsumptionJoules"), int((fv != 0).sum())))

    print("\nPredicted power consumption for kernels:")
    for name, pred, actual, num_nonzero in predictions:
        print(f"  - {name}: Predicted = {pred:.6f} J, Actual = {actual:.6f} J, Features = {num_nonzero}/{len(instruction_indices)}")

    if missing:
        print("\n[WARNING] Missing instructions encountered:")
        for instr in sorted(missing):
            print(f"  - {instr}")


if __name__ == "__main__":
    main()
