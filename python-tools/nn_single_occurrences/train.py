from __future__ import annotations
import json
import time
import numpy as np

from .model import DATASETS, WEIGHTS_PATH, make_model, normalize_data, save_weights


def main():
    instruction_indices: dict[str, int] = {}
    # build instruction index map
    for data_dir in DATASETS:
        data_json_files = sorted(data_dir.glob("*.json"))
        for json_file in data_json_files:
            with json_file.open("r") as f:
                data = json.load(f)
            for instruction in data.get("instructionOccurrences", {}).keys():
                if instruction not in instruction_indices:
                    instruction_indices[instruction] = len(instruction_indices)

    # collect feature vectors and targets
    all_files = []
    for data_dir in DATASETS:
        all_files.extend(sorted(data_dir.glob("*.json")))

    if not all_files:
        raise RuntimeError("No JSON files found in configured DATASETS")

    kernel_xs = {}
    kernel_ys = {}
    missing = set()
    
    for json_file in all_files:
        with json_file.open("r") as f:
            data = json.load(f)
        name = json_file.stem
        features, target = normalize_data(data, instruction_indices, missing)
        kernel_xs[name] = features
        kernel_ys[name] = target

        print(features, target)

    xs = np.array(list(kernel_xs.values()), dtype=np.float32)
    ys = np.array(list(kernel_ys.values()), dtype=np.float32)

    model, batch_size, epochs = make_model(len(instruction_indices))

    print("Starting training...")

    start = time.time()
    interrupted = False
    try:
        model.fit(x=xs, y=ys, epochs=epochs, batch_size=batch_size, verbose=1)
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted; saving weights and stats.")
    elapsed = time.time() - start
    if not interrupted:
        print(f"Training finished in {elapsed:.2f}s")

    final_loss = model.evaluate(xs, ys, verbose=0)

    # Print this in scientific notation to handle very small values
    print(f"Final training loss: {final_loss:.6e}")

    if all_files:
        print("\nCalculating aggregates over all training files...")
        all_preds_normalized = model.predict(xs, verbose=0).flatten()
        
        errors = []
        actuals = []
        
        for json_file, pred_normalized in zip(all_files, all_preds_normalized):
            with json_file.open("r") as f:
                data = json.load(f)
            
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
            
            pred_joules = (pred_normalized / 1e11) * scale_factor
            actual_joules = data.get("powerConsumptionJoules", 0.0)
            
            errors.append(abs(pred_joules - actual_joules))
            actuals.append(actual_joules)
            
        errors = np.array(errors)
        actuals = np.array(actuals)
        
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        max_error = np.max(errors)
        mean_actual = np.mean(actuals)
        mape = np.mean(errors / np.clip(actuals, 1e-9, None)) * 100
        
        print(f"  Mean Absolute Error (Joules): {mae:.6f}")
        print(f"  Root Mean Squared Error:      {rmse:.6f}")
        print(f"  Max Absolute Error:           {max_error:.6f}")
        print(f"  Mean Actual Joules:           {mean_actual:.6f}")
        print(f"  MAPE:                         {mape:.2f}%")

    save_weights(model, str(WEIGHTS_PATH))

if __name__ == "__main__":
    main()
