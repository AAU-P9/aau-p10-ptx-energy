from __future__ import annotations
import json
import os
from pathlib import Path
import time

# JAX backend will auto-select available hardware
os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
from keras.optimizers import Adam
import numpy as np

# Verify available JAX devices
print(f"JAX devices: {jax.devices()}")

from keras import Sequential
from keras.layers import Dense, Input

# Boolean to skip training and load weights instead
weights_file = "weights.npz"

weights_path = os.path.join(os.path.dirname(__file__), weights_file)
SKIP_TRAINING = os.path.exists(weights_path)

# Load JSON files from the data folder next to this script
datasets = [Path("/home/rasmus/aau-p10-ptx-energy/data/linear_model_microbenchmarks")]
kernel_datasets = [Path("/home/rasmus/aau-p10-ptx-energy/data/kernels")]


def build_feature_vector(data: dict) -> list[int]:
    feature_vector = [0] * len(instruction_indices)

    # Keep feature construction identical for training and inference.
    for instruction, count in data.get("instructionOccurrences", {}).items():
        if instruction in instruction_indices:
            total_blocks = data.get("gridDim", {}).get("x", 0) * data.get("gridDim", {}).get("y", 0) * data.get("gridDim", {}).get("z", 0) * data.get("blockDim", {}).get("x", 0) * data.get("blockDim", {}).get("y", 0) * data.get("blockDim", {}).get("z", 0)
            feature_vector[instruction_indices[instruction]] = count * total_blocks
        else:
            print(f"[Warning] Instruction '{instruction}' not in instructions, skipping.")

    return feature_vector

# Create the feature set from the dataset by using the unique instructions as features
instruction_indices = {}
all_data_json_files = []
for data_dir in datasets:
    data_json_files = sorted(data_dir.glob("*.json"))
    all_data_json_files.extend(data_json_files)
    for json_file in data_json_files:
        with json_file.open("r") as f: 
            data = json.load(f)
        for instruction in data.get("instructionOccurrences", {}).keys():
            if instruction not in instruction_indices:
                instruction_indices[instruction] = len(instruction_indices)

# Iterate over JSON files and build feature vectors and targets
kernel_xs = {}
kernel_ys = {}

missing_instructions = []
for json_file in all_data_json_files:
    with json_file.open("r") as f:
        data = json.load(f)

    kernel_name = json_file.stem

    feature_vector = build_feature_vector(data)

    for instruction in data.get("instructionOccurrences", {}).keys():
        if instruction not in instruction_indices:
            missing_instructions.append(instruction)

    kernel_xs[kernel_name] = feature_vector
    kernel_ys[kernel_name] = data.get("powerConsumptionJoules")

if missing_instructions:
    unique_missing = set(missing_instructions)
    print(f"\n[Summary] Missing instructions not in the fixed set (total {len(missing_instructions)}, unique {len(unique_missing)}):")
    for instr in unique_missing:
        print(f"  - {instr}")
    exit(1)

if not kernel_xs:
    raise RuntimeError(f"No JSON files found in {data_dir}")

# Convert the kernel_xs and kernel_ys dictionaries to numpy arrays
raw_xs = np.array(list(kernel_xs.values()))
raw_ys = np.array(list(kernel_ys.values())).reshape(-1, 1)

# Print the first pair of feature vector and target value to verify the data
print("First feature vector (X[0]):\n", raw_xs[0])
print("First target value (y[0]):\n", raw_ys[0])

# Apply log scaling to prevent NaN from large values
xs = raw_xs.astype(np.float32) / raw_xs.max()
ys = raw_ys.astype(np.float32) / raw_ys.max()

# numeric stability: heavy-tailed counts cause overflow/NaN in training.
instruction_names = None
loaded_stats = False

# number of input features = number of unique instructions
output_units = 1

layers = [
    Input(shape=(len(instruction_indices),)),
    Dense(len(instruction_indices) * 8, activation="leaky_relu"),
    Dense(output_units, kernel_constraint="non_neg") 
]
model = Sequential(layers)

model.compile(
    optimizer=Adam(learning_rate=0.000001),
    loss="mse",
)

if SKIP_TRAINING:
    # Load weights from disk
    loaded_weights = np.load(weights_path, allow_pickle=True)
    weights_list = [loaded_weights[f"arr_{i}"] for i in range(len(loaded_weights.files))]
    model.set_weights(weights_list)
    print(f"Weights loaded from {weights_path}")
else:
    print("Starting training on GPU...")

    start_time = time.time()
    model.fit(x=xs, y=ys, epochs=100_000, batch_size=len(xs), verbose=1)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")

    # Print the final training loss
    final_loss = model.evaluate(xs, ys, verbose=0)
    print(f"Final training loss: {final_loss:.6f}")

    # Print the model's weights to verify that it has learned something
    weights = model.get_weights()
    # print("Model weights:\n", weights)


    # Save weights to disk
    np.savez(weights_path, *weights)
    print(f"Weights saved to {weights_path}")

# Verify on the training data that the model can predict the power consumption
error_accum = 0.0
for i in range(len(xs)):
    features = xs[i]
    actual_power = raw_ys[i][0]

    predicted_power = model.predict(features.reshape(1, -1), verbose=0)
    predicted_power = predicted_power[0][0] * raw_ys.max()  # Scale back to original units

    error = abs(predicted_power - actual_power)
    error_accum += error

print(f"Average prediction error on training data: {error_accum / len(xs):.6f} Joules")

predictions = []
# Verify on the kernels JSON files that the model can predict the power consumption'
for kernel_data_dir in kernel_datasets:
    kernels_json_files = sorted(kernel_data_dir.glob("*.json"))
    for json_file in kernels_json_files:
        with json_file.open("r") as f:
            data = json.load(f)

        kernel_name = json_file.stem

        feature_vector = np.array(build_feature_vector(data), dtype=np.float32) / raw_xs.max()
        predicted_power = model.predict(feature_vector.reshape(1, -1), verbose=0)
        predicted_power = predicted_power[0][0] * raw_ys.max()  # Scale back to original units

        predictions.append((kernel_name, predicted_power, data.get("powerConsumptionJoules")))

print("\nPredicted power consumption for kernels:")
for kernel_name, predicted_power, actual_power in predictions:
    print(f"  - {kernel_name}: Predicted = {predicted_power:.6f} J, Actual = {actual_power:.6f} J")