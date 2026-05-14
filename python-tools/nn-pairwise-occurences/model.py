from __future__ import annotations
from keras.optimizers import Adam
import os
import json
import glob
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

from keras import Sequential
from keras.layers import Dense, Input

# Boolean to skip training and load weights instead
SKIP_TRAINING = False
weights_file = "weights.npz"

# Load JSON files from the data folder next to this script
data_dir = Path("/home/rasmus/aau-p10-ptx-energy/data/generates")
data_json_files = sorted(data_dir.glob("*.json"))

kernels_dir = Path(__file__).resolve().parent / "kernels"
kernels_json_files = sorted(kernels_dir.glob("*.json"))

# Extract unique instructions from the dataset to create a fixed set of features
instruction_indices = {}
for json_file in data_json_files:
    with json_file.open("r") as f: 
        data = json.load(f)
        for instruction in data.get("instructionOccurrences", {}).keys():
            instruction_indices[instruction] = 1

# Create pairs of instructions to use as features for the model. i.e. "add.f32, addf32", "add.f32, mul.f32"...
instructions = list(instruction_indices.keys())
instruction_pairs = [f"{i},{j}" for i in instructions for j in instructions]
instruction_indices = {pair: idx for idx, pair in enumerate(instruction_pairs)}

# Load feature vectors and targets from JSON files
kernel_xs = {}
kernel_ys = {}

print(len(instruction_pairs))
exit(1)

for json_file in data_json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        kernel_name = Path(json_file).stem

        # Create feature vector initialized to zeros
        feature_vector = [0] * len(instruction_pairs)
        
        # Add counts from dependent pairs (only if in our fixed set)
        for pair, count in data.get("dependentPairs", {}).items():
            if pair in instruction_indices:
                feature_vector[instruction_indices[pair]] = count
            else:
                print(f"Warning: Pair '{pair}' from {json_file} not in instruction_pairs, skipping.") 

        # Add counts from independent pairs (combine with dependent)
        for pair, count in data.get("independentPairs", {}).items():
            if pair in instruction_indices:
                feature_vector[instruction_indices[pair]] += count
            else:
                print(f"Warning: Pair '{pair}' from {json_file} not in instruction_pairs, skipping.")

        kernel_xs[kernel_name] = feature_vector
        kernel_ys[kernel_name] = data.get("powerConsumptionJoules")

# Convert the kernel_xs and kernel_ys dictionaries to numpy arrays
kernel_names = list(kernel_xs.keys())
raw_xs = np.array(list(kernel_xs.values()))
raw_ys = np.array(list(kernel_ys.values())).reshape(-1, 1)

# Print the first pair of feature vector and target value to verify the data
print("First feature vector (X[0]):\n", raw_xs[0])
print("First target value (y[0]):\n", raw_ys[0])

# Apply log scaling to prevent NaN from large values
xs = raw_xs.astype(np.float32) / raw_xs.max()
ys = raw_ys.astype(np.float32) / raw_ys.max()

print("First normalized feature vector (X[0]):\n", xs[0])
print("First normalized target value (y[0]):\n", ys[0])

# numeric stability: heavy-tailed counts cause overflow/NaN in training.
instruction_names = None
loaded_stats = False

# number of input features = number of unique instruction pairs
output_units = 1

layers = [
    Input(shape=(len(instruction_pairs),)),
    Dense(len(instructions) * 4, activation="leaky_relu"),
    Dense(output_units)
]
model = Sequential(layers)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="mse",
)

if SKIP_TRAINING:
    # Load weights from disk
    weights_path = os.path.join(os.path.dirname(__file__), weights_file)
    loaded_weights = np.load(weights_path, allow_pickle=True)
    weights_list = [loaded_weights[f"arr_{i}"] for i in range(len(loaded_weights.files))]
    model.set_weights(weights_list)
    print(f"Weights loaded from {weights_path}")
else:
    model.fit(x=xs, y=ys, epochs=2_000, verbose=1)

    # Print the model's weights to verify that it has learned something
    weights = model.get_weights()

    # Save weights to disk
    weights_path = os.path.join(os.path.dirname(__file__), "weights.npz")
    np.savez(weights_path, *weights)
    print(f"Weights saved to {weights_path}")

# Verify that the model can predict the power consumption using the test data
predictions = model.predict(xs)
predictions_original = np.expm1(predictions)
actual_original = np.expm1(ys)

# print("\nPredictions:")
# print(f"{'Filename':<60} {'Log-scaled Pred':<20} {'Original Pred':<20} {'Log-scaled Actual':<20} {'Original Actual':<20} {'% Error':<15}")
# print("-" * 155)
# for i, filename in enumerate(kernel_names):
#     percent_error = abs(predictions_original[i][0] - actual_original[i][0]) / actual_original[i][0] * 100
#     print(f"{filename:<60} {predictions[i][0]:<20.6f} {predictions_original[i][0]:<20.6f} {ys[i][0]:<20.6f} {actual_original[i][0]:<20.6f} {percent_error:<15.2f}%")

# Itterate over kernels in 'kernels_json_files' and predict power consumption, then print the results
print("\nPredictions for kernels:")
print(f"{'Filename':<60} {'Actual Power':<20} {'Original Pred':<20}")
print("-" * 100)
for json_file in kernels_json_files:
    feature_vector = [0] * len(instruction_pairs)
    with open(json_file, 'r') as f:
        data = json.load(f)
        for pair, count in data.get("dependentPairs", {}).items():
            if pair in instruction_indices:
                feature_vector[instruction_indices[pair]] = count
            else:
                print(f"[Warning] Pair '{pair}' from {json_file} not in instruction_pairs, skipping.")
        for pair, count in data.get("independentPairs", {}).items():
            if pair in instruction_indices:
                feature_vector[instruction_indices[pair]] += count
            else:
                print(f"[Warning] Pair '{pair}' from {json_file} not in instruction_pairs, skipping.")

        # Get the actual power consumption from the JSON file for comparison
        actual_power = data.get("powerConsumptionJoules")

        # Normalize the feature vector using log scaling
        feature_vector = np.array(feature_vector, dtype=np.float32) / raw_xs.max()

        # Predict the log-scaled power consumption and convert back to original scale
        predicted_log_power = model.predict(feature_vector, verbose=0)[0][0]
        predicted_power = predicted_log_power * raw_ys.max()

        print(f"{json_file.name:<60} {actual_power:<20.6f} {predicted_power:<20.6f}")
