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
SKIP_TRAINING = False
weights_file = "weights_0_076.npz"

# Load JSON files from the data folder next to this script
data_dir = Path("/home/rasmus/aau-p10-ptx-energy/data/generates")
data_json_files = sorted(data_dir.glob("*.json"))

kernels_dir = Path(__file__).resolve().parent / "kernels"
kernels_json_files = sorted(kernels_dir.glob("*.json"))

instruction_indices = {}
for json_file in data_json_files:
    with json_file.open("r") as f: 
        data = json.load(f)
        for instruction in data.get("instructionOccurrences", {}).keys():
            instruction_indices[instruction] = 1       


# Create a hashmap with the indicies for each instruction

# Iterate over JSON files and build feature vectors and targets
kernel_xs = {}
kernel_ys = {}
for json_file in data_json_files:
    with json_file.open("r") as f:
        data = json.load(f)

    kernel_name = json_file.stem

    # Create feature vector initialized to zeros
    feature_vector = [0] * len(instructions)

    # Calculate the itterations from the block and grid size
    # NOTE: Uncomment if we want to use itterations as a feature instead of baking it into the instruction counts
    block_multiplier = data.get("blockDim").get("x") * data.get("blockDim").get("y") * data.get("blockDim").get("z")
    grid_multiplier = data.get("gridDim").get("x") * data.get("gridDim").get("y") * data.get("gridDim").get("z")
    iters_multiplier = block_multiplier * grid_multiplier

    # Add counts from instruction occurrences if they exist in our fixed set
    for instruction, count in data.get("instructionOccurrences", {}).items():
        if instruction in instruction_indices:
            feature_vector[instruction_indices[instruction]] = count * iters_multiplier
        else:
            print(f"[Warning]: Instruction '{instruction}' from {json_file} not in instructions, skipping.")

    kernel_xs[kernel_name] = feature_vector
    kernel_ys[kernel_name] = data.get("powerConsumptionJoules")

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
input_features = len(instructions)
output_units = 1

layers = [
    Input(shape=(len(instructions),)),
    Dense(len(instructions) * 8, activation="leaky_relu"),
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
    weights_path = os.path.join(os.path.dirname(__file__), "weights.npz")
    np.savez(weights_path, *weights)
    print(f"Weights saved to {weights_path}")

# Print model weights (use last layer to be robust whether Input created a separate layer)
weights = model.layers[-1].get_weights()
print(f"Model weights: {weights}")

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

# Verify on the kernels JSON files that the model can predict the power consumption
for json_file in kernels_json_files:
    with json_file.open("r") as f:
        data = json.load(f)

    kernel_name = json_file.stem

    # Create feature vector initialized to zeros
    feature_vector = [0] * len(instructions)

    # Add counts from instruction occurrences if they exist in our fixed set
    for instruction, count in data.get("instructionOccurrences", {}).items():
        if instruction in instruction_indices:
            feature_vector[instruction_indices[instruction]] = count
        # else:
        #     print(f"[Warning]: Instruction '{instruction}' from {json_file} not in instructions, skipping.")

    # Apply the same log scaling to the feature vector as we did for training
    feature_vector = np.array(feature_vector, dtype=np.float32) / raw_xs.max()
    predicted_power = model.predict(feature_vector.reshape(1, -1), verbose=0)
    predicted_power = predicted_power[0][0] * raw_ys.max()  # Scale back to original units

    # Print the predicted power consumption for this kernel
    print(f"'{kernel_name}': Power consumption {data.get("powerConsumptionJoules")} Joules, Predicted: {predicted_power:.6f} Joules")
          