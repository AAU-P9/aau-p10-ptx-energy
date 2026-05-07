from __future__ import annotations
from keras.optimizers import Adam
import os

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

from keras import Sequential
from keras.layers import Dense, Input

# load dataset and prepare features/targets
import pandas as pd

# Boolean to skip training and load weights instead
SKIP_TRAINING = True

# locate data.csv next to this example.py
data_path = os.path.join(os.path.dirname(__file__), "data.csv")

# Read the dataset from the CSV file
csv_df = pd.read_csv(data_path)

# Instructions
instructions = [
    "add.f32",
    "add.s32",
    "add.s64",   
    "bra",
    "bra.uni",
    "cvt.s64.s32",
    "ld.param.u64",
    "mov.b32",
    "mov.f32",
    "mov.u32",
    "mul.lo.s32",
    "not.pred",
    "ret",
    "setp.lt.s32",
    "shl.b64",
    "st.f32",
]

# Create a hashmap with the indicies for each instruction
instruction_indices = {instruction: idx for idx, instruction in enumerate(instructions)}

# Itterate over the CSV and verify we have a kernel for each instruction
for instruction in instructions:
    if instruction not in csv_df["kernelName"].values:
        print(f"[Error]: instruction {instruction} not found in dataset")


# Iterate over CSV and build feature vectors and targets
kernel_xs = {}
kernel_ys = {}
for idx, row in csv_df.iterrows():
    if not row["kernelName"] in kernel_xs:
        print(f"[Error]: kernelName {row['kernelName']} not found in instruction indices")
        kernel_xs[row["kernelName"]] = [0] * len(instructions)
        kernel_ys[row["kernelName"]] = row["total_power_consumption_joules"]

    if row["kernelName"] in kernel_xs:
        # Create an empty vector with 0s for all instructions
        if row["instruction"] in instruction_indices:
            kernel_xs[row["kernelName"]][instruction_indices[row["instruction"]]] = row["occurrences"]
        else:
            print(f"[Error]: instruction for {row['kernelName']} {row['instruction']} not found in instruction indices")

# Create a copy of the kernels you want to use for testing the model after training (e.g., "matrix_mul")
testing_kernels = ["matrix_mul", "vector_add_old"]
testing_kernels_xs = {}
testing_kernels_ys = {}
for kernel in testing_kernels:
    testing_kernels_xs[kernel] = kernel_xs.get(kernel)
    testing_kernels_ys[kernel] = kernel_ys.get(kernel)

    del kernel_xs[kernel]
    del kernel_ys[kernel]

    filtered_kernel_xs = {k: v for k, v in kernel_xs.items() if k != kernel}
    filtered_kernel_ys = {k: v for k, v in kernel_ys.items() if k != kernel}

# Verify that the data is loaded correctly and does not contain the testing kernels
print(filtered_kernel_ys.keys())

# Convert the kernel_xs and kernel_ys dictionaries to numpy arrays
raw_xs = np.array(list(kernel_xs.values()))
raw_ys = np.array(list(kernel_ys.values())).reshape(-1, 1)

filtered_xs = np.array(list(filtered_kernel_xs.values()))
filtered_ys = np.array(list(filtered_kernel_ys.values())).reshape(-1, 1)

# Print the first pair of feature vector and target value to verify the data
print("First feature vector (X[0]):\n", raw_xs[0])
print("First target value (y[0]):\n", raw_ys[0])

# Apply log scaling to prevent NaN from large values
kernel_xs = np.log1p(filtered_xs)
ys = np.log1p(filtered_ys)

print("First normalized feature vector (X[0]):\n", kernel_xs[0])
print("First normalized target value (y[0]):\n", ys[0])

# numeric stability: heavy-tailed counts cause overflow/NaN in training.
instruction_names = None
loaded_stats = False

# number of input features = number of unique instructions
input_features = len(instructions)
output_units = 1

layers = [Input(shape=(input_features,))]
layers.append(Dense(output_units, kernel_constraint="non_neg"))
model = Sequential(layers)


model.compile(
    optimizer=Adam(learning_rate=0.05),
    loss="mse"
)

if SKIP_TRAINING:
    # Load weights from disk
    weights_path = os.path.join(os.path.dirname(__file__), "weights_858e05.npz")
    loaded_weights = np.load(weights_path, allow_pickle=True)
    weights_list = [loaded_weights[f"arr_{i}"] for i in range(len(loaded_weights.files))]
    model.set_weights(weights_list)
    print(f"Weights loaded from {weights_path}")
else:
    model.fit(x=kernel_xs, y=ys, epochs=10_000, verbose=1)

    # Print the model's weights to verify that it has learned something
    weights = model.get_weights()
    print("Model weights:\n", weights)


    # Save weights to disk
    weights_path = os.path.join(os.path.dirname(__file__), "weights.npz")
    np.savez(weights_path, *weights)
    print(f"Weights saved to {weights_path}")

# Verify that the model can predict the power consumption for the 'matrix_mul' kernel
for kernel_name in testing_kernels:
    kernel_xs = testing_kernels_xs[kernel_name]
    actual_power = testing_kernels_ys[kernel_name]

    xs_log = np.log1p([kernel_xs])
    normalized_target = model.predict(xs_log)
    print(f"Log-scaled prediction for '{kernel_name}' kernel:", normalized_target)

    predicted_power = np.expm1(normalized_target)
    predicted_power = predicted_power[0][0]
    print(f"Predicted power consumption for '{kernel_name}' kernel:", predicted_power)
    print(f"Actual power consumption for '{kernel_name}' kernel:", actual_power)

    # Print the % error of the prediction
    error_percentage = abs(predicted_power - actual_power) / actual_power * 100
    print(f"Prediction error for '{kernel_name}' kernel: {error_percentage:.2f}%\n")