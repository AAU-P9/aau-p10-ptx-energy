from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Set, List

# JAX backend will auto-select available hardware
os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import numpy as np
from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Verify available JAX devices on import
print(f"JAX devices: {jax.devices()}")

# Filenames
WEIGHTS_FILE = "weights.npz"
STATS_FILE = "preproc_stats.npz"

# Paths (module lives in python-tools/nn_single_occurrences)
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / WEIGHTS_FILE
STATS_PATH = BASE_DIR / STATS_FILE

# Default data locations (keep original dataset locations)
DATASETS = [
    Path("/home/rasmus/aau-p10-ptx-energy/data/linear_model_microbenchmarks"),
    Path("/home/rasmus/aau-p10-ptx-energy/data/linear_model_microbenchmarks_2x"),
]
KERNEL_DATASETS = [Path("/home/rasmus/aau-p10-ptx-energy/data/kernels")]


def load_preproc_stats(path: str) -> tuple[Dict[str, int], float, float, Dict[str, float]]:
    loaded = np.load(path, allow_pickle=True)
    if "instruction_indices_keys" in loaded.files and "instruction_indices_values" in loaded.files:
        keys = loaded["instruction_indices_keys"].tolist()
        values = loaded["instruction_indices_values"].tolist()
        instruction_indices = {k: int(v) for k, v in zip(keys, values)}
    else:
        names = loaded["instruction_names"].tolist()
        instruction_indices = {name: i for i, name in enumerate(names)}

    return instruction_indices


def save_preproc_stats(
    path: str,
    instruction_indices: Dict[str, int],
) -> None:
    sorted_items = sorted(instruction_indices.items(), key=lambda item: item[1])
    keys = np.array([k for k, _ in sorted_items], dtype=object)
    vals = np.array([v for _, v in sorted_items], dtype=np.int32)

    savez_dict = {
        "instruction_indices_keys": keys,
        "instruction_indices_values": vals,
    }

    np.savez(path, **savez_dict)

def normalize_data(data: dict, instruction_indices: Dict[str, int], missing_instructions: Set[str]) -> List[float]:
    features = [0.0] * len(instruction_indices)

    total_threads = data.get("gridDim", {}).get("x", 1) * data.get("gridDim", {}).get("y", 1) * data.get("gridDim", {}).get("z", 1) * data.get("blockDim", {}).get("x", 1) * data.get("blockDim", {}).get("y", 1) * data.get("blockDim", {}).get("z", 1)
    total_instructions = data.get("totalInstructions", 1)

    for instruction, count in data.get("instructionOccurrences", {}).items():
        if instruction in instruction_indices:
            features[instruction_indices[instruction]] = float(count) / total_instructions
        else:
            missing_instructions.add(instruction)

    scale_constant = 1e11 # Scale into a numeric range the NN can actually process
    target = (data.get("powerConsumptionJoules", 0.0) / (total_threads * total_instructions)) * scale_constant

    return features, target

def make_model(num_features: int) -> tuple[Sequential, int, int]:
    batch_size = 4
    epochs = 500

    layers = [
        Input(shape=(num_features,)),
        Dense(num_features * 8, activation="leaky_relu"),
        Dense(num_features * 4, activation="leaky_relu"),
        Dense(1),
    ]
    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")
    return model, batch_size, epochs


def save_weights(model: Sequential, path: str) -> None:
    weights = model.get_weights()
    np.savez(path, *weights)


def load_weights(model: Sequential, path: str) -> None:
    loaded = np.load(path, allow_pickle=True)
    weights_list = [loaded[f"arr_{i}"] for i in range(len(loaded.files))]
    model.set_weights(weights_list)
