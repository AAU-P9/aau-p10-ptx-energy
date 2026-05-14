from pathlib import Path
import json
import sys
import webbrowser
import pandas as pd
import plotly.graph_objects as go

data_path = Path("/home/rasmus/aau-p10-ptx-energy/data/generates")
output_path = data_path.parent / "scatter_plot.html"

# Get output path from argument or use default

# Load all JSON files and extract data
# First pass: collect all instruction combinations and their signatures
all_combinations = set()
data = []

def get_instruction_signature(combination):
    """Extract unique instruction types from combination for similarity grouping."""
    # Split by underscore to get individual instructions
    instructions = combination.split("_")
    # Extract base instruction types (e.g., "add.f32" -> "add")
    instruction_types = set()
    for instr in instructions:
        base_type = instr.split(".")[0]
        instruction_types.add(base_type)
    # Return sorted tuple for consistent hashing
    return tuple(sorted(instruction_types))

for json_file in sorted(data_path.glob("*.json")):
    try:
        # Extract instruction combination from filename (everything before _r<number>)
        filename = json_file.stem
        combination = filename.rsplit("_r", 1)[0] if "_r" in filename else filename
        all_combinations.add(combination)
    except:
        pass

# Create mapping of instruction signatures to numeric values for coloring
# This groups combinations with similar instruction types
signature_map = {}
signature_counter = 0
for combination in sorted(all_combinations):
    signature = get_instruction_signature(combination)
    if signature not in signature_map:
        signature_map[signature] = signature_counter
        signature_counter += 1

# Second pass: extract data
for json_file in sorted(data_path.glob("*.json")):
    try:
        with open(json_file, "r") as f:
            json_data = json.load(f)
            block_dim = json_data.get("blockDim", {})
            grid_dim = json_data.get("gridDim", {})
            total_blocks = (block_dim.get("x", 1) * block_dim.get("y", 1) * block_dim.get("z", 1) * 
                           grid_dim.get("x", 1) * grid_dim.get("y", 1) * grid_dim.get("z", 1))
            
            # Get instruction combination from filename
            filename = json_file.stem
            combination = filename.rsplit("_r", 1)[0] if "_r" in filename else filename
            signature = get_instruction_signature(combination)
            combination_value = signature_map.get(signature, 0)
            
            data.append({
                "file": json_file.stem,
                "powerConsumptionJoules": json_data.get("powerConsumptionJoules"),
                "totalInstructions": json_data.get("totalInstructions"),
                "totalBlocks": total_blocks,
                "instructionCombination": combination,
                "combinationValue": combination_value,
            })
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {json_file}: {e}")

# Create DataFrame
df = pd.DataFrame(data)

# Create scatter plot
fig = go.Figure(data=go.Scatter(
    x=df["totalInstructions"],
    y=df["powerConsumptionJoules"],
    mode="markers",
    marker=dict(
        size=8,
        color=df["combinationValue"],
        colorscale="HSV",
        showscale=True,
        colorbar=dict(title="Instruction Combination"),
        line=dict(width=0.5, color="white")
    ),
    text=df.apply(lambda row: f"{row['file']}<br>Combination: {row['instructionCombination']}<br>Total Blocks: {row['totalBlocks']}", axis=1),
    hovertemplate="<b>%{text}</b><br>Instructions: %{x}<br>Power: %{y:.2f} J<extra></extra>",
))

fig.update_layout(
    title="Power Consumption vs Total Instructions (colored by instruction combination)",
    xaxis_title="Total Instructions",
    yaxis_title="Power Consumption (Joules)",
    hovermode="closest",
    width=1200,
    height=700,
    template="plotly_white",
)

# Save to HTML
fig.write_html(str(output_path))
print(f"Plot saved to {output_path}")
print(f"Loaded {len(df)} data points")

# Open in browser
webbrowser.open(f"file://{output_path.resolve()}")
print(f"Opening plot in browser...")
