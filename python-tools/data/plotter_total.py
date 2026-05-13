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
data = []
for json_file in sorted(data_path.glob("*.json")):
    try:
        with open(json_file, "r") as f:
            json_data = json.load(f)
            block_dim = json_data.get("blockDim", {})
            grid_dim = json_data.get("gridDim", {})
            total_blocks = (block_dim.get("x", 1) * block_dim.get("y", 1) * block_dim.get("z", 1) * 
                           grid_dim.get("x", 1) * grid_dim.get("y", 1) * grid_dim.get("z", 1))
            data.append({
                "file": json_file.stem,
                "powerConsumptionJoules": json_data.get("powerConsumptionJoules"),
                "totalInstructions": json_data.get("totalInstructions"),
                "totalBlocks": total_blocks,
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
        color=df["powerConsumptionJoules"],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Power (Joules)"),
        line=dict(width=0.5, color="white")
    ),
    text=df["file"],
    hovertemplate="<b>%{text}</b><br>Instructions: %{x}<br>Power: %{y:.2f} J<extra></extra>",
))

fig.update_layout(
    title="Power Consumption vs Total Instructions",
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
