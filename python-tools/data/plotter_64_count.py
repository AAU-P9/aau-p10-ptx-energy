from pathlib import Path
import json
import webbrowser
import pandas as pd
import plotly.graph_objects as go

data_path = Path("/home/rasmus/aau-p10-ptx-energy/data/generates")
output_path = data_path.parent / "scatter_plot_64_count.html"

# Load all JSON files and extract data
data = []

def count_64_in_instructions(instruction_occurrences):
    """Count how many instruction types contain '64' in their name."""
    count = 0
    for instruction_name in instruction_occurrences.keys():
        if "64" in instruction_name:
            count += 1
    return count

# Extract data from JSON files
for json_file in sorted(data_path.glob("*.json")):
    try:
        with open(json_file, "r") as f:
            json_data = json.load(f)
            
            instruction_occurrences = json_data.get("instructionOccurrences", {})
            count_64 = count_64_in_instructions(instruction_occurrences)
            
            data.append({
                "file": json_file.stem,
                "powerConsumptionJoules": json_data.get("powerConsumptionJoules"),
                "count64": count_64,
                "totalInstructions": json_data.get("totalInstructions"),
            })
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {json_file}: {e}")

# Create DataFrame
df = pd.DataFrame(data)

# Create scatter plot
fig = go.Figure(data=go.Scatter(
    x=df["count64"],
    y=df["powerConsumptionJoules"],
    mode="markers",
    marker=dict(
        size=8,
        color=df["totalInstructions"],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Total Instructions"),
        line=dict(width=0.5, color="white")
    ),
    text=df.apply(lambda row: f"{row['file']}<br>Count of 64: {row['count64']}<br>Total Instructions: {row['totalInstructions']}", axis=1),
    hovertemplate="<b>%{text}</b><br>Power: %{y:.2f} J<extra></extra>",
))

fig.update_layout(
    title="Power Consumption vs Count of '64' in Instruction Names (colored by total instructions)",
    xaxis_title="Count of '64' in Instruction Types",
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
print(f"\nData distribution:")
print(df[["count64", "powerConsumptionJoules", "totalInstructions"]].describe())

# Open in browser
webbrowser.open(f"file://{output_path.resolve()}")
print(f"Opening plot in browser...")
