import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load data
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
linear_df = pd.read_csv(os.path.join(base_dir, 'data/miscellaneous/linear_model_results.csv'))
single_df = pd.read_csv(os.path.join(base_dir, 'data/miscellaneous/single_results.csv'))

# Calculate Absolute Percentage Error (APE) for both
linear_df['linear_ape'] = np.abs(linear_df['predictedPowerConsumptionJoules'] - linear_df['powerConsumptionJoules']) / linear_df['powerConsumptionJoules'] * 100
single_df['single_ape'] = np.abs(single_df['predictedPowerConsumptionJoules'] - single_df['powerConsumptionJoules']) / single_df['powerConsumptionJoules'] * 100

# Merge on kernelName
merged = pd.merge(linear_df[['kernelName', 'linear_ape']], single_df[['kernelName', 'single_ape']], on='kernelName')

# Calculate improvement (positive means single_ape is lower than linear_ape)
merged['improvement'] = merged['linear_ape'] - merged['single_ape']
merged = merged.sort_values(by='improvement')

# Create plot showing improvement
fig = px.bar(
    merged, 
    x='kernelName', 
    y='improvement',
    title='Improvement in Model Accuracy (Linear Model Error % - Single Occurrences Model Error %)',
    labels={'kernelName': 'Kernel Name', 'improvement': 'Improvement (Percentage Points)'},
    color='improvement',
    color_continuous_scale=px.colors.diverging.RdYlGn,
    color_continuous_midpoint=0
)

# Optional: Add a grouped bar chart option
# fig2 = go.Figure(data=[
#     go.Bar(name='Linear Model', x=merged['kernelName'], y=merged['linear_ape']),
#     go.Bar(name='Single Model', x=merged['kernelName'], y=merged['single_ape'])
# ])
# fig2.update_layout(barmode='group', title='Linear vs Single Model APE (%)')
# fig2.show()

if __name__ == "__main__":
    fig.show()
