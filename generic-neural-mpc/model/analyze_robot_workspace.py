#!/usr/bin/env python3
"""
Analyze the real robot dataset to find min/max values for position and velocity columns.
"""

import pandas as pd
import os

# Path to the real robot dataset
DATASET_PATH = os.path.join("model", "data", "output_exp_2025-07-22_12-23-07.csv")

def analyze_dataset(filepath):
    """Analyze the real robot dataset for position and velocity bounds."""
    
    df = pd.read_csv(filepath)
    
    # Clean column names (remove units in parentheses)
    df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)
    
    # Define the state columns
    state_cols = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    
    # Remove rows with NaN values in the state columns
    df_clean = df[state_cols].dropna()
    
    # Calculate min/max for each state variable
    for col in state_cols:
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        print(f"{col:20s}: min={min_val:8.4f}, max={max_val:8.4f}")

if __name__ == "__main__":
    analyze_dataset(DATASET_PATH)
