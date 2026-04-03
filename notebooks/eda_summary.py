# # ======================================
# Programmer: Lester Aranca
# Date Programmed: April 03, 2026

# Tasks performed:
# Load the processed dataset
# Generate summary statistics (mean, median, etc.)

# Outputs:
# outputs/basic_stats.csv
# # ======================================

import pandas as pd
import os
from pathlib import Path

# Get paths
BASE_DIR = Path(os.path.abspath('')).resolve().parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cleaned_stroke_data.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "basic_stats.csv"

# Get processed data
df = pd.read_csv(PROCESSED_PATH)

# Attributes for EDA
attrs = ['age', 'avg_glucose_level', 'bmi']

rows = []
for attr in attrs:
    rows.append([
        attr, 
        df[attr].mean(), 
        df[attr].median(), 
        df[attr].mode().values, 
        df[attr].max(),
        df[attr].min(),
        df[attr].std(), 
        df[attr].var()])

# Get summary data
sdf = pd.DataFrame(rows, columns=['Attribute', 'Mean', 'Median', 'Mode', 'Maximum', 'Minimum', 'Standard Deviation', 'Variance'])

# Export summary to CSV
sdf.to_csv(OUTPUT_PATH)