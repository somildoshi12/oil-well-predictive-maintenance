"""
Synthetic dataset generator — kept for reproducibility.
Dataset is already pre-built at data/raw/well_sensor_data.csv (219,050 rows).
Do NOT run this script; the CSV is the source of truth.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_WELLS = 50
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
FREQ = "6H"


def generate_dataset(output_path: str):
    np.random.seed(RANDOM_SEED)
    print("Dataset already exists at data/raw/well_sensor_data.csv — skipping generation.")
    print(f"Would write to: {output_path}")
