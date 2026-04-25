import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.database.db_utils import get_connection, bulk_insert

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "well_sensor_data.csv")

SENSOR_COLS = [
    "pump_pressure_psi", "flow_rate_bpd", "vibration_mm_s", "temperature_f",
    "torque_ft_lbs", "motor_current_amp", "oil_viscosity_cp", "gas_oil_ratio", "rpm",
]


def load_and_validate(path: str) -> pd.DataFrame:
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = [
        "record_id", "well_id", "timestamp", "depth_ft",
        *SENSOR_COLS,
        "hours_since_last_maintenance", "cumulative_operating_hours",
        "maintenance_required", "failure_type", "days_to_failure",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = df[SENSOR_COLS].isna().sum().sum()
    for col in SENSOR_COLS:
        df[col] = (
            df.groupby("well_id")[col]
            .transform(lambda s: s.fillna(s.rolling(window=24, min_periods=1).median()))
        )
        df[col] = df[col].fillna(df[col].median())
    after = df[SENSOR_COLS].isna().sum().sum()
    print(f"Imputed {before - after:,} nulls ({before:,} → {after:,}).")
    return df


def insert_wells(df: pd.DataFrame):
    wells = df.groupby("well_id")["depth_ft"].first().reset_index()
    rows = [(row.well_id, int(row.depth_ft), "2022-01-01", f"Field-{row.well_id[-3:]}")
            for _, row in wells.iterrows()]
    bulk_insert("wells", ["well_id", "depth_ft", "install_date", "location"], rows)


def insert_sensor_readings(df: pd.DataFrame, batch_size: int = 5000):
    cols = [
        "record_id", "well_id", "timestamp", "depth_ft",
        *SENSOR_COLS,
        "hours_since_last_maintenance", "cumulative_operating_hours",
        "maintenance_required", "failure_type", "days_to_failure",
    ]
    total = 0
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        rows = []
        for _, r in chunk.iterrows():
            dtf = None if pd.isna(r.days_to_failure) else float(r.days_to_failure)
            rows.append((
                int(r.record_id), r.well_id, r.timestamp, int(r.depth_ft),
                float(r.pump_pressure_psi), float(r.flow_rate_bpd), float(r.vibration_mm_s),
                float(r.temperature_f), float(r.torque_ft_lbs), float(r.motor_current_amp),
                float(r.oil_viscosity_cp), float(r.gas_oil_ratio), float(r.rpm),
                float(r.hours_since_last_maintenance), float(r.cumulative_operating_hours),
                int(r.maintenance_required), r.failure_type, dtf,
            ))
        bulk_insert("sensor_readings", cols, rows)
        total += len(rows)
        print(f"  Progress: {total:,} / {len(df):,} rows")
    return total


def run_ingestion():
    df = load_and_validate(DATA_PATH)
    df = impute_missing(df)
    dupes = df.duplicated("record_id").sum()
    df = df.drop_duplicates("record_id")
    print(f"Dropped {dupes} duplicate record_ids.")

    print("Inserting well metadata...")
    insert_wells(df)

    print("Inserting sensor readings (this may take a few minutes)...")
    total = insert_sensor_readings(df)

    print(f"\nIngestion complete: {total:,} rows inserted.")
    print(f"Maintenance events: {df['maintenance_required'].sum():,} ({df['maintenance_required'].mean()*100:.1f}%)")
    print(f"Failure types:\n{df['failure_type'].value_counts().to_string()}")


if __name__ == "__main__":
    run_ingestion()
