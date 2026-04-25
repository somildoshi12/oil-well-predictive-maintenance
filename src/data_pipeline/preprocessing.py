import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

SENSOR_COLS = [
    "pump_pressure_psi", "flow_rate_bpd", "vibration_mm_s", "temperature_f",
    "torque_ft_lbs", "motor_current_amp", "oil_viscosity_cp", "gas_oil_ratio", "rpm",
]

ENGINEERED_COLS = [
    "pressure_rolling_mean", "pressure_rolling_std",
    "vibration_rolling_max", "temp_rolling_mean",
    "pressure_change_rate", "vibration_trend",
    "maintenance_age_ratio",
]

ALL_FEATURES = SENSOR_COLS + [
    "hours_since_last_maintenance", "cumulative_operating_hours",
    "depth_ft",
] + ENGINEERED_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["well_id", "timestamp"]).copy()

    grp = df.groupby("well_id")

    df["pressure_rolling_mean"] = grp["pump_pressure_psi"].transform(
        lambda s: s.rolling(window=4, min_periods=1).mean()
    )
    df["pressure_rolling_std"] = grp["pump_pressure_psi"].transform(
        lambda s: s.rolling(window=4, min_periods=1).std().fillna(0)
    )
    df["vibration_rolling_max"] = grp["vibration_mm_s"].transform(
        lambda s: s.rolling(window=4, min_periods=1).max()
    )
    df["temp_rolling_mean"] = grp["temperature_f"].transform(
        lambda s: s.rolling(window=4, min_periods=1).mean()
    )
    df["pressure_change_rate"] = grp["pump_pressure_psi"].transform(
        lambda s: s.diff().fillna(0)
    )

    def linear_slope(s, window=24):
        def _slope(vals):
            if len(vals) < 2:
                return 0.0
            x = np.arange(len(vals))
            return np.polyfit(x, vals, 1)[0]
        return s.rolling(window=window, min_periods=2).apply(_slope, raw=True).fillna(0)

    df["vibration_trend"] = grp["vibration_mm_s"].transform(
        lambda s: linear_slope(s, window=24)
    )
    df["maintenance_age_ratio"] = df["hours_since_last_maintenance"] / 720.0

    return df


def load_and_prepare(csv_path: str, scaler_path: str = None, fit_scaler: bool = True):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    for col in SENSOR_COLS + ["hours_since_last_maintenance", "cumulative_operating_hours"]:
        df[col] = df.groupby("well_id")[col].transform(
            lambda s: s.fillna(s.rolling(24, min_periods=1).median())
        )
        df[col] = df[col].fillna(df[col].median())

    df = engineer_features(df)

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values
    y_clf = df["maintenance_required"].values
    y_reg = df["days_to_failure"].values

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

    return df, X_scaled, y_clf, y_reg, feature_cols


def split_data(X, y_clf, y_reg):
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, stratify=y_clf, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Positive rate — Train: {yc_train.mean():.3f}  Test: {yc_test.mean():.3f}")
    return X_train, X_test, yc_train, yc_test, yr_train, yr_test
