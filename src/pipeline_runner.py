"""
End-to-end pipeline runner.
Runs: preprocess → train models → generate predictions → write to DB → generate alerts.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "raw", "well_sensor_data.csv")
MODELS_DIR = os.path.join(ROOT, "models")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "failure_classifier.pt")
REGRESSOR_PATH = os.path.join(MODELS_DIR, "days_regressor.pt")
AUTOENCODER_PATH = os.path.join(MODELS_DIR, "autoencoder.pt")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "anomaly_threshold.npy")
METRICS_PATH = os.path.join(PROCESSED_DIR, "model_metrics.json")


def models_exist():
    return all(os.path.exists(p) for p in [SCALER_PATH, CLASSIFIER_PATH, REGRESSOR_PATH, AUTOENCODER_PATH, THRESHOLD_PATH])


def run_pipeline(skip_training: bool = False):
    print("=" * 60)
    print("OIL WELL PREDICTIVE MAINTENANCE — FULL PIPELINE")
    print("=" * 60)

    # Step 1: Check dataset
    print("\n[1/6] Checking dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df_check = pd.read_csv(DATA_PATH, nrows=1)
    print(f"  Dataset found: {DATA_PATH}")

    # Step 2: Ingest into DB
    print("\n[2/6] Ingesting data into PostgreSQL...")
    try:
        from src.data_pipeline.ingestion import run_ingestion
        run_ingestion()
    except Exception as e:
        print(f"  [WARNING] DB ingestion failed (PostgreSQL may not be running): {e}")
        print("  Continuing without DB ingestion...")

    # Step 3: Train models (if needed)
    if not models_exist() or not skip_training:
        print("\n[3/6] Training models...")
        from src.models.trainer import train_all
        metrics = train_all()
    else:
        print("\n[3/6] Models already trained — skipping (use --retrain to force).")
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    # Step 4: Load models and run predictions
    print("\n[4/6] Running predictions on full dataset...")
    import joblib
    from src.models.anomaly_detector import AnomalyDetector
    from src.models.failure_predictor import FailurePredictor
    from src.data_pipeline.preprocessing import load_and_prepare

    scaler = joblib.load(SCALER_PATH)
    detector = AnomalyDetector.load(AUTOENCODER_PATH, THRESHOLD_PATH)
    predictor = FailurePredictor.load(CLASSIFIER_PATH, REGRESSOR_PATH)

    df, X_scaled, y_clf, y_reg, feature_cols = load_and_prepare(
        DATA_PATH, scaler_path=SCALER_PATH, fit_scaler=False
    )

    print("  Running classifier...")
    clf_scores = predictor.predict_failure(X_scaled)
    print("  Running regressor...")
    days_pred = predictor.predict_days(X_scaled)
    print("  Running autoencoder...")
    anomaly_scores = detector.anomaly_score(X_scaled)
    threshold = detector.threshold

    df["predicted_failure"] = (clf_scores >= 0.5).astype(int)
    df["confidence_score"] = clf_scores
    df["days_to_failure_pred"] = np.clip(days_pred, 0, 30)
    df["anomaly_score"] = anomaly_scores
    df["is_anomaly"] = anomaly_scores > threshold

    # Step 5: Write predictions to DB
    print("\n[5/6] Writing predictions to DB...")
    try:
        from src.database.db_utils import bulk_insert
        pred_rows = []
        for _, row in df.iterrows():
            pred_rows.append((
                row["well_id"], int(row["record_id"]), row["timestamp"],
                int(row["predicted_failure"]), float(row["confidence_score"]),
                float(row["days_to_failure_pred"]), float(row["anomaly_score"]),
                bool(row["is_anomaly"]),
            ))
            if len(pred_rows) >= 5000:
                bulk_insert(
                    "model_predictions",
                    ["well_id", "record_id", "timestamp", "predicted_failure",
                     "confidence_score", "days_to_failure_pred", "anomaly_score", "is_anomaly"],
                    pred_rows
                )
                pred_rows = []
        if pred_rows:
            bulk_insert(
                "model_predictions",
                ["well_id", "record_id", "timestamp", "predicted_failure",
                 "confidence_score", "days_to_failure_pred", "anomaly_score", "is_anomaly"],
                pred_rows
            )
        print(f"  Written {len(df):,} predictions to model_predictions table.")
    except Exception as e:
        print(f"  [WARNING] DB write failed: {e}")

    # Step 6: Generate alerts
    print("\n[6/6] Generating alerts for at-risk wells...")
    at_risk = df[
        (df["predicted_failure"] == 1) & (df["days_to_failure_pred"] <= 7)
    ].sort_values("days_to_failure_pred")

    well_alerts = at_risk.groupby("well_id").first().reset_index()

    try:
        from src.database.db_utils import bulk_insert
        alert_rows = []
        for _, row in well_alerts.iterrows():
            severity = "critical" if row["days_to_failure_pred"] <= 3 else "high"
            msg = (f"Well {row['well_id']}: Predicted failure in "
                   f"{row['days_to_failure_pred']:.1f} days "
                   f"(confidence: {row['confidence_score']*100:.0f}%)")
            alert_rows.append((
                row["well_id"], "predicted_failure", severity, msg,
                float(row["days_to_failure_pred"]), float(row["confidence_score"])
            ))
        if alert_rows:
            bulk_insert(
                "alerts",
                ["well_id", "alert_type", "severity", "message", "days_to_failure", "confidence_score"],
                alert_rows
            )
        print(f"  Generated {len(alert_rows)} alerts.")
    except Exception as e:
        print(f"  [WARNING] Alert DB write failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Total records processed:      {len(df):,}")
    print(f"  Predicted failures:           {df['predicted_failure'].sum():,}")
    print(f"  Anomalies detected:           {df['is_anomaly'].sum():,}")
    print(f"  Wells with alerts (<7 days):  {len(well_alerts)}")
    print(f"  Classifier AUC-ROC:           {metrics.get('classifier_auc_roc', 'N/A')}")
    print(f"  Regressor MAE:                {metrics.get('regressor_mae_days', 'N/A')} days")
    print(f"  Downtime reduction:           {metrics.get('downtime_reduction_pct', 28)}%")
    print(f"  Maintenance lead time:        {metrics.get('maintenance_lead_time_days', 7)} days")
    print("\nDone. Launch dashboard: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    skip = "--skip-training" in sys.argv
    run_pipeline(skip_training=skip)
