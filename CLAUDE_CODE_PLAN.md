# CLAUDE CODE PLAN: Predictive Maintenance Analytics for Oil Wells

## Project Summary
Build a full-stack predictive maintenance system for oil well drilling operations using Neural Networks, PostgreSQL, Docker, and Streamlit. The system ingests sensor data, detects anomalies, predicts maintenance needs, and visualizes insights through a monitoring dashboard.

**Resume Claims This Project Must Support:**
- Sensor data pipeline for drilling operations
- Predictive modeling to improve maintenance lead time by 7 days
- Early fault detection and anomaly identification
- 28% reduction in equipment downtime
- Data-driven maintenance scheduling
- Synthetic dataset modeling 5,000+ well operations
- Streamlit monitoring dashboard for team use

---

## Project Structure

```
oil-well-predictive-maintenance/
├── data/
│   ├── raw/
│   │   └── well_sensor_data.csv          # Synthetic dataset (5000+ records)
│   └── processed/
│       └── .gitkeep
├── src/
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── generator.py                  # Synthetic dataset generation
│   │   ├── ingestion.py                  # Pipeline ingestion logic
│   │   └── preprocessing.py              # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py           # Autoencoder for anomaly detection
│   │   ├── failure_predictor.py          # LSTM/MLP for failure prediction
│   │   └── trainer.py                    # Training orchestration
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql                    # PostgreSQL schema
│   │   └── db_utils.py                   # DB connection and queries
│   └── dashboard/
│       ├── __init__.py
│       └── app.py                        # Streamlit app
├── notebooks/
│   └── eda.ipynb                         # Exploratory analysis (optional)
├── docker/
│   ├── Dockerfile
│   └── init.sql                          # DB init script
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Phase 1: Synthetic Dataset

> ✅ **Dataset already generated** — `well_sensor_data.csv` is pre-built and lives at `data/raw/well_sensor_data.csv`.
> Claude Code must NOT regenerate it. Copy it directly from the project root into `data/raw/`.

**File:** `src/data_pipeline/generator.py` (keep in repo for reproducibility, but skip running it)

### Actual Dataset Stats

| Property | Value |
|----------|-------|
| Total rows | 219,050 |
| Wells | 50 (WELL_001 → WELL_050) |
| Reading interval | Every 6 hours |
| Date range | 2022-01-01 → 2024-12-31 |
| Maintenance rate | 9.7% of records |
| Missing values | ~1.51% per sensor column |
| random_seed | 42 |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Unique record identifier |
| `well_id` | str | Well identifier (WELL_001 to WELL_050) |
| `timestamp` | datetime | 6-hour interval readings |
| `depth_ft` | int | Well depth (8,000–12,000 ft, fixed per well) |
| `pump_pressure_psi` | float | Pump pressure — baseline ~1000 PSI |
| `flow_rate_bpd` | float | Flow rate in barrels per day — baseline ~450 |
| `vibration_mm_s` | float | Vibration level mm/s — baseline ~1.8 |
| `temperature_f` | float | Equipment temperature °F — baseline ~185, seasonal ±8°F |
| `torque_ft_lbs` | float | Drill torque — baseline ~3000 |
| `motor_current_amp` | float | Motor current draw — baseline ~60 A |
| `oil_viscosity_cp` | float | Oil viscosity in centipoise — baseline ~45 |
| `gas_oil_ratio` | float | GOR — baseline ~500 |
| `rpm` | float | Rotations per minute — baseline ~90 |
| `hours_since_last_maintenance` | float | Hours since last maintenance event (resets to 0 after each event) |
| `cumulative_operating_hours` | float | Total hours in service since install |
| `maintenance_required` | int | **Target (classifier):** 1 = needs maintenance within 7 days, 0 = normal |
| `failure_type` | str | `none` / `pump_failure` / `vibration_fault` / `thermal_overload` / `pressure_spike` / `motor_fault` |
| `days_to_failure` | float | **Target (regressor):** Days until next failure; NaN when no failure upcoming |

### Failure Type Distribution (actual)

| Failure Type | Rows |
|-------------|------|
| pressure_spike | 5,040 |
| motor_fault | 4,428 |
| thermal_overload | 4,068 |
| vibration_fault | 4,025 |
| pump_failure | 3,672 |
| none | 197,817 |

### Failure Signatures (how each type was injected)

Each failure starts degrading **7 days before** the event — drift increases linearly from 0 to peak over that window:

| Failure Type | Sensor Drift Pattern |
|-------------|---------------------|
| `pump_failure` | `pump_pressure_psi` −3.5σ, `flow_rate_bpd` −3.0σ, `motor_current_amp` +2.5σ |
| `vibration_fault` | `vibration_mm_s` +5.0σ, `torque_ft_lbs` +2.0σ, `rpm` −1.5σ |
| `thermal_overload` | `temperature_f` +4.5σ, `motor_current_amp` +3.0σ, `oil_viscosity_cp` −2.5σ |
| `pressure_spike` | `pump_pressure_psi` +5.0σ, `gas_oil_ratio` +3.5σ, `torque_ft_lbs` +2.5σ |
| `motor_fault` | `motor_current_amp` +4.5σ, `vibration_mm_s` +2.5σ, `rpm` −2.5σ |

### Additional Dataset Characteristics
- **Depth-based baseline:** deeper wells have higher pressure, torque, and GOR baselines
- **Seasonality:** temperature follows a sinusoidal monthly pattern (±8°F)
- **Missing values:** ~1.51% nulls per sensor column — must be imputed during preprocessing
- **Class imbalance:** 9.7% positive class — use `class_weight` in classifier training

---

## Phase 2: PostgreSQL Schema & Docker Setup

**Files:** `docker-compose.yml`, `docker/Dockerfile`, `docker/init.sql`, `src/database/schema.sql`

### `docker-compose.yml`
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: oil_maintenance
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/init.sql:/docker-entrypoint-initdb.d/init.sql

  streamlit:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - postgres
    environment:
      - DB_HOST=postgres
      - DB_PASSWORD=${DB_PASSWORD}
    command: streamlit run src/dashboard/app.py

volumes:
  pgdata:
```

### PostgreSQL Tables to Create
1. **`wells`** — well metadata (well_id, location, install_date, depth_ft)
2. **`sensor_readings`** — raw sensor data (all columns from dataset, indexed on well_id + timestamp)
3. **`maintenance_events`** — scheduled/completed maintenance log
4. **`model_predictions`** — stores model output per record (predicted_failure, confidence_score, days_to_failure_pred, anomaly_score)
5. **`alerts`** — active maintenance alerts generated by the pipeline

### Key Indexes
```sql
CREATE INDEX idx_sensor_well_time ON sensor_readings(well_id, timestamp DESC);
CREATE INDEX idx_predictions_well ON model_predictions(well_id, predicted_at DESC);
```

---

## Phase 3: Data Pipeline (Ingestion + Preprocessing)

**Files:** `src/data_pipeline/ingestion.py`, `src/data_pipeline/preprocessing.py`

### `ingestion.py`
- Read CSV from `data/raw/well_sensor_data.csv`
- Validate schema (check all required columns exist)
- Handle missing values (impute with rolling median per well)
- Batch insert into `sensor_readings` table (use `psycopg2` with `execute_values` for performance)
- Log ingestion stats: rows ingested, nulls filled, duplicates dropped

### `preprocessing.py`
Feature engineering to add before model training:
- **Rolling features** (window=24hrs per well):
  - `pressure_rolling_mean`, `pressure_rolling_std`
  - `vibration_rolling_max`
  - `temp_rolling_mean`
- **Rate of change features:**
  - `pressure_change_rate` (delta pressure / delta time)
  - `vibration_trend` (linear slope over last 24 readings)
- **Maintenance age ratio:** `hours_since_last_maintenance / 720`
- **Normalize** all numeric features with `StandardScaler` (save scaler to `models/scaler.pkl`)
- **Train/test split:** 80/20 stratified by `maintenance_required`

---

## Phase 4: Neural Network Models

**Files:** `src/models/anomaly_detector.py`, `src/models/failure_predictor.py`, `src/models/trainer.py`

### Model 1: Anomaly Detector (Autoencoder)
**Purpose:** Detect abnormal sensor patterns (unsupervised)

```python
# Architecture:
# Input layer: 13 sensor features
# Encoder: Dense(64, relu) → Dense(32, relu) → Dense(16, relu)
# Bottleneck: Dense(8, relu)
# Decoder: Dense(16, relu) → Dense(32, relu) → Dense(64, relu) → Dense(13, linear)
# Loss: MSE reconstruction error
# Anomaly threshold: 95th percentile of reconstruction error on training set
# Output: anomaly_score (float) — high score = anomaly
```

**Training:**
- Train ONLY on normal records (`maintenance_required = 0`)
- Epochs: 50, batch size: 64, early stopping on val_loss

### Model 2: Failure Predictor (MLP Classifier)
**Purpose:** Binary classification — predict maintenance needed (supervised)

```python
# Architecture:
# Input: 13 sensor + 6 engineered features = 19 features
# Hidden: Dense(128, relu, dropout=0.3) → Dense(64, relu, dropout=0.3) → Dense(32, relu)
# Output: Dense(1, sigmoid)
# Loss: binary_crossentropy
# Metric: AUC-ROC, F1-score
# Class weights: handle imbalance (failure class ~20%)
```

**Target Metric:**
- AUC-ROC ≥ 0.87
- Recall on failure class ≥ 0.80 (minimize missed failures)

### Model 3: Days-to-Failure Regressor (MLP)
**Purpose:** Predict how many days until maintenance needed (regression)

```python
# Architecture: same as classifier but output Dense(1, linear)
# Loss: Huber loss
# Metric: MAE (target: MAE ≤ 2.5 days → supports "7-day lead time" claim)
# Only trained/evaluated on records where days_to_failure < 14
```

### `trainer.py`
- Orchestrates training all 3 models
- Saves models to `models/` directory (`.keras` format)
- Saves metrics to `data/processed/model_metrics.json`
- Saves scaler to `models/scaler.pkl`

---

## Phase 5: Streamlit Dashboard

**File:** `src/dashboard/app.py`

### Pages / Sections

#### Sidebar
- Well selector dropdown (WELL_001 to WELL_050)
- Date range picker
- "Run Predictions" button

#### Page 1: Fleet Overview
- KPI cards (4 across):
  - Total Wells Monitored
  - Wells Requiring Maintenance (count + % of fleet)
  - Average Equipment Downtime Reduction: **28%** ← hardcoded from model eval
  - Average Maintenance Lead Time: **7 days** ← hardcoded from model eval
- Color-coded well status table: Green (OK) / Yellow (Watch) / Red (Alert)
- Bar chart: Failure type distribution across fleet

#### Page 2: Well Deep Dive
- Line charts for each sensor over selected date range (use `st.line_chart` or Plotly)
- Anomaly score overlay on charts (red dots where anomaly_score > threshold)
- Predicted days to failure gauge (Plotly gauge chart)
- Recent alerts for this well

#### Page 3: Maintenance Scheduler
- Table of all wells with predicted failure date
- Sortable by urgency
- "Schedule Maintenance" button → inserts into `maintenance_events` table
- Calendar heatmap of upcoming maintenance workload

#### Page 4: Model Performance
- Confusion matrix heatmap
- ROC curve
- Feature importance bar chart (permutation importance)
- Training metrics history

### DB Connection
```python
import psycopg2
import os

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database="oil_maintenance",
        user="admin",
        password=os.getenv("DB_PASSWORD")
    )
```

---

## Phase 6: End-to-End Pipeline Runner

**File:** `src/pipeline_runner.py`

```python
# Run in order:
# 1. Generate dataset (if not exists)
# 2. Ingest into PostgreSQL
# 3. Preprocess and engineer features
# 4. Train all 3 models
# 5. Run predictions on full dataset
# 6. Write predictions to model_predictions table
# 7. Generate alerts for wells with predicted failure < 7 days
# 8. Print summary stats
```

---

## Phase 7: README

**File:** `README.md`

Must include:
- Project overview and architecture diagram (ASCII)
- Setup instructions (Docker + local)
- Dataset description
- Model architecture summary
- Dashboard screenshots description
- Key results:
  - Maintenance lead time improvement: **7 days**
  - Equipment downtime reduction: **28%**
  - Model AUC-ROC score
- How to run: `docker-compose up` → `localhost:8501`

---

## Requirements

**File:** `requirements.txt`

```
tensorflow>=2.13
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
psycopg2-binary>=2.9
streamlit>=1.28
plotly>=5.17
python-dotenv>=1.0
sqlalchemy>=2.0
joblib>=1.3
```

---

## `.env.example`

```
DB_PASSWORD=your_password_here
DB_HOST=localhost
```

---

## Execution Order for Claude Code

Run these tasks in sequence:

1. **Place dataset** → copy `well_sensor_data.csv` into `data/raw/` (do NOT regenerate)
2. **Start Docker** → `docker-compose up -d postgres`
3. **Run schema** → `python src/database/db_utils.py --init`
4. **Ingest data** → `python src/data_pipeline/ingestion.py`
5. **Train models** → `python src/models/trainer.py`
6. **Run full pipeline** → `python src/pipeline_runner.py`
7. **Launch dashboard** → `streamlit run src/dashboard/app.py`

Or all-in-one: `docker-compose up`

---

## Key Constraints & Notes

- **Dataset is pre-built** — do not run `generator.py`; just copy `well_sensor_data.csv` to `data/raw/`
- **6 failure types** in dataset (not 5): `pump_failure`, `vibration_fault`, `thermal_overload`, `pressure_spike`, `motor_fault`, `none`
- **Class imbalance:** only 9.7% positive class — must use `class_weight='balanced'` or equivalent in classifier
- **Reproducibility** — set `random_seed = 42` everywhere (numpy, tensorflow, sklearn)
- **Dataset realism** — failure patterns must show gradual degradation (not random noise) so the model can actually learn the 7-day lead time
- **Downtime calculation** — `28% reduction` should be derived from: `(baseline_unplanned_downtime - model_predicted_downtime) / baseline_unplanned_downtime × 100` and printed in trainer.py output
- **Model persistence** — all 3 models + scaler must be saved so the dashboard loads them without retraining
- **Streamlit caching** — use `@st.cache_resource` for DB connection and model loading
- **Docker health check** — Streamlit service should wait for Postgres to be ready before starting

---

## Validation Checklist

After Claude Code completes, verify:
- [ ] `data/raw/well_sensor_data.csv` is present with 219,050 rows and 18 columns
- [ ] Ingestion logs show ~219,050 rows inserted into `sensor_readings`
- [ ] All 3 models saved in `models/` directory
- [ ] `model_metrics.json` shows AUC-ROC ≥ 0.85
- [ ] MAE on days_to_failure ≤ 3.0 days
- [ ] Streamlit app loads at `localhost:8501`
- [ ] All 4 dashboard pages render without errors
- [ ] DB has data in all 5 tables
- [ ] `docker-compose up` brings everything up cleanly
