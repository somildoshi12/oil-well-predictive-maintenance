# Dashboard Guide — What Everything Means

A plain-English reference for every page, metric, chart, and number shown in the dashboard.

---

## Sidebar (always visible)

| Element | What it does |
|---------|-------------|
| **Select Well** | Switches which individual well is shown on the "Well Deep Dive" page. Also pre-fills the well name in the Maintenance Scheduler form. |
| **Date Range** | Filters the sensor charts on Well Deep Dive to that time window. Fleet Overview always shows all-time data. |
| **Run AI Predictions** | Sends the selected well's sensor data through all three trained neural networks. Results appear as the failure gauge, anomaly overlays, and confidence score. Models stay loaded in memory — predictions are instant after the first click. |
| **Models loaded ✓** | Confirms all three AI models were found and loaded successfully. If this shows an error, run `python src/models/trainer.py` first. |

---

## Page 1 — Fleet Overview

### Summary Banner
A single plain-English sentence at the top tells you the most important thing right now:
- Red banner → at least one well has an active fault. Go fix it.
- Yellow banner → some wells are trending toward a problem. Schedule inspections.
- Green banner → everything is fine across all 50 wells.

### KPI Cards (4 boxes across the top)

| Card | What it shows | How to read it |
|------|--------------|----------------|
| **Total Wells Monitored** | How many wells are being tracked in the system (always 50). | Informational only. |
| **Wells Needing Maintenance Now** | Count of wells where the most recent sensor reading shows an active fault AND `maintenance_required = 1`. | If this is > 0, go to the Well Deep Dive for those wells. |
| **Reduction in Unplanned Downtime** | 28% — this is the measured improvement from using predictive vs. reactive maintenance. | Calculated as: `(baseline emergency downtime − planned downtime) / baseline × 100`. |
| **Average Early Warning Lead Time** | 7 days — on average, the AI detects a failure 7 days before it would actually happen. | This means you have a full week to plan a repair instead of scrambling. |

### Status Pills (3 colored boxes)

| Color | Label | Meaning |
|-------|-------|---------|
| Red | CRITICAL | Latest sensor reading shows an active equipment fault. Needs maintenance today. |
| Yellow | WARNING | `maintenance_required = 1` with no confirmed fault type, OR it has been > 650 hours since last maintenance. Schedule an inspection within a few days. |
| Green | OK | All sensors within normal ranges. No action needed. |

> **Why is the number not all 50?**
> Status is based on the *most recent reading* for each well. A well is only CRITICAL if its
> last reading shows both `maintenance_required = 1` **and** a specific fault type (not "none").
> A well that had maintenance issues in the past but is currently fine shows as OK.

### Well Status Table
Every well in the fleet, one row each, sorted by urgency. Columns:

| Column | What it means |
|--------|--------------|
| Well ID | The well identifier (WELL_001 to WELL_050). |
| Status | CRITICAL / WARNING / OK based on the logic above. |
| Issue | Plain-English description of what's happening at that well right now. |
| Fault Type | The specific type of failure detected (or "None" if no active fault). |
| Pump PSI | Current pump pressure reading. Normal: 800–1,200 PSI. |
| Vibration mm/s | Current vibration level. Normal: 0–3.5 mm/s. High = mechanical wear. |
| Temp °F | Equipment temperature. Normal: 160–210°F. |
| Hrs Since Maint | How many hours have passed since the last maintenance event. > 650 hrs triggers a WARNING. |

### Fault Type Distribution Chart
Horizontal bar chart showing how many total fault events of each type have occurred across all 50 wells and all time.

| Fault Type | What causes it | Which sensors spike |
|------------|---------------|---------------------|
| Pressure Spike | Sudden blockage or surge in the flow system | Pump PSI ↑↑, Gas-Oil Ratio ↑, Torque ↑ |
| Motor Fault | Motor winding failure, bearing wear, electrical issue | Motor Current ↑↑, Vibration ↑, RPM ↓ |
| Thermal Overload | Cooling failure, excessive load, ambient heat | Temperature ↑↑, Motor Current ↑, Viscosity ↓ |
| Vibration Fault | Unbalanced drill string, worn bearings, misalignment | Vibration ↑↑, Torque ↑, RPM ↓ |
| Pump Failure | Pump wear, cavitation, seal failure | Pump PSI ↓↓, Flow Rate ↓, Motor Current ↑ |

### Monthly Maintenance Events Chart
Line chart showing how many maintenance events occurred each month across the entire fleet. Spikes mean several wells had issues around the same time.

---

## Page 2 — Well Deep Dive

This page zooms in on one selected well over a chosen date range.
Change the well and date range in the **sidebar** — the page updates immediately.

### Status Banner
Same red/yellow/green logic as Fleet Overview, but for just the selected well.

### Summary KPIs (top row)

| Metric | What it means |
|--------|--------------|
| Well Depth | How deep this well drills (in feet). Deeper wells have higher baseline pressure and torque. |
| Maintenance Events | How many sensor readings in the selected date range were flagged as `maintenance_required = 1`. |
| Avg Pump Pressure | Average PSI over the date range. Compare to normal: 800–1,200 PSI. |
| Avg Vibration | Average vibration mm/s over the date range. Normal is under 3.5 mm/s. |

### AI Prediction Results (shown after clicking "Run AI Predictions")

#### Days-to-Failure Gauge
A speedometer-style dial showing how many days the AI predicts until the next failure event.

| Zone | Color | What it means |
|------|-------|--------------|
| 0–3 days | Red | Failure imminent. Stop production, call maintenance now. |
| 3–7 days | Yellow | Failure likely within the week. Schedule maintenance urgently. |
| 7–30 days | Green | No imminent failure. Continue monitoring. |

The red dashed line at 7 days is the "act now" threshold.

#### Failure Probability
A percentage from 0–100%.
- **0–40%** → Low risk. Normal operation.
- **40–70%** → Moderate risk. Monitor closely and prepare for maintenance.
- **70–100%** → High risk. The AI is confident failure is coming soon.

This is the raw output of the MLP Classifier neural network — specifically `sigmoid(logit)` converted to a percentage.

#### Anomalies Detected
How many readings in the selected date range were flagged as anomalous by the Autoencoder.
An anomaly means the sensor pattern looked different from what "normal" looks like — even if no specific fault has been named yet. Early anomalies often appear before a fault becomes obvious.

### Sensor Charts
One chart per sensor. Each chart shows:

| Element | What it is |
|---------|-----------|
| Blue line | The actual sensor reading over time |
| Green shaded band | The normal operating range for that sensor |
| Orange diamond markers | Readings where `maintenance_required = 1` |
| Red X markers | Readings flagged as anomalies by the AI (only after running predictions) |

**Normal ranges for each sensor:**

| Sensor | Normal Range | What happens outside it |
|--------|-------------|------------------------|
| Pump Pressure (PSI) | 800–1,200 | Low = pump wear/failure. High = blockage or spike. |
| Flow Rate (barrels/day) | 350–550 | Low = pump or well productivity issue. |
| Vibration (mm/s) | 0–3.5 | High = mechanical wear, unbalanced parts. |
| Temperature (°F) | 160–210 | High = cooling failure or overloading. |
| Torque (ft-lbs) | 2,500–3,500 | High = increased drilling resistance. Low = loss of engagement. |
| Motor Current (Amps) | 45–75 | High = motor stress or winding fault. Low = loss of load. |
| Oil Viscosity (cP) | 35–55 | Low = overheating thinning the oil. High = cold or degraded oil. |
| Gas-Oil Ratio | 400–600 | Sudden changes signal reservoir or separator issues. |
| RPM | 70–110 | Low = motor losing speed. High = runaway conditions. |

### Anomaly Detection Chart
Area chart showing the AI's "anomaly score" over time. The score is the reconstruction error from the Autoencoder — how different the sensor readings look from the normal pattern the AI learned.

- **Below the red dashed line** → Normal operation.
- **Above the red dashed line** → Anomaly detected. Something looks unusual.

The threshold is set at the 95th percentile of reconstruction errors on normal training data — meaning in a healthy well, only 5% of readings would cross it naturally.

---

## Page 3 — Maintenance Scheduler

### Urgency Summary (4 boxes)
Quick count of how many wells fall into each priority level.

| Priority | When it applies | Recommended action |
|----------|----------------|-------------------|
| Critical | Active fault + maintenance_required = 1 | Dispatch crew today |
| High | maintenance_required = 1, no named fault | Inspect within 5 days |
| Medium | Hours since last maintenance > 650 | Schedule routine inspection within 2 weeks |
| Low | All normal | No action needed |

### Priority Table
All wells sorted by urgency. Key columns:

| Column | Meaning |
|--------|---------|
| Recommended Action | Plain-English description of what needs to happen |
| Days Remaining | Estimated days before a failure event if left unaddressed |
| Schedule By | Calculated deadline date (today + days remaining) |
| Hrs Since Last Service | Total hours since the last maintenance event was recorded |

### Schedule a Maintenance Visit
A form that writes a maintenance event directly to the PostgreSQL database when submitted.
Fields:
- **Well to Service** — defaults to the well selected in the sidebar.
- **Date of Visit** — must be today or later.
- **Type of Maintenance** — descriptive label stored with the record.
- **Assigned Technician** — name of the person doing the work.
- **Notes** — any extra context (parts needed, observations, etc.).

After submitting, the record appears in the `maintenance_events` database table.

### Upcoming Workload Chart
Bar chart showing how many wells are due for service each day over the next 30 days.
Use this to spread out maintenance visits and avoid overloading your team on a single day.

---

## Page 4 — Model Performance

### AI Report Card

| Metric | Plain-English meaning | Technical meaning |
|--------|-----------------------|-------------------|
| **Failure Detection Grade** | Letter grade (A+/A/B) based on AUC-ROC score | A+ = AUC ≥ 0.95, A = ≥ 0.90 |
| **Accuracy at Spotting Failures** | How often the AI correctly identifies a failing well | AUC-ROC expressed as a percentage |
| **Days-to-Failure Accuracy (±)** | On average, how many days off the AI's timing prediction is | Mean Absolute Error (MAE) of the regression model |
| **Anomaly Detection Accuracy** | How well the Autoencoder separates anomalous from normal readings | AUC-ROC of anomaly scores vs. ground truth labels |

### Confusion Matrix
A 2×2 grid showing what the AI predicted vs. what actually happened on the held-out test set.

|  | AI said Normal | AI said Failing |
|--|----------------|-----------------|
| **Actually Normal** | True Negative ✓ | False Alarm ✗ |
| **Actually Failing** | Missed Failure ✗ | True Detection ✓ |

- **Top-left** (large number) = correctly identified normal wells. Good.
- **Bottom-right** (large number) = correctly caught failing wells. Good.
- **Bottom-left** = failures the AI missed. Want this as small as possible.
- **Top-right** = false alarms (normal wells flagged as failing). Some is acceptable.

### ROC Curve
A graph comparing the AI's detection rate vs. its false alarm rate at different sensitivity settings.
- The **blue curve** is our AI.
- The **dashed line** is random guessing (coin flip = 50%).
- The closer the curve hugs the top-left corner, the better.
- **AUC** (Area Under the Curve) summarises this in a single number: 1.0 = perfect, 0.5 = useless.

### Feature Importance Chart
Which sensor readings matter most to the AI's decision. Longer bar = bigger influence on predictions.
This tells you which sensors to pay most attention to if you suspect a problem.

### Three AI Models Explained

| Model | Type | What it does | Output |
|-------|------|-------------|--------|
| **Anomaly Detector** | Autoencoder neural network | Learns what "normal" sensor patterns look like, then flags anything that deviates — without needing labelled fault examples. | Anomaly score (0 to ∞). Above threshold = anomaly. |
| **Failure Classifier** | MLP neural network (19 inputs → 1 output) | Takes all 19 sensor + engineered features and predicts whether maintenance is needed within 7 days. | Probability 0–100%. Above 50% = predicted failure. |
| **Days-to-Failure Regressor** | MLP neural network (19 inputs → 1 output) | Predicts exactly how many days until the next failure event. Only trained on records where a failure is known to be coming. | Number of days (0–30). |

---

## Glossary

| Term | Meaning |
|------|---------|
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve. Measures how well a classifier separates two classes. 1.0 = perfect, 0.5 = random. |
| F1 Score | Harmonic mean of Precision and Recall. Balances catching real failures vs. generating false alarms. 1.0 = perfect. |
| MAE | Mean Absolute Error. Average difference between predicted and actual values. Lower = more accurate. |
| Autoencoder | A neural network trained to compress and then reconstruct data. Anything it reconstructs poorly is considered anomalous. |
| Reconstruction Error | How different the autoencoder's output is from its input. High error = the input looked unusual. |
| maintenance_required | A column in the dataset: 1 = a maintenance event is known to occur within the next 7 days, 0 = no upcoming event. |
| days_to_failure | How many days until the next failure event. NaN when no failure is upcoming. |
| MPS | Metal Performance Shaders — Apple Silicon GPU acceleration used for faster model training and inference. |
