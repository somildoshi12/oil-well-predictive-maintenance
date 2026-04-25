"""
Generates pipeline_diagram.png — a visual overview of the full
oil-well predictive maintenance pipeline.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(ROOT, "pipeline_diagram.png")

# ── Colour palette ────────────────────────────────────────────────────────────
C_DATA   = "#1565C0"   # deep blue   – data / storage
C_PROC   = "#2E7D32"   # dark green  – processing
C_MODEL  = "#6A1B9A"   # deep purple – models
C_OUTPUT = "#E65100"   # deep orange – outputs / dashboard
C_ARROW  = "#37474F"   # dark slate  – arrows
C_BG     = "#F5F7FA"   # near white  – canvas background
C_CARD   = "#FFFFFF"   # white       – card fill

fig, ax = plt.subplots(figsize=(22, 14))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis("off")

# ── Helper: rounded box ───────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sublabel="", text_color="white", radius=0.35, fontsize=11):
    fancy = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=2, edgecolor=color, facecolor=C_CARD,
    )
    ax.add_patch(fancy)
    # Top colour bar
    bar = FancyBboxPatch(
        (x - w/2, y + h/2 - 0.38), w, 0.38,
        boxstyle=f"round,pad=0.0,rounding_size=0.1",
        linewidth=0, facecolor=color, clip_on=True,
    )
    ax.add_patch(bar)
    ax.text(x, y + h/2 - 0.19, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=5)
    if sublabel:
        ax.text(x, y - 0.05, sublabel,
                ha="center", va="center", fontsize=8.5,
                color="#455A64", zorder=5, linespacing=1.4)

def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color,
                    lw=2.0, mutation_scale=18,
                    connectionstyle="arc3,rad=0.0",
                ))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                fontsize=8, color=color, style="italic")

def section_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=10,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                      edgecolor=color, linewidth=0))

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(11, 13.35, "Oil Well Predictive Maintenance — Full Pipeline",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1A237E")
ax.text(11, 12.85, "50 Wells · 219,050 Records · 3 Neural Networks · Streamlit Dashboard",
        ha="center", va="center", fontsize=11, color="#546E7A")

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Data Layer  (y ≈ 11.3)
# ═══════════════════════════════════════════════════════════════════════════════
section_label(ax, 1.1, 11.3, "DATA", C_DATA)

box(ax, 3.5, 11.3, 3.8, 1.5, C_DATA, "Raw Sensor Data",
    "well_sensor_data.csv\n219,050 rows · 50 wells\n6-hour intervals · 2022–2024")

box(ax, 8.0, 11.3, 3.6, 1.5, C_PROC, "Data Ingestion",
    "ingestion.py\nSchema validation · Null impute\nBatch insert → PostgreSQL")

box(ax, 12.5, 11.3, 3.6, 1.5, C_PROC, "Feature Engineering",
    "preprocessing.py\nRolling stats · Rate of change\nStandardScaler (19 features)")

box(ax, 17.5, 11.3, 3.6, 1.5, C_DATA, "PostgreSQL DB",
    "5 tables:\nwells · sensor_readings\nmodel_predictions · alerts")

arrow(ax, 5.4, 11.3, 6.2, 11.3)
arrow(ax, 9.8, 11.3, 10.7, 11.3)
arrow(ax, 14.3, 11.3, 15.7, 11.3)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Model Layer  (y ≈ 8.5)
# ═══════════════════════════════════════════════════════════════════════════════
section_label(ax, 1.1, 8.5, "MODELS", C_MODEL)

box(ax, 4.5, 8.5, 4.0, 1.8, C_MODEL, "Anomaly Detector",
    "anomaly_detector.py\nAutoencoder  13→64→32→16→8→…→13\n"
    "Train on normal records only\nOutput: reconstruction error score")

box(ax, 9.8, 8.5, 4.0, 1.8, C_MODEL, "Failure Classifier",
    "failure_predictor.py\nMLP  19→128→64→32→1 (sigmoid)\n"
    "BCEWithLogits + class weights\nOutput: P(maintenance needed)")

box(ax, 15.1, 8.5, 4.0, 1.8, C_MODEL, "Days-to-Failure Regressor",
    "failure_predictor.py\nMLP  19→128→64→32→1 (linear)\n"
    "Huber loss · near-failure records\nOutput: days until failure")

# Arrows from preprocessing → each model
arrow(ax, 12.5, 10.55, 4.5, 9.4,  label="19 features", color=C_MODEL)
arrow(ax, 12.5, 10.55, 9.8, 9.4,  label="19 features", color=C_MODEL)
arrow(ax, 12.5, 10.55, 15.1, 9.4, label="19 features", color=C_MODEL)

# ── Checkpoint banner ─────────────────────────────────────────────────────────
ax.text(11, 7.25,
        ">>  Checkpoints saved every 5 epochs to  checkpoints/  |  "
        "Auto-resume on interrupt  |  --retrain to reset",
        ha="center", va="center", fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#37474F",
                  edgecolor="#37474F", linewidth=0))

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Outputs / Pipeline runner  (y ≈ 5.6)
# ═══════════════════════════════════════════════════════════════════════════════
section_label(ax, 1.1, 5.6, "OUTPUT", C_OUTPUT)

box(ax, 4.5, 5.6, 3.8, 1.6, C_OUTPUT, "Anomaly Alerts",
    "anomaly_score > p95 threshold\nis_anomaly flag per record\nStored in alerts table")

box(ax, 9.0, 5.6, 3.8, 1.6, C_OUTPUT, "Failure Predictions",
    "confidence_score per record\npredicted_failure (0/1)\nStored in model_predictions")

box(ax, 13.5, 5.6, 3.8, 1.6, C_OUTPUT, "Maintenance Schedule",
    "days_to_failure_pred\nUrgency: Critical/High/Medium\nAuto-alerts for <7 days")

box(ax, 18.5, 5.6, 3.0, 1.6, C_DATA, "Model Metrics",
    "model_metrics.json\nAUC-ROC · F1 · MAE\nConfusion matrix")

arrow(ax, 4.5,  9.6, 4.5,  6.4)
arrow(ax, 9.8,  9.6, 9.0,  6.4)
arrow(ax, 15.1, 9.6, 13.5, 6.4)
arrow(ax, 9.0,  4.8, 18.5, 5.05)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Dashboard  (y ≈ 2.8)
# ═══════════════════════════════════════════════════════════════════════════════
section_label(ax, 1.1, 2.8, "DASH", "#00796B")

box(ax, 4.2, 2.8, 3.5, 1.7, "#00796B", "Fleet Overview",
    "KPI cards:\n28% downtime ↓ · 7-day lead\nWell status table (R/Y/G)\nFailure distribution chart")

box(ax, 8.3, 2.8, 3.5, 1.7, "#00796B", "Well Deep Dive",
    "Interactive sensor timelines\nAnomaly overlay (red dots)\nDays-to-failure gauge\nRecent alerts list")

box(ax, 12.4, 2.8, 3.5, 1.7, "#00796B", "Maintenance Scheduler",
    "Priority table by urgency\nSchedule form → DB insert\n30-day workload calendar")

box(ax, 16.5, 2.8, 3.5, 1.7, "#00796B", "Model Performance",
    "Confusion matrix heatmap\nROC curve + AUC\nFeature importance bars\nFull metrics JSON")

# Arrows into dashboard from outputs
arrow(ax, 4.5,  4.8, 4.2,  3.65)
arrow(ax, 9.0,  4.8, 8.3,  3.65)
arrow(ax, 13.5, 4.8, 12.4, 3.65)
arrow(ax, 18.5, 4.8, 16.5, 3.65)

# Streamlit label
ax.text(11, 1.4,
        "streamlit run src/dashboard/app.py  →  localhost:8501",
        ha="center", va="center", fontsize=12, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#00796B",
                  edgecolor="#004D40", linewidth=2))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_DATA,   label="Data / Storage"),
    mpatches.Patch(facecolor=C_PROC,   label="Processing"),
    mpatches.Patch(facecolor=C_MODEL,  label="Neural Network Models"),
    mpatches.Patch(facecolor=C_OUTPUT, label="Model Outputs"),
    mpatches.Patch(facecolor="#00796B",label="Streamlit Dashboard"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=9,
          framealpha=0.9, edgecolor="#B0BEC5",
          bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {OUT}")
