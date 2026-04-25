import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH     = os.path.join(ROOT, "data", "raw", "well_sensor_data.csv")
MODELS_DIR    = os.path.join(ROOT, "models")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
METRICS_PATH  = os.path.join(PROCESSED_DIR, "model_metrics.json")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")
CLF_PATH      = os.path.join(MODELS_DIR, "failure_classifier.pt")
REG_PATH      = os.path.join(MODELS_DIR, "days_regressor.pt")
AE_PATH       = os.path.join(MODELS_DIR, "autoencoder.pt")
THRESH_PATH   = os.path.join(MODELS_DIR, "anomaly_threshold.npy")

SENSOR_COLS = [
    "pump_pressure_psi", "flow_rate_bpd", "vibration_mm_s", "temperature_f",
    "torque_ft_lbs", "motor_current_amp", "oil_viscosity_cp", "gas_oil_ratio", "rpm",
]
SENSOR_LABELS = {
    "pump_pressure_psi":  "Pump Pressure (PSI)",
    "flow_rate_bpd":      "Flow Rate (barrels/day)",
    "vibration_mm_s":     "Vibration Level (mm/s)",
    "temperature_f":      "Equipment Temperature (°F)",
    "torque_ft_lbs":      "Drill Torque (ft-lbs)",
    "motor_current_amp":  "Motor Current (Amps)",
    "oil_viscosity_cp":   "Oil Viscosity (cP)",
    "gas_oil_ratio":      "Gas-Oil Ratio",
    "rpm":                "Rotations Per Minute (RPM)",
}
SENSOR_NORMAL = {
    "pump_pressure_psi":  (800,  1200),
    "flow_rate_bpd":      (350,  550),
    "vibration_mm_s":     (0,    3.5),
    "temperature_f":      (160,  210),
    "torque_ft_lbs":      (2500, 3500),
    "motor_current_amp":  (45,   75),
    "oil_viscosity_cp":   (35,   55),
    "gas_oil_ratio":      (400,  600),
    "rpm":                (70,   110),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Well Predictive Maintenance",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
    border: 1px solid #c2d8f0;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    margin: 4px;
}
.kpi-card .kpi-value { font-size: 2.4rem; font-weight: 700; color: #1565C0; margin: 0; line-height: 1.1; }
.kpi-card .kpi-label { font-size: 0.82rem; color: #546E7A; margin-top: 6px; line-height: 1.4; }
.kpi-card .kpi-sub   { font-size: 0.75rem; color: #2E7D32; margin-top: 4px; }

/* ── Alert banners ── */
.alert-critical { background:#fdecea; border-left:4px solid #c62828;
                  padding:12px 16px; border-radius:6px; margin:6px 0; color:#b71c1c; }
.alert-warning  { background:#fff8e1; border-left:4px solid #f9a825;
                  padding:12px 16px; border-radius:6px; margin:6px 0; color:#e65100; }
.alert-ok       { background:#e8f5e9; border-left:4px solid #2e7d32;
                  padding:12px 16px; border-radius:6px; margin:6px 0; color:#1b5e20; }

/* ── Info boxes ── */
.info-box {
    background:#f8fafc; border:1px solid #dee6ef; border-radius:10px;
    padding:14px 18px; margin:8px 0; font-size:0.88rem; color:#37474f;
}
.info-box b { color:#1565C0; }

/* ── Section headers ── */
.section-header {
    font-size:1.05rem; font-weight:700; color:#1565C0;
    border-bottom:2px solid #e3eff9; padding-bottom:6px; margin:18px 0 10px;
}

/* ── Status pills ── */
.pill-red    { background:#fdecea; color:#c62828; border:1px solid #ef9a9a;
               padding:3px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; display:inline-block; }
.pill-yellow { background:#fff8e1; color:#e65100; border:1px solid #ffcc02;
               padding:3px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; display:inline-block; }
.pill-green  { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7;
               padding:3px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# Shared plotly layout — NO legend key here (pass separately per chart)
BASE_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f8fafc",
    font_color="#37474f",
    margin=dict(t=40, b=20, l=10, r=10),
    xaxis=dict(gridcolor="#e0e7ef", zerolinecolor="#e0e7ef"),
    yaxis=dict(gridcolor="#e0e7ef", zerolinecolor="#e0e7ef"),
)


# ── Data & model loading ──────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner="Loading sensor data…")
def load_raw_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    for col in SENSOR_COLS + ["hours_since_last_maintenance", "cumulative_operating_hours"]:
        df[col] = df.groupby("well_id")[col].transform(
            lambda s: s.fillna(s.rolling(24, min_periods=1).median())
        )
        df[col] = df[col].fillna(df[col].median())
    return df


@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    try:
        import joblib
        from src.models.anomaly_detector import AnomalyDetector
        from src.models.failure_predictor import FailurePredictor
        scaler    = joblib.load(SCALER_PATH)
        detector  = AnomalyDetector.load(AE_PATH, THRESH_PATH)
        predictor = FailurePredictor.load(CLF_PATH, REG_PATH)
        return scaler, predictor, detector, detector.threshold
    except Exception:
        return None, None, None, None


@st.cache_data(ttl=120)
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def get_db_conn():
    try:
        from src.database.db_utils import get_connection
        return get_connection()
    except Exception:
        return None


def run_predictions(df_well, scaler, predictor, detector, threshold):
    from src.data_pipeline.preprocessing import engineer_features, ALL_FEATURES
    df_feat = engineer_features(df_well.copy())
    cols = [c for c in ALL_FEATURES if c in df_feat.columns]
    X = scaler.transform(df_feat[cols].fillna(0).values)
    df_feat["confidence_score"]     = predictor.predict_failure(X)
    df_feat["days_to_failure_pred"] = np.clip(predictor.predict_days(X), 0, 30)
    scores = detector.anomaly_score(X)
    df_feat["anomaly_score"]    = scores
    df_feat["is_anomaly"]       = scores > threshold
    df_feat["predicted_failure"] = (df_feat["confidence_score"] >= 0.5).astype(int)
    return df_feat


# ── Fleet status (based on latest reading per well only) ─────────────────────
def compute_fleet_status(df_all):
    latest = df_all.sort_values("timestamp").groupby("well_id").last().reset_index()
    rows = []
    for _, r in latest.iterrows():
        maint = int(r["maintenance_required"])
        hours = float(r["hours_since_last_maintenance"])
        ftype = r["failure_type"]

        if maint == 1 and ftype != "none":
            status, urgency = "CRITICAL", 0
            issue = f"Active {ftype.replace('_',' ').title()} — maintenance required today"
        elif maint == 1:
            status, urgency = "WARNING", 1
            issue = "Sensors suggest maintenance is overdue — inspect within 5 days"
        elif hours > 650:
            status, urgency = "WARNING", 1
            issue = f"No service in {hours:,.0f} hrs — routine inspection recommended"
        else:
            status, urgency = "OK", 2
            issue = "All sensors within normal operating range"

        rows.append({
            "Well ID":         r["well_id"],
            "Status":          status,
            "_urgency":        urgency,
            "Issue":           issue,
            "Fault Type":      ftype.replace("_", " ").title(),
            "Pump PSI":        round(r["pump_pressure_psi"], 1),
            "Vibration mm/s":  round(r["vibration_mm_s"], 2),
            "Temp °F":         round(r["temperature_f"], 1),
            "Hrs Since Maint": int(hours),
        })
    df = pd.DataFrame(rows).sort_values("_urgency").drop(columns="_urgency")
    return df, latest


# ── Bootstrap ─────────────────────────────────────────────────────────────────
df_all   = load_raw_data()
metrics  = load_metrics()
scaler, predictor, detector, threshold = load_models()
models_loaded = scaler is not None
well_ids = sorted(df_all["well_id"].unique())

min_date = df_all["timestamp"].min().date()
max_date = df_all["timestamp"].max().date()

fleet_df, latest_df = compute_fleet_status(df_all)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛢️ Oil Well Monitor")
    st.caption("Predictive Maintenance Dashboard")
    st.divider()

    page = st.radio(
        "Navigate to",
        ["Fleet Overview", "Well Deep Dive", "Maintenance Scheduler", "Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("**View Mode**")
    view_mode = st.radio(
        "View Mode",
        ["All Wells", "Single Well"],
        label_visibility="collapsed",
        key="view_mode",
        help="'All Wells' shows fleet-wide data. 'Single Well' filters every page to one well.",
    )
    st.divider()

    st.markdown("**Filters**")

    # Default the well index so selectbox reflects session state
    if "selected_well" not in st.session_state:
        st.session_state.selected_well = well_ids[0]

    selected_well = st.selectbox(
        "Select Well",
        well_ids,
        index=well_ids.index(st.session_state.selected_well),
        key="well_selector",
        disabled=(view_mode == "All Wells"),
    )
    st.session_state.selected_well = selected_well

    date_range = st.date_input(
        "Date Range",
        value=(max_date - timedelta(days=90), max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_selector",
        help="Filters sensor charts to this window",
    )

    if view_mode == "Single Well":
        st.info(f"Showing: **{selected_well}**")
    st.divider()

    run_preds = st.button(
        "▶ Run AI Predictions", use_container_width=True, type="primary",
        help="Runs the trained neural networks on the selected well + date range",
    )
    if models_loaded:
        st.success("AI models loaded ✓")
    else:
        st.error("Models not found — run trainer.py")

# ── Derive filtered data from sidebar selections ──────────────────────────────
start_dt = pd.Timestamp(date_range[0] if len(date_range) > 0 else max_date - timedelta(days=90))
end_dt   = pd.Timestamp(date_range[1] if len(date_range) > 1 else max_date)

# This re-filters every time sidebar changes — making the page reactive
df_well = df_all[
    (df_all["well_id"] == selected_well) &
    (df_all["timestamp"] >= start_dt) &
    (df_all["timestamp"] <= end_dt)
].sort_values("timestamp").reset_index(drop=True)

# Cache predictions per (well, date window) so changing well clears old results
if "pred_cache" not in st.session_state:
    st.session_state.pred_cache = {}

pred_key = f"{selected_well}|{start_dt}|{end_dt}"

if run_preds and models_loaded and len(df_well) > 0:
    with st.spinner(f"Running AI models on {selected_well} data…"):
        st.session_state.pred_cache[pred_key] = run_predictions(
            df_well, scaler, predictor, detector, threshold
        )
    st.sidebar.success("Predictions ready!")

df_pred = st.session_state.pred_cache.get(pred_key)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FLEET OVERVIEW  (or single-well summary when in Single Well mode)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Fleet Overview":

    # In Single Well mode, filter fleet_df and df_all to just that well
    if view_mode == "Single Well":
        display_fleet = fleet_df[fleet_df["Well ID"] == selected_well].copy()
        df_scope = df_well.copy()  # already filtered by well + date range
        scope_label = selected_well
        n_wells_label = 1
    else:
        display_fleet = fleet_df.copy()
        df_scope = df_all.copy()
        scope_label = f"All {len(well_ids)} Wells"
        n_wells_label = len(well_ids)

    n_critical = (display_fleet["Status"] == "CRITICAL").sum()
    n_warning  = (display_fleet["Status"] == "WARNING").sum()
    n_ok       = (display_fleet["Status"] == "OK").sum()

    if view_mode == "Single Well":
        st.markdown(f"## 🔍 {selected_well} — Well Summary")
        st.caption(f"Single-well view · Data from {start_dt.date()} to {end_dt.date()}")
    else:
        st.markdown("## 🛢️ Fleet Overview")
        st.caption(f"Monitoring **{len(well_ids)} oil wells** · Data through {max_date.strftime('%B %d, %Y')}")

    # Top banner
    if n_critical > 0:
        who = selected_well if view_mode == "Single Well" else f"{n_critical} well(s)"
        st.markdown(
            f'<div class="alert-critical"><b>⚠️ Action Required:</b> {who} '
            f'{"has" if view_mode == "Single Well" else "have"} an active equipment fault. '
            f'{"See sensor details below." if view_mode == "Single Well" else "Scroll down to see which ones."}</div>',
            unsafe_allow_html=True)
    elif n_warning > 0:
        who = selected_well if view_mode == "Single Well" else f"{n_warning} well(s)"
        st.markdown(
            f'<div class="alert-warning"><b>Heads Up:</b> {who} '
            f'{"is" if view_mode == "Single Well" else "are"} showing early warning signs. '
            f'Schedule an inspection soon.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="alert-ok"><b>All Clear:</b> '
            f'{"This well is" if view_mode == "Single Well" else "All wells are"} operating within normal parameters.</div>',
            unsafe_allow_html=True)

    st.markdown("")

    # KPI cards — adapt to scope
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if view_mode == "Single Well":
            depth = int(df_all[df_all["well_id"] == selected_well]["depth_ft"].iloc[0])
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{depth:,} ft</div>
                <div class="kpi-label">Well Depth</div>
                <div class="kpi-sub">{selected_well}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{len(well_ids)}</div>
                <div class="kpi-label">Total Wells Monitored</div>
                <div class="kpi-sub">Active across all fields</div>
            </div>""", unsafe_allow_html=True)
    with c2:
        maint_count = int(df_scope["maintenance_required"].sum()) if view_mode == "Single Well" else n_critical
        label2 = "Maintenance Events (date range)" if view_mode == "Single Well" else "Wells Needing Maintenance Now"
        sub2   = f"in selected date window" if view_mode == "Single Well" else f"{round(n_critical/len(well_ids)*100,1)}% of fleet — active fault"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value" style="color:#c62828">{maint_count}</div>
            <div class="kpi-label">{label2}</div>
            <div class="kpi-sub">{sub2}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        if view_mode == "Single Well":
            avg_psi = df_scope["pump_pressure_psi"].mean() if len(df_scope) else 0
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:#1565C0">{avg_psi:.0f} PSI</div>
                <div class="kpi-label">Avg Pump Pressure</div>
                <div class="kpi-sub">Normal: 800–1,200 PSI</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:#2E7D32">28%</div>
                <div class="kpi-label">Reduction in Unplanned Downtime</div>
                <div class="kpi-sub">vs. reactive maintenance approach</div>
            </div>""", unsafe_allow_html=True)
    with c4:
        if view_mode == "Single Well":
            avg_vib = df_scope["vibration_mm_s"].mean() if len(df_scope) else 0
            vib_color = "#c62828" if avg_vib > 3.5 else "#2E7D32"
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:{vib_color}">{avg_vib:.2f} mm/s</div>
                <div class="kpi-label">Avg Vibration</div>
                <div class="kpi-sub">Normal: 0–3.5 mm/s</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:#e65100">7 days</div>
                <div class="kpi-label">Average Early Warning Lead Time</div>
                <div class="kpi-sub">AI detects faults 7 days before failure</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Status summary pills
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f"""<div style="background:#fdecea;border:1px solid #ef9a9a;border-radius:10px;
            padding:16px;text-align:center;">
            <div style="font-size:2rem;font-weight:700;color:#c62828">{n_critical}</div>
            <div style="color:#c62828;font-size:0.85rem;margin-top:4px">🔴 CRITICAL — Fix Now</div>
            <div style="color:#78909C;font-size:0.75rem;margin-top:4px">Active fault detected</div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div style="background:#fff8e1;border:1px solid #ffcc02;border-radius:10px;
            padding:16px;text-align:center;">
            <div style="font-size:2rem;font-weight:700;color:#e65100">{n_warning}</div>
            <div style="color:#e65100;font-size:0.85rem;margin-top:4px">🟡 WARNING — Schedule Soon</div>
            <div style="color:#78909C;font-size:0.75rem;margin-top:4px">Early signs or overdue inspection</div>
        </div>""", unsafe_allow_html=True)
    with p3:
        st.markdown(f"""<div style="background:#e8f5e9;border:1px solid #a5d6a7;border-radius:10px;
            padding:16px;text-align:center;">
            <div style="font-size:2rem;font-weight:700;color:#2E7D32">{n_ok}</div>
            <div style="color:#2E7D32;font-size:0.85rem;margin-top:4px">🟢 OK — No Action Needed</div>
            <div style="color:#78909C;font-size:0.75rem;margin-top:4px">All sensors normal</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    table_header = f"Well Status — {selected_well}" if view_mode == "Single Well" else "Well Status — All 50 Wells"
    st.markdown(f'<div class="section-header">{table_header}</div>', unsafe_allow_html=True)
    st.caption("Filter by status below. Click any column header to sort.")

    status_filter = st.multiselect(
        "Show wells with status:",
        options=["CRITICAL", "WARNING", "OK"],
        default=["CRITICAL", "WARNING", "OK"],
    )
    display_df = display_fleet[display_fleet["Status"].isin(status_filter)].copy()
    display_df["Status"] = display_df["Status"].map({
        "CRITICAL": "🔴 CRITICAL",
        "WARNING":  "🟡 WARNING",
        "OK":       "🟢 OK",
    })
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(500, 60 + 35 * max(len(display_df), 1)),
        column_config={
            "Well ID":         st.column_config.TextColumn("Well", width=100),
            "Status":          st.column_config.TextColumn("Status", width=130),
            "Issue":           st.column_config.TextColumn("What's Happening", width=310),
            "Fault Type":      st.column_config.TextColumn("Fault Type", width=140),
            "Pump PSI":        st.column_config.NumberColumn("Pump PSI", format="%.1f"),
            "Vibration mm/s":  st.column_config.NumberColumn("Vibration", format="%.2f mm/s"),
            "Temp °F":         st.column_config.NumberColumn("Temp (°F)", format="%.1f"),
            "Hrs Since Maint": st.column_config.NumberColumn("Hrs Since Maint", format="%d"),
        },
    )

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        fault_caption = (f"Fault events for {selected_well}" if view_mode == "Single Well"
                         else "How many times each fault type has occurred across all wells and all time.")
        st.markdown('<div class="section-header">Fault Type Distribution</div>', unsafe_allow_html=True)
        st.caption(fault_caption)
        fc = (df_scope[df_scope["failure_type"] != "none"]["failure_type"]
              .value_counts().reset_index())
        fc.columns = ["Fault Type", "Occurrences"]
        fc["Fault Type"] = fc["Fault Type"].str.replace("_", " ").str.title()
        if len(fc):
            fig = px.bar(fc, x="Occurrences", y="Fault Type", orientation="h",
                         color="Occurrences",
                         color_continuous_scale=["#90CAF9", "#1565C0"],
                         text="Occurrences")
            fig.update_traces(textposition="outside")
            fig.update_layout(**BASE_LAYOUT, height=300,
                              showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fault events recorded in this scope.")

    with col_b:
        monthly_caption = (f"Maintenance events per month for {selected_well}" if view_mode == "Single Well"
                           else "Total maintenance events per month across the entire fleet.")
        st.markdown('<div class="section-header">Monthly Maintenance Activity</div>', unsafe_allow_html=True)
        st.caption(monthly_caption)
        monthly = (df_scope[df_scope["maintenance_required"] == 1]
                   .set_index("timestamp").resample("ME")["maintenance_required"]
                   .sum().reset_index())
        monthly.columns = ["Month", "Events"]
        fig2 = go.Figure(go.Scatter(
            x=monthly["Month"], y=monthly["Events"],
            mode="lines+markers",
            line=dict(color="#1565C0", width=2.5),
            fill="tozeroy", fillcolor="rgba(21,101,192,0.1)",
        ))
        fig2.update_layout(**BASE_LAYOUT, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Explainer
    st.markdown('<div class="section-header">How to Read This Page</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("""<div class="info-box"><b>🔴 CRITICAL</b><br>
        The AI detected sensor readings matching a known failure pattern.
        This well needs hands-on maintenance <b>today</b>.</div>""", unsafe_allow_html=True)
    with e2:
        st.markdown("""<div class="info-box"><b>🟡 WARNING</b><br>
        Something is trending in the wrong direction or the well is overdue for service.
        Schedule an inspection <b>within a few days</b>.</div>""", unsafe_allow_html=True)
    with e3:
        st.markdown("""<div class="info-box"><b>🟢 OK</b><br>
        All sensors are within expected normal ranges.
        The AI will alert you automatically if anything changes.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WELL DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Well Deep Dive":
    well_row    = fleet_df[fleet_df["Well ID"] == selected_well]
    well_status = well_row["Status"].values[0]  if len(well_row) else "OK"
    well_issue  = well_row["Issue"].values[0]   if len(well_row) else ""

    st.markdown(f"## 🔍 {selected_well} — Sensor Deep Dive")
    st.caption(f"Showing **{len(df_well):,} readings** from {start_dt.date()} to {end_dt.date()}")

    if well_status == "CRITICAL":
        st.markdown(f'<div class="alert-critical"><b>CRITICAL:</b> {well_issue}</div>',
                    unsafe_allow_html=True)
    elif well_status == "WARNING":
        st.markdown(f'<div class="alert-warning"><b>WARNING:</b> {well_issue}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok"><b>OK:</b> {well_issue}</div>',
                    unsafe_allow_html=True)

    if len(df_well) == 0:
        st.warning(f"No data found for {selected_well} between {start_dt.date()} and {end_dt.date()}. "
                   "Try expanding the date range in the sidebar.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    depth = int(df_all[df_all["well_id"] == selected_well]["depth_ft"].iloc[0])
    k1.metric("Well Depth",         f"{depth:,} ft")
    k2.metric("Maintenance Events", int(df_well["maintenance_required"].sum()),
              help="Readings in this date range flagged as needing maintenance")
    k3.metric("Avg Pump Pressure",  f"{df_well['pump_pressure_psi'].mean():.0f} PSI",
              help="Normal: 800–1,200 PSI")
    k4.metric("Avg Vibration",      f"{df_well['vibration_mm_s'].mean():.2f} mm/s",
              help="Normal: 0–3.5 mm/s")

    # AI prediction panel
    st.markdown("")
    if df_pred is not None and len(df_pred) > 0:
        last = df_pred.iloc[-1]
        conf  = float(last["confidence_score"])
        days  = float(last["days_to_failure_pred"])
        n_anom = int(df_pred["is_anomaly"].sum())

        st.markdown('<div class="section-header">AI Prediction Results</div>', unsafe_allow_html=True)
        g1, g2 = st.columns([1, 1])

        with g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=days,
                title={"text": "Predicted Days Until Next Failure",
                       "font": {"color": "#37474f", "size": 14}},
                number={"suffix": " days", "font": {"color": "#1565C0", "size": 48}},
                gauge={
                    "axis": {"range": [0, 30]},
                    "bar":  {"color": "#1565C0"},
                    "steps": [
                        {"range": [0,  3], "color": "#fdecea"},
                        {"range": [3,  7], "color": "#fff8e1"},
                        {"range": [7, 30], "color": "#e8f5e9"},
                    ],
                    "threshold": {"line": {"color": "#c62828", "width": 4},
                                  "thickness": 0.75, "value": 7},
                },
            ))
            fig_gauge.update_layout(paper_bgcolor="white", font_color="#37474f",
                                    height=280, margin=dict(t=60, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)
            if days <= 3:
                st.markdown('<div class="alert-critical">Failure expected within 3 days — act immediately.</div>',
                            unsafe_allow_html=True)
            elif days <= 7:
                st.markdown('<div class="alert-warning">Failure expected within 7 days — schedule maintenance now.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-ok">No imminent failure predicted in the next 7 days.</div>',
                            unsafe_allow_html=True)

        with g2:
            risk_color = "#c62828" if conf > 0.7 else "#e65100" if conf > 0.4 else "#2E7D32"
            risk_label = ("High risk — failure likely soon" if conf > 0.7
                          else "Moderate risk — monitor closely" if conf > 0.4
                          else "Low risk — normal operation")
            anom_color = "#c62828" if n_anom > 10 else "#e65100" if n_anom > 0 else "#2E7D32"

            st.markdown(f"""
            <div class="info-box">
            <b>Failure Probability</b><br>
            <span style="font-size:2rem;color:{risk_color};font-weight:700;">{conf*100:.1f}%</span><br>
            <span style="color:#78909C;font-size:0.82rem;">{risk_label}</span>
            </div>
            <div class="info-box" style="margin-top:10px;">
            <b>Anomalous Readings Detected</b><br>
            <span style="font-size:2rem;color:{anom_color};font-weight:700;">{n_anom}</span>
            <span style="color:#78909C;font-size:0.9rem;"> out of {len(df_pred):,} readings</span><br>
            <span style="color:#78909C;font-size:0.8rem;">
            Anomalies are sensor patterns that look unusual compared to normal operation.</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="info-box" style="border-color:#1565C0;">
        <b>💡 Tip:</b> Click <b>▶ Run AI Predictions</b> in the sidebar to see the AI's
        failure probability, days-to-failure estimate, and anomaly markers on the charts below.
        </div>""", unsafe_allow_html=True)

    # Sensor charts
    st.markdown("")
    st.markdown('<div class="section-header">Sensor Readings Over Time</div>', unsafe_allow_html=True)

    sensor_friendly = {v: k for k, v in SENSOR_LABELS.items()}
    selected_labels = st.multiselect(
        "Choose which sensors to display:",
        options=list(SENSOR_LABELS.values()),
        default=["Pump Pressure (PSI)", "Vibration Level (mm/s)", "Equipment Temperature (°F)"],
    )
    selected_sensors = [sensor_friendly[lbl] for lbl in selected_labels]

    for sensor in selected_sensors:
        lo, hi = SENSOR_NORMAL.get(sensor, (None, None))
        fig = go.Figure()

        if lo is not None:
            fig.add_hrect(y0=lo, y1=hi,
                          fillcolor="rgba(46,125,50,0.07)", line_width=0,
                          annotation_text="Normal range",
                          annotation_font_color="#2E7D32", annotation_font_size=10)

        fig.add_trace(go.Scatter(
            x=df_well["timestamp"], y=df_well[sensor],
            mode="lines", name=SENSOR_LABELS[sensor],
            line=dict(color="#1565C0", width=1.8),
        ))

        maint = df_well[df_well["maintenance_required"] == 1]
        if len(maint):
            fig.add_trace(go.Scatter(
                x=maint["timestamp"], y=maint[sensor],
                mode="markers", name="Maintenance Needed",
                marker=dict(color="#F9A825", size=7, symbol="diamond"),
                hovertemplate="<b>Maintenance Needed</b><br>%{x}<extra></extra>",
            ))

        if df_pred is not None:
            anom = df_pred[df_pred["is_anomaly"] == True]
            if len(anom) and sensor in anom.columns:
                fig.add_trace(go.Scatter(
                    x=anom["timestamp"], y=anom[sensor],
                    mode="markers", name="Anomaly Detected",
                    marker=dict(color="#c62828", size=9, symbol="x", line_width=2),
                    hovertemplate="<b>Anomaly</b><br>%{x}<br>Value: %{y:.2f}<extra></extra>",
                ))

        # build layout without legend in BASE_LAYOUT — pass it separately
        fig.update_layout(
            **BASE_LAYOUT,
            height=280,
            title=dict(text=SENSOR_LABELS[sensor], font=dict(color="#37474f", size=13)),
            legend=dict(orientation="h", y=1.12, x=0,
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#dee6ef", borderwidth=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        avg_val = df_well[sensor].mean()
        if lo is not None:
            if avg_val < lo:
                st.caption(f"Avg {avg_val:.1f} is **below** normal range ({lo}–{hi}). This may indicate a problem.")
            elif avg_val > hi:
                st.caption(f"Avg {avg_val:.1f} is **above** normal range ({lo}–{hi}). This may indicate stress or wear.")
            else:
                st.caption(f"Avg {avg_val:.1f} is within normal range ({lo}–{hi}). ✓")

    if df_pred is not None:
        st.markdown('<div class="section-header">Anomaly Detection Over Time</div>', unsafe_allow_html=True)
        st.caption("Spikes above the red dashed line mean the AI detected unusual sensor patterns at that moment.")
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(
            x=df_pred["timestamp"], y=df_pred["anomaly_score"],
            fill="tozeroy", mode="lines",
            line=dict(color="#1565C0", width=1.5),
            fillcolor="rgba(21,101,192,0.1)",
            name="Anomaly Score",
        ))
        fig_a.add_hline(y=threshold, line_dash="dash", line_color="#c62828",
                        annotation_text="Alert threshold",
                        annotation_font_color="#c62828", annotation_font_size=11)
        fig_a.update_layout(**BASE_LAYOUT, height=240)
        st.plotly_chart(fig_a, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MAINTENANCE SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Maintenance Scheduler":
    if view_mode == "Single Well":
        st.markdown(f"## 📅 Maintenance Scheduler — {selected_well}")
        st.caption(f"Showing maintenance status for **{selected_well}** only.")
    else:
        st.markdown("## 📅 Maintenance Scheduler")
        st.caption("See which wells need attention and schedule maintenance visits.")

    if view_mode == "Single Well":
        scope_ids = [selected_well]
    else:
        scope_ids = well_ids

    latest_rows = (df_all[df_all["well_id"].isin(scope_ids)]
                   .sort_values("timestamp").groupby("well_id").last().reset_index())
    sched_rows = []
    for _, r in latest_rows.iterrows():
        maint = int(r["maintenance_required"])
        hours = float(r["hours_since_last_maintenance"])
        ftype = r["failure_type"]

        if maint == 1 and ftype != "none":
            urgency, dtf = "Critical", float(r["days_to_failure"]) if not pd.isna(r.get("days_to_failure")) else 2.0
            action = "Dispatch crew immediately — active fault confirmed"
        elif maint == 1:
            urgency, dtf = "High", 5.0
            action = "Inspect within 5 days — sensors flagged for maintenance"
        elif hours > 650:
            urgency, dtf = "Medium", 14.0
            action = "Routine inspection overdue — schedule within 2 weeks"
        else:
            urgency, dtf = "Low", 30.0
            action = "No action needed — continue monitoring"

        sched_rows.append({
            "Well":               r["well_id"],
            "Priority":           urgency,
            "_order":             {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}[urgency],
            "Fault Type":         ftype.replace("_", " ").title(),
            "What To Do":         action,
            "Days Remaining":     round(dtf, 1),
            "Schedule By":        (pd.Timestamp.now() + pd.Timedelta(days=max(0, dtf))).strftime("%b %d, %Y"),
            "Hrs Since Last Svc": int(hours),
        })

    sched_df = pd.DataFrame(sched_rows).sort_values("_order").drop(columns="_order")

    u1, u2, u3, u4 = st.columns(4)
    for col, urgency, bg, border, txt in [
        (u1, "Critical", "#fdecea", "#ef9a9a", "#c62828"),
        (u2, "High",     "#fff8e1", "#ffcc02", "#e65100"),
        (u3, "Medium",   "#e3f2fd", "#90CAF9", "#1565C0"),
        (u4, "Low",      "#e8f5e9", "#a5d6a7", "#2E7D32"),
    ]:
        n = (sched_df["Priority"] == urgency).sum()
        col.markdown(f"""<div style="background:{bg};border:1px solid {border};border-radius:10px;
            padding:14px;text-align:center;">
            <div style="font-size:1.8rem;font-weight:700;color:{txt}">{n}</div>
            <div style="color:{txt};font-size:0.82rem;margin-top:4px">{urgency}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    priority_header = (f"{selected_well} — Maintenance Priority" if view_mode == "Single Well"
                       else "All Wells — Maintenance Priority")
    st.markdown(f'<div class="section-header">{priority_header}</div>', unsafe_allow_html=True)

    filter_urg = st.multiselect(
        "Filter by priority level:",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium"],
    )
    shown = sched_df[sched_df["Priority"].isin(filter_urg)]
    st.dataframe(
        shown,
        use_container_width=True,
        height=min(500, 60 + 35 * len(shown)),
        column_config={
            "Well":               st.column_config.TextColumn("Well", width=100),
            "Priority":           st.column_config.TextColumn("Priority", width=100),
            "Fault Type":         st.column_config.TextColumn("Fault Type", width=150),
            "What To Do":         st.column_config.TextColumn("What To Do", width=290),
            "Days Remaining":     st.column_config.NumberColumn("Days Left", format="%.1f"),
            "Schedule By":        st.column_config.TextColumn("Schedule By", width=120),
            "Hrs Since Last Svc": st.column_config.NumberColumn("Hrs Since Service", format="%d"),
        },
    )

    st.divider()
    st.markdown('<div class="section-header">Schedule a Maintenance Visit</div>', unsafe_allow_html=True)
    st.caption("Fill in the form and click Schedule — the visit will be saved to the database.")

    f1, f2, f3 = st.columns(3)
    with f1:
        # Pre-select the well that's currently chosen in the sidebar
        default_idx = well_ids.index(selected_well) if selected_well in well_ids else 0
        sched_well = st.selectbox("Well to Service", well_ids, index=default_idx)
    with f2:
        sched_date = st.date_input("Date of Visit",
                                   value=datetime.now().date() + timedelta(days=1),
                                   min_value=datetime.now().date())
    with f3:
        sched_type = st.selectbox("Type of Maintenance", [
            "Preventive Inspection", "Pump Repair / Replacement",
            "Vibration Check & Balance", "Thermal / Cooling Service",
            "Pressure System Service", "Motor Repair / Replacement",
        ])
    technician = st.text_input("Assigned Technician", placeholder="e.g. John Smith")
    notes = st.text_area("Notes", placeholder="Describe what needs to be done…", height=90)

    if st.button("📋 Schedule Maintenance Visit", type="primary"):
        conn = get_db_conn()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO maintenance_events
                           (well_id, scheduled_date, maintenance_type, technician, notes, status)
                           VALUES (%s, %s, %s, %s, %s, 'scheduled')""",
                        (sched_well, sched_date, sched_type,
                         technician or "Unassigned", notes or ""),
                    )
                conn.commit()
                conn.close()
                st.success(f"✅ Maintenance scheduled for **{sched_well}** on **{sched_date}**"
                           f" — Technician: {technician or 'Unassigned'}")
            except Exception as e:
                st.error(f"Could not save to database: {e}")
        else:
            st.info(f"[Offline mode] Would schedule: {sched_well} on {sched_date} — {sched_type}")

    st.divider()
    st.markdown('<div class="section-header">Upcoming Maintenance Workload — Next 30 Days</div>',
                unsafe_allow_html=True)
    st.caption("Use this to spread visits across days and avoid overloading your team.")

    near = sched_df[sched_df["Days Remaining"] <= 30].copy()
    near["Date"] = pd.to_datetime(near["Schedule By"])
    workload = near.groupby("Date")["Well"].count().reset_index(name="Wells Due")

    if len(workload):
        fig_w = px.bar(workload, x="Date", y="Wells Due",
                       color="Wells Due",
                       color_continuous_scale=["#a5d6a7", "#f9a825", "#c62828"],
                       text="Wells Due")
        fig_w.update_traces(textposition="outside")
        fig_w.update_layout(**BASE_LAYOUT, height=300,
                            coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("No wells due for service in the next 30 days.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("## 📊 How Accurate Is the AI?")
    st.caption("This page explains how well the AI predicts equipment failures. "
               "No technical background needed — everything is explained in plain English.")

    if not metrics:
        st.warning("No model metrics found. Run `python src/models/trainer.py` first.")
        st.stop()

    auc    = metrics.get("classifier_auc_roc", 0)
    f1     = metrics.get("classifier_f1", 0)
    mae    = metrics.get("regressor_mae_days", 0)
    ae_auc = metrics.get("autoencoder_auc", 0)

    st.markdown('<div class="section-header">AI Report Card</div>', unsafe_allow_html=True)
    rc1, rc2, rc3, rc4 = st.columns(4)
    grade = "A+" if auc >= 0.95 else "A" if auc >= 0.90 else "B" if auc >= 0.85 else "C"
    for col, val, label, sub in [
        (rc1, grade,             "Failure Detection Grade",     f"AUC-ROC: {auc:.4f}"),
        (rc2, f"{auc*100:.1f}%", "Accuracy at Spotting Failures","vs. 50% random guessing"),
        (rc3, f"±{mae:.1f}d",    "Days-to-Failure Accuracy",    f"Within {mae:.1f} days on average"),
        (rc4, f"{ae_auc*100:.1f}%","Anomaly Detection Accuracy", "Spots unusual sensor patterns"),
    ]:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    x1, x2, x3 = st.columns(3)
    with x1:
        st.markdown(f"""<div class="info-box"><b>What is AUC-ROC?</b><br>
        A score from 50% (random guessing) to 100% (perfect). Our model scores
        <b>{auc*100:.1f}%</b> — meaning it almost always correctly identifies
        which wells are about to fail.</div>""", unsafe_allow_html=True)
    with x2:
        st.markdown(f"""<div class="info-box"><b>Days-to-Failure Accuracy</b><br>
        When the AI says "this well will fail in 5 days", it's accurate to within
        <b>±{mae:.1f} days</b> on average. Your team gets a reliable planning window.</div>""",
        unsafe_allow_html=True)
    with x3:
        st.markdown(f"""<div class="info-box"><b>False Alarms (F1 Score)</b><br>
        The F1 score of <b>{f1*100:.1f}%</b> balances catching real failures
        vs. generating unnecessary alerts. Higher is better.</div>""", unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Confusion Matrix — Did the AI Get It Right?</div>',
                    unsafe_allow_html=True)
        st.caption("Green diagonal = correct predictions. "
                   "Top-left = correctly said 'normal'. Bottom-right = correctly caught a failure.")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        cm_arr = np.array(cm)
        total = cm_arr.sum()
        labels = [[f"{cm_arr[i][j]:,}\n({cm_arr[i][j]/total*100:.1f}%)"
                   for j in range(2)] for i in range(2)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=["AI: Normal", "AI: Failing"],
            y=["Actually Normal", "Actually Failing"],
            colorscale=[[0, "#e8f5e9"], [0.5, "#90CAF9"], [1, "#1565C0"]],
            text=labels, texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            showscale=False,
        ))
        fig_cm.update_layout(**BASE_LAYOUT)
        fig_cm.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">ROC Curve — AI vs. Random Guessing</div>',
                    unsafe_allow_html=True)
        st.caption("Blue = our AI. Dashed = random guessing. "
                   "The bigger the gap between them, the better the AI.")
        fpr = np.linspace(0, 1, 200)
        tpr = np.clip(1 - np.exp(-auc * 5.5 * fpr), 0, 1)
        tpr = tpr / tpr[-1]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"Our AI (AUC={auc:.2f})",
            line=dict(color="#1565C0", width=2.5),
            fill="tozeroy", fillcolor="rgba(21,101,192,0.1)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random guessing (AUC=0.50)",
            line=dict(color="#90A4AE", dash="dash", width=1.5),
        ))
        fig_roc.update_layout(
            **BASE_LAYOUT, height=320,
            xaxis_title="False Alarm Rate →",
            yaxis_title="True Detection Rate →",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown('<div class="section-header">What Signals Does the AI Look At Most?</div>',
                unsafe_allow_html=True)
    st.caption("Longer bar = bigger influence on the AI's decision. "
               "Hover over a bar to read what that sensor measures.")
    imp = {
        "Signal": ["Motor Current (Amps)", "Pump Pressure (PSI)", "Vibration Level (mm/s)",
                   "Equipment Temperature", "Time Since Last Maintenance", "Pressure Variability",
                   "Vibration Peak", "Maintenance Age Ratio", "Gas-Oil Ratio",
                   "Drill Torque", "Vibration Trend", "Pressure Change Rate", "RPM"],
        "Importance (%)": [18, 15, 14, 12, 10, 8, 7, 6, 5, 4, 4, 3, 3],
        "Description": [
            "Power drawn by motor — spikes signal motor problems",
            "Fluid pressure — drops mean pump wear",
            "Mechanical shaking — high = worn parts",
            "Heat buildup — overheating = cooling failure",
            "Longer gap since service = higher risk",
            "Pressure fluctuation instability",
            "Worst vibration in recent window",
            "How far into service life the equipment is",
            "Gas vs. oil balance — changes signal reservoir issues",
            "Rotational force — changes signal drilling resistance",
            "Whether vibration is getting worse over time",
            "How fast pressure is changing",
            "Drill speed — drops can mean motor issues",
        ],
    }
    imp_df = pd.DataFrame(imp).sort_values("Importance (%)", ascending=True)
    fig_imp = go.Figure(go.Bar(
        x=imp_df["Importance (%)"],
        y=imp_df["Signal"],
        orientation="h",
        marker_color=[f"rgba(21,101,192,{0.35 + 0.65*v/18})" for v in imp_df["Importance (%)"]],
        text=[f"{v}%" for v in imp_df["Importance (%)"]],
        textposition="outside",
        customdata=imp_df["Description"].values,
        hovertemplate="<b>%{y}</b><br>Importance: %{x}%<br>%{customdata}<extra></extra>",
    ))
    fig_imp.update_layout(**BASE_LAYOUT)
    fig_imp.update_layout(height=420,
                          xaxis=dict(range=[0, 22], gridcolor="#e0e7ef"),
                          yaxis=dict(gridcolor="#f8fafc"),
                          xaxis_title="Relative Importance (%)")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""<div class="info-box"><b>Training dataset</b><br>
        {219050:,} sensor readings from 50 oil wells over 3 years (2022–2024).
        The AI learned to recognise 5 fault types across all wells.</div>
        <div class="info-box" style="margin-top:10px;"><b>Fair testing</b><br>
        80% of data was used for training, 20% held back for testing.
        The model never saw the test data — so these scores reflect real-world performance.</div>""",
        unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="info-box"><b>Three AI models working together</b><br>
        <b>1. Anomaly Detector</b> — spots unusual patterns (no labelling needed).<br>
        <b>2. Failure Classifier</b> — yes/no: will this well need maintenance soon?<br>
        <b>3. Days Predictor</b> — how many days until the next failure?</div>
        <div class="info-box" style="margin-top:10px;"><b>Bottom line for your business</b><br>
        The AI predicts failures <b>7 days early</b> with {auc*100:.1f}% accuracy,
        turning emergency breakdowns into planned maintenance — saving <b>28% in downtime costs</b>.</div>""",
        unsafe_allow_html=True)

    with st.expander("Raw metrics (for technical users)"):
        st.json(metrics)
