"""
Gut But — Behavioral Health Trigger Detection
=============================================
Streamlit application that loads a trained LightGBM model and lets a user
input their daily wearable signals to receive a real-time trigger risk score,
SHAP-driven signal breakdown, and a plain-language health insight.

Project layout expected:
  project/
    app.py                  ← this file
    gutbut_pipeline.py      ← pipeline functions (feature engineering etc.)
    model/
      lgbm_model.txt        ← saved LightGBM booster (from pipeline step 6)
      feature_cols.json     ← ordered list of feature names
      calibrator.pkl        ← isotonic calibrator (optional)
    outputs/
      feature_impact.csv    ← SHAP impact table (from pipeline step 6)

Run:
    streamlit run app.py

Dependencies:
    pip install streamlit lightgbm shap pandas numpy matplotlib plotly
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")          # non-interactive backend required in Streamlit
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gut But · Trigger Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# Global CSS — dark clinical aesthetic
# ─────────────────────────────────────────────────────────────
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&family=DM+Sans:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e0f;
    color: #e8f0ee;
}
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #111618;
    border-right: 1px solid #1e2729;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── Typography ── */
h1 { font-family: 'Fraunces', serif; font-weight: 300; font-size: 2rem;
     letter-spacing: -0.5px; color: #e8f0ee; }
h2 { font-family: 'Fraunces', serif; font-weight: 300; font-size: 1.3rem;
     color: #e8f0ee; letter-spacing: -0.3px; }
h3 { font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 1.5px;
     text-transform: uppercase; color: #3d5652; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #111618;
    border: 1px solid #1e2729;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    font-size: 2rem !important;
    color: #2dd4bf !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #3d5652 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #2dd4bf !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #2dd4bf;
    color: #0a0e0f;
    border: none;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85; background: #2dd4bf; color: #0a0e0f; }

/* ── Dividers ── */
hr { border-color: #1e2729; margin: 1.5rem 0; }

/* ── Info / warning boxes ── */
.risk-card {
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 3px solid;
}
.risk-low    { background: rgba(45,212,191,.07);  border-color: #2dd4bf; }
.risk-medium { background: rgba(245,158,11,.07);  border-color: #f59e0b; }
.risk-high   { background: rgba(248,113,113,.07); border-color: #f87171; }

.risk-pct {
    font-family: 'DM Mono', monospace;
    font-size: 3.5rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.risk-low    .risk-pct { color: #2dd4bf; }
.risk-medium .risk-pct { color: #f59e0b; }
.risk-high   .risk-pct { color: #f87171; }

.risk-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.risk-low    .risk-label { color: #2dd4bf; }
.risk-medium .risk-label { color: #f59e0b; }
.risk-high   .risk-label { color: #f87171; }

.insight-text {
    font-family: 'Fraunces', serif;
    font-size: 1rem;
    font-weight: 300;
    line-height: 1.7;
    color: #c8d8d5;
}

/* ── Signal tags ── */
.signal-tag {
    display: inline-block;
    background: rgba(45,212,191,.1);
    border: 1px solid #1a6b62;
    color: #2dd4bf;
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    margin: 2px;
}

/* ── Section header ── */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3d5652;
    border-bottom: 1px solid #1e2729;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #111618;
    border-color: #1e2729;
    color: #e8f0ee;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
MODEL_DIR   = Path("model")
OUTPUT_DIR  = Path("outputs")

MODEL_PATH       = MODEL_DIR / "lgbm_model.txt"
FEATURE_COL_PATH = MODEL_DIR / "feature_cols.json"
CALIBRATOR_PATH  = MODEL_DIR / "calibrator.pkl"
IMPACT_PATH      = OUTPUT_DIR / "feature_impact.csv"


# ─────────────────────────────────────────────────────────────
# Model loading  (cached — loads once per session)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the LightGBM booster, feature column list, and optional calibrator.

    Uses @st.cache_resource so the model is loaded once and shared across
    all user sessions without reloading on every interaction.

    Falls back gracefully to a demo mode if model files aren't present —
    useful when demoing the UI before running the full training pipeline.
    """
    if not MODEL_PATH.exists():
        st.warning(
            "⚠️ No trained model found at `model/lgbm_model.txt`.  "
            "Running in **demo mode** with a synthetic model.  "
            "Train the pipeline first to use real predictions.",
            icon="⚠️",
        )
        return _build_demo_model()

    booster = lgb.Booster(model_file=str(MODEL_PATH))

    # Feature column list is required to align input vector correctly
    if not FEATURE_COL_PATH.exists():
        raise FileNotFoundError(
            "feature_cols.json not found.  "
            "Re-run train_lightgbm() and save feature_cols alongside the model."
        )
    with open(FEATURE_COL_PATH) as f:
        feature_cols = json.load(f)

    # Isotonic calibrator is optional — fall back to raw probabilities
    calibrator = None
    if CALIBRATOR_PATH.exists():
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = pickle.load(f)

    return booster, feature_cols, calibrator


@st.cache_data(show_spinner=False)
def load_impact_table():
    """Load pre-computed SHAP impact table from the last pipeline run."""
    if IMPACT_PATH.exists():
        return pd.read_csv(IMPACT_PATH)
    return None


# ─────────────────────────────────────────────────────────────
# Feature engineering for user inputs
# ─────────────────────────────────────────────────────────────

# The full feature list includes lag/rolling columns that are not directly
# entered by the user.  For real-time single-day prediction we:
#   1. Use today's values where available.
#   2. Fill lag and rolling features with the same value (assumes no change —
#      conservative but reasonable for a single snapshot).
#   3. Fill everything else with 0.0 so LightGBM uses its missing-value path.
#
# In production this would pull the user's 7-day history from a database
# and compute proper lags — this function handles the single-entry demo case.

DIRECT_SIGNAL_MAP = {
    # user-facing name  →  base feature name in model
    "sleep_hours":         "sleep_hours",
    "sleep_efficiency":    "sleep_efficiency",
    "sleep_rem_ratio":     "sleep_rem_ratio",
    "steps":               "steps",
    "active_minutes":      "active_minutes",
    "resting_hr":          "resting_hr",
    "stress_score":        "stress_score",
    "rmssd":               "rmssd",
    "spo2":                "spo2",
    "sedentary_fraction":  "sedentary_fraction",
}

# ─────────────────────────────────────────────────────────────
# Median loader  — fills the 26 non-slider features
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_medians() -> dict:
    """
    Load per-feature training-set medians from model/feature_medians.json.

    Why medians instead of NaN
    --------------------------
    The model was trained on real wearable data where every feature had a
    value.  Leaving unset features as NaN tells LightGBM "this person has
    missing data" — which it treats as a strong signal toward high risk,
    causing the 99% bug regardless of slider values.

    Filling with the training median tells the model "this person is
    average on signals I don't have slider data for", which is a neutral
    and honest assumption.  The 10 slider values then push the score up
    or down from that neutral baseline.

    Fallback
    --------
    If feature_medians.json doesn't exist yet (first run before exporting
    from Colab), hard-coded population averages from the LifeSnaps dataset
    are used for the 26 known NaN columns.  Generate the real file with
    the export_model_artifacts.py cell in Colab for best accuracy.
    """
    medians_path = MODEL_DIR / "feature_medians.json"
    if medians_path.exists():
        with open(medians_path) as f:
            return json.load(f)

    # Hard-coded fallback — LifeSnaps population averages
    return {
        "sleep_deep_ratio":          0.17,
        "sleep_wake_ratio":          0.07,
        "nremhr":                    58.2,
        "scl_avg":                   2.1,
        "temp_deviation":            0.0,
        "calories":                  2050.0,
        "is_weekend":                0.0,
        "sleep_deep_ratio_lag1":     0.17,
        "sleep_deep_ratio_lag3":     0.17,
        "sleep_deep_ratio_lag7":     0.17,
        "sleep_deep_ratio_roll3":    0.17,
        "sleep_deep_ratio_roll7":    0.17,
        "sleep_deep_ratio_delta1":   0.0,
        "temp_deviation_lag1":       0.0,
        "temp_deviation_lag3":       0.0,
        "temp_deviation_lag7":       0.0,
        "temp_deviation_roll3":      0.0,
        "temp_deviation_roll7":      0.0,
        "temp_deviation_delta1":     0.0,
        "scl_avg_lag1":              2.1,
        "scl_avg_lag3":              2.1,
        "scl_avg_lag7":              2.1,
        "scl_avg_roll3":             2.1,
        "scl_avg_roll7":             2.1,
        "scl_avg_delta1":            0.0,
    }


# ─────────────────────────────────────────────────────────────
# Feature vector builder  (fixed — median baseline)
# ─────────────────────────────────────────────────────────────

def build_input_vector(user_inputs: dict, feature_cols: list) -> pd.DataFrame:
    """
    Convert 10 slider inputs into the full 96-feature model vector.

    Three-step filling strategy
    ---------------------------
    Step 1 — seed every feature with its training-set median.
              This is the critical fix: unset features now represent
              "average person" instead of "missing data".
    Step 2 — override the 10 direct slider signals with user values.
    Step 3 — propagate each slider into its lag/roll/delta variants
              (approximates "no change from recent days").

    Result: 0 NaN columns, correct score sensitivity to slider changes.
    """
    medians = load_medians()

    # Step 1 — neutral median baseline for all 96 features
    row = {col: medians.get(col, 0.0) for col in feature_cols}

    # Step 2 + 3 — override with user values and propagate to lags
    for ui_name, base_name in DIRECT_SIGNAL_MAP.items():
        if ui_name not in user_inputs:
            continue
        val = user_inputs[ui_name]

        # Direct feature
        if base_name in row:
            row[base_name] = val

        # Lag features — approximate with today's value
        for k in [1, 3, 7]:
            key = f"{base_name}_lag{k}"
            if key in row:
                row[key] = val

        # Rolling means — approximate with today's value
        for w in [3, 7]:
            key = f"{base_name}_roll{w}"
            if key in row:
                row[key] = val

        # Delta — 0 means "no change from yesterday"
        delta_key = f"{base_name}_delta1"
        if delta_key in row:
            row[delta_key] = 0.0

    # Composite features
    sleep  = user_inputs.get("sleep_hours", 7.0)
    active = max(user_inputs.get("active_minutes", 30), 1)

    if "sleep_activity_ratio" in row:
        row["sleep_activity_ratio"] = sleep / active
    if "hr_above_baseline" in row:
        row["hr_above_baseline"] = 0.0
    if "is_weekend" in row:
        row["is_weekend"] = 0.0
    if "day_of_week" in row:
        row["day_of_week"] = 0.0

    df = pd.DataFrame([row], columns=feature_cols)

    # Safety net — fill any remaining NaN with 0
    remaining_nan = df.isna().sum().sum()
    if remaining_nan > 0:
        df = df.fillna(0.0)

    return df


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

def predict_risk(input_df: pd.DataFrame, booster, calibrator) -> tuple[float, float]:
    """
    Run the model and return (raw_probability, calibrated_probability).

    Returns probabilities in [0, 1].
    """
    raw_proba = booster.predict(input_df)[0]
    cal_proba = calibrator.predict([raw_proba])[0] if calibrator else raw_proba
    return float(raw_proba), float(cal_proba)


# ─────────────────────────────────────────────────────────────
# SHAP for a single prediction
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_shap_explainer(_booster):
    """
    Build a TreeExplainer for the loaded booster.
    Cached so it is created once and reused across predictions.
    The leading underscore on _booster tells Streamlit not to hash it
    (LightGBM Booster objects are not hashable).
    """
    return shap.TreeExplainer(_booster)


def compute_shap(input_df: pd.DataFrame, explainer, feature_cols: list) -> pd.DataFrame:
    """
    Compute SHAP values for a single input row.

    Returns
    -------
    pd.DataFrame  columns: [feature, shap_value, abs_shap, direction]
                  sorted by abs_shap descending
    """
    sv = explainer.shap_values(input_df)
    if isinstance(sv, list):
        sv = sv[1]   # LightGBM binary: index 1 = positive class

    shap_df = pd.DataFrame({
        "feature":    feature_cols,
        "shap_value": sv[0],
        "input_val":  input_df.iloc[0].values,
    })
    shap_df["abs_shap"]  = shap_df["shap_value"].abs()
    shap_df["direction"] = np.where(shap_df["shap_value"] > 0, "↑ raises risk", "↓ lowers risk")
    return shap_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Plain-language explanation builder
# ─────────────────────────────────────────────────────────────

# Maps base signal names to readable phrases used in the insight text
SIGNAL_PHRASES = {
    "sleep_hours":        ("sleep duration", "hours of sleep", 7.0, False),
    "sleep_efficiency":   ("sleep efficiency", "% sleep efficiency", 85.0, False),
    "sleep_rem_ratio":    ("REM sleep proportion", "REM ratio", 0.2, False),
    "steps":              ("step count", "steps", 7000, False),
    "active_minutes":     ("active minutes", "active mins", 30, False),
    "resting_hr":         ("resting heart rate", "bpm resting HR", 65, True),
    "stress_score":       ("stress score", "stress score", 50, True),
    "rmssd":              ("heart rate variability", "ms HRV", 35, False),
    "spo2":               ("blood oxygen", "% SpO2", 96, False),
    "sedentary_fraction": ("sedentary time", "sedentary fraction", 0.5, True),
}


def build_plain_insight(risk_pct: float, shap_df: pd.DataFrame,
                        user_inputs: dict) -> str:
    """
    Generate a 2–3 sentence plain-language explanation of the risk score.

    Logic:
      1. Open with the risk level classification.
      2. Name the top 1–2 contributing signals with their actual values.
      3. Offer one actionable suggestion targeting the strongest driver.
    """
    top = shap_df[shap_df["abs_shap"] > 0].head(3)

    level = "low" if risk_pct < 78 else "moderate" if risk_pct < 89 else "elevated"
    opener = {
        "low":      f"Your current risk score of {risk_pct:.0f}% suggests a **low likelihood** of an anxiety trigger today.",
        "moderate": f"Your risk score of {risk_pct:.0f}% indicates a **moderate** chance of elevated anxiety — worth monitoring.",
        "elevated": f"Your risk score of {risk_pct:.0f}% signals an **elevated** probability of an anxiety trigger today.",
    }[level]

    # Build signal-specific commentary for the top driver
    lines = [opener]
    if len(top):
        feat_base = top.iloc[0]["feature"].split("_lag")[0].split("_roll")[0].split("_delta")[0]
        info = SIGNAL_PHRASES.get(feat_base)
        if info:
            phrase, unit, threshold, higher_is_worse = info
            val = user_inputs.get(feat_base)
            if val is not None:
                direction = "above" if val > threshold else "below"
                concern   = "which is pushing your risk up" if (
                    (higher_is_worse and val > threshold) or
                    (not higher_is_worse and val < threshold)
                ) else "which is helping keep your risk down"
                lines.append(
                    f"Your **{phrase}** ({val} {unit}) is {direction} the "
                    f"typical threshold ({threshold} {unit}), {concern}."
                )

    # Actionable suggestion based on the single biggest positive SHAP driver
    pos_drivers = top[top["shap_value"] > 0]
    if len(pos_drivers):
        top_driver_feat = pos_drivers.iloc[0]["feature"]
        base = top_driver_feat.split("_lag")[0].split("_roll")[0].split("_delta")[0]
        suggestions = {
            "stress_score":    "Try a 10-minute breathing exercise or short walk to reduce stress before it compounds overnight.",
            "resting_hr":      "An elevated resting HR can indicate insufficient recovery — consider an earlier bedtime tonight.",
            "sleep_hours":     "Even 30 more minutes of sleep tonight could measurably reduce tomorrow's trigger probability.",
            "sleep_efficiency":"Reduce screen time 1 hour before bed to improve sleep efficiency.",
            "active_minutes":  "A 20-minute walk today could meaningfully lower your risk score.",
            "steps":           "Aim for at least 7,000 steps — your step count is below the protective threshold.",
            "sedentary_fraction": "Breaking up sedentary time every 90 minutes has been shown to reduce anxiety markers.",
            "rmssd":           "Low HRV signals fatigue — prioritise rest and avoid intense exercise today.",
            "spo2":            "Low SpO2 may affect sleep quality. Ensure good ventilation in your sleep environment.",
        }
        suggestion = suggestions.get(base)
        if suggestion:
            lines.append(suggestion)

    return "  \n".join(lines)


# ─────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────

def shap_bar_chart(shap_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart of SHAP values for the top_n features.
    Bars are coloured red (raises risk) or teal (lowers risk).
    """
    df = shap_df.head(top_n).copy().sort_values("shap_value")
    colors = ["#f87171" if v > 0 else "#2dd4bf" for v in df["shap_value"]]
    clean_names = [
        f.replace("_lag1", " (yesterday)")
         .replace("_lag3", " (3d ago)")
         .replace("_lag7", " (7d ago)")
         .replace("_roll3", " (3d avg)")
         .replace("_roll7", " (7d avg)")
         .replace("_delta1", " (change)")
         .replace("_", " ")
        for f in df["feature"]
    ]

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=clean_names,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#111618",
        plot_bgcolor="#111618",
        font=dict(family="DM Mono, monospace", size=11, color="#7a9590"),
        xaxis=dict(
            gridcolor="#1e2729", zeroline=True, zerolinecolor="#2dd4bf",
            zerolinewidth=1, tickfont=dict(size=10)
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
        margin=dict(l=10, r=20, t=10, b=30),
        height=max(300, top_n * 36),
        bargap=0.3,
    )
    return fig


def gauge_chart(risk_pct: float) -> go.Figure:
    """Semi-circular gauge showing trigger risk percentage."""
    colour = "#2dd4bf" if risk_pct < 35 else "#f59e0b" if risk_pct < 65 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 42, "family": "DM Mono, monospace",
                                         "color": colour}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1,
                      "tickcolor": "#3d5652", "tickfont": {"size": 10}},
            "bar":   {"color": colour, "thickness": 0.25},
            "bgcolor": "#111618",
            "steps": [
                {"range": [0,  35], "color": "rgba(45,212,191,.08)"},
                {"range": [35, 65], "color": "rgba(245,158,11,.08)"},
                {"range": [65,100], "color": "rgba(248,113,113,.08)"},
            ],
            "threshold": {
                "line":  {"color": colour, "width": 3},
                "thickness": 0.75,
                "value": risk_pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e0f",
        font={"color": "#7a9590"},
        margin=dict(l=20, r=20, t=20, b=10),
        height=220,
    )
    return fig


def signal_radar_chart(user_inputs: dict) -> go.Figure:
    """
    Normalised radar chart showing today's signal values vs healthy baselines.
    Each signal is expressed as percentage of its healthy target.
    """
    BASELINES = {
        "Sleep hours":      (user_inputs.get("sleep_hours", 7),        8.0,   True),
        "Sleep efficiency": (user_inputs.get("sleep_efficiency", 85),  90.0,  True),
        "Active minutes":   (user_inputs.get("active_minutes", 30),    60.0,  True),
        "Steps (k)":        (user_inputs.get("steps", 6000) / 1000,    10.0,  True),
        "HRV (RMSSD)":      (user_inputs.get("rmssd", 35),             50.0,  True),
        "Low stress":       (100 - user_inputs.get("stress_score", 50), 80.0, True),
    }
    labels = list(BASELINES.keys())
    vals   = [min(100, v / t * 100) for v, t, _ in BASELINES.values()]
    vals  += [vals[0]]   # close the radar loop
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[100]*len(labels_closed), theta=labels_closed,
        fill="toself", fillcolor="rgba(45,212,191,.04)",
        line=dict(color="rgba(45,212,191,.15)", width=1),
        name="Target", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=labels_closed,
        fill="toself", fillcolor="rgba(45,212,191,.12)",
        line=dict(color="#2dd4bf", width=2),
        name="Today", hovertemplate="%{theta}: %{r:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#111618",
            radialaxis=dict(visible=True, range=[0, 110],
                            tickfont=dict(size=9, color="#3d5652"),
                            gridcolor="#1e2729"),
            angularaxis=dict(tickfont=dict(size=11, color="#7a9590"),
                             gridcolor="#1e2729"),
        ),
        paper_bgcolor="#0a0e0f",
        legend=dict(font=dict(size=10, color="#7a9590"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=30, b=20),
        height=300,
        showlegend=True,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Demo model fallback  (no training required to run the UI)
# ─────────────────────────────────────────────────────────────

def _build_demo_model():
    """
    Build a tiny synthetic LightGBM model for UI demonstration.
    Uses the 10 core wearable signals so the full UI renders correctly
    without requiring a trained model file.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=10,
                                n_informative=6, random_state=42)
    feature_cols = list(DIRECT_SIGNAL_MAP.keys())
    m = lgb.LGBMClassifier(n_estimators=50, verbose=-1, random_state=42)
    m.fit(X, y)

    # Wrap as a Booster for consistent API
    booster = m.booster_
    return booster, feature_cols, None


# ─────────────────────────────────────────────────────────────
# Sidebar — Signal Inputs
# ─────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render all user input controls in the sidebar.
    Returns a dict of {signal_name: value}.
    """
    with st.sidebar:
        st.markdown(
            "<div style='font-family:Fraunces,serif;font-size:1.4rem;"
            "font-weight:300;color:#2dd4bf;margin-bottom:2px'>gut but</div>"
            "<div style='font-family:DM Mono,monospace;font-size:0.6rem;"
            "letter-spacing:1.5px;color:#3d5652;margin-bottom:1.5rem'>"
            "TRIGGER INTELLIGENCE</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-header'>SLEEP SIGNALS</div>",
                    unsafe_allow_html=True)
        sleep_hours = st.slider(
            "Sleep duration (hours)", 2.0, 12.0, 7.0, 0.25,
            help="Total hours of sleep last night"
        )
        sleep_eff = st.slider(
            "Sleep efficiency (%)", 50, 100, 85, 1,
            help="Time asleep ÷ time in bed × 100"
        )
        rem_ratio = st.slider(
            "REM sleep ratio", 0.05, 0.40, 0.20, 0.01,
            help="Fraction of sleep time spent in REM stage"
        )

        st.markdown("<div class='section-header' style='margin-top:1rem'>ACTIVITY SIGNALS</div>",
                    unsafe_allow_html=True)
        steps = st.slider(
            "Daily step count", 500, 20000, 7000, 500,
            help="Total steps taken today"
        )
        active_mins = st.slider(
            "Active minutes", 0, 180, 35, 5,
            help="Fairly active + very active minutes"
        )
        sedentary = st.slider(
            "Sedentary fraction", 0.1, 1.0, 0.55, 0.01,
            help="Sedentary minutes ÷ 1440 (minutes in a day)"
        )

        st.markdown("<div class='section-header' style='margin-top:1rem'>PHYSIOLOGICAL SIGNALS</div>",
                    unsafe_allow_html=True)
        resting_hr = st.slider(
            "Resting heart rate (bpm)", 40, 110, 65, 1,
            help="Resting HR measured by Fitbit"
        )
        rmssd = st.slider(
            "HRV — RMSSD (ms)", 10, 100, 40, 1,
            help="Root mean square of successive differences — higher = better recovery"
        )
        spo2 = st.slider(
            "Blood oxygen SpO2 (%)", 88, 100, 97, 1,
            help="Average blood oxygen saturation from wearable"
        )
        stress_score = st.slider(
            "Stress score", 0, 100, 45, 1,
            help="Fitbit computed stress score (0 = calm, 100 = high stress)"
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        predict_btn = st.button("⟶  Analyse Triggers", key="predict")

    return {
        "sleep_hours":        sleep_hours,
        "sleep_efficiency":   sleep_eff,
        "sleep_rem_ratio":    rem_ratio,
        "steps":              steps,
        "active_minutes":     active_mins,
        "sedentary_fraction": sedentary,
        "resting_hr":         resting_hr,
        "rmssd":              rmssd,
        "spo2":               spo2,
        "stress_score":       stress_score,
        "_predict":           predict_btn,
    }


# ─────────────────────────────────────────────────────────────
# Main UI  — page layout
# ─────────────────────────────────────────────────────────────

def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            "<h1>Behavioral Trigger <em style='color:#2dd4bf'>Detection</em></h1>"
            "<p style='color:#7a9590;font-size:.9rem;margin-top:.3rem'>"
            "Enter today's wearable signals in the sidebar, then click "
            "<strong style='color:#2dd4bf'>Analyse Triggers</strong> to get "
            "your personalised risk score and SHAP-driven signal breakdown.</p>",
            unsafe_allow_html=True,
        )


def render_initial_state(user_inputs: dict):
    """Shown before the user runs a prediction."""
    st.markdown("<div class='section-header'>SIGNAL OVERVIEW</div>",
                unsafe_allow_html=True)
    st.plotly_chart(signal_radar_chart(user_inputs),
                    use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "<p style='color:#3d5652;font-size:.85rem;text-align:center;"
        "font-family:DM Mono,monospace'>Adjust sliders in the sidebar, "
        "then click Analyse Triggers.</p>",
        unsafe_allow_html=True,
    )


def render_results(user_inputs: dict, booster, feature_cols: list, calibrator):
    """Full results view after the user clicks Analyse."""
    # ── Build input vector & predict ──────────────────────────────────────
    input_df = build_input_vector(user_inputs, feature_cols)
    raw_p, cal_p = predict_risk(input_df, booster, calibrator)
    risk_pct = cal_p * 100

    # ── Risk level ────────────────────────────────────────────────────────
    level     = "low" if risk_pct < 78 else "medium" if risk_pct < 89 else "high"
    level_txt = {"low": "LOW RISK", "medium": "MODERATE RISK", "high": "HIGH RISK"}[level]

    # ── SHAP ──────────────────────────────────────────────────────────────
    explainer = get_shap_explainer(booster)
    shap_df   = compute_shap(input_df, explainer, feature_cols)

    # ── Plain-language insight ────────────────────────────────────────────
    insight = build_plain_insight(risk_pct, shap_df, user_inputs)

    # ── Layout — top row ──────────────────────────────────────────────────
    col_gauge, col_risk, col_radar = st.columns([1.2, 1.8, 1.5])

    with col_gauge:
        st.plotly_chart(gauge_chart(risk_pct),
                        use_container_width=True,
                        config={"displayModeBar": False})

    with col_risk:
        st.markdown(
            f"<div class='risk-card risk-{level}'>"
            f"  <div class='risk-label'>{level_txt}</div>"
            f"  <div class='risk-pct'>{risk_pct:.0f}%</div>"
            f"  <div class='insight-text'>{insight}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_radar:
        st.markdown("<div class='section-header'>TODAY vs TARGETS</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(signal_radar_chart(user_inputs),
                        use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── SHAP breakdown ────────────────────────────────────────────────────
    col_shap, col_table = st.columns([1.8, 1.2])

    with col_shap:
        st.markdown("<div class='section-header'>SIGNAL CONTRIBUTION (SHAP)</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#3d5652;font-size:.8rem;font-family:DM Mono,monospace;"
            "margin-bottom:.6rem'>"
            "<span style='color:#f87171'>■</span> raises risk &nbsp;&nbsp; "
            "<span style='color:#2dd4bf'>■</span> lowers risk</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            shap_bar_chart(shap_df, top_n=12),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_table:
        st.markdown("<div class='section-header'>TOP SIGNAL DRIVERS</div>",
                    unsafe_allow_html=True)

        top5 = shap_df.head(5)
        for _, row in top5.iterrows():
            base = row["feature"].split("_lag")[0].split("_roll")[0].split("_delta")[0]
            info = SIGNAL_PHRASES.get(base)
            label = info[0] if info else row["feature"].replace("_", " ")
            val   = user_inputs.get(base)
            val_str = f"{val}" if val is not None else "—"
            arrow = "↑" if row["shap_value"] > 0 else "↓"
            arrow_col = "#f87171" if row["shap_value"] > 0 else "#2dd4bf"

            st.markdown(
                f"<div style='display:flex;align-items:flex-start;gap:10px;"
                f"padding:10px 0;border-bottom:1px solid #1e2729'>"
                f"  <span style='font-size:1.1rem;color:{arrow_col};width:18px'>{arrow}</span>"
                f"  <div style='flex:1'>"
                f"    <div style='font-size:.8rem;color:#e8f0ee'>{label}</div>"
                f"    <div style='font-family:DM Mono,monospace;font-size:.7rem;"
                f"color:#3d5652'>value: {val_str} &nbsp;·&nbsp; "
                f"SHAP: {row['shap_value']:+.4f}</div>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Top 2 risk-raising signals as tags
        pos = shap_df[shap_df["shap_value"] > 0].head(2)
        if len(pos):
            st.markdown(
                "<div style='margin-top:1rem'>"
                "<div class='section-header' style='margin-bottom:.5rem'>KEY TRIGGERS</div>"
                + "".join(
                    f"<span class='signal-tag'>{r['feature'].replace('_',' ')}</span>"
                    for _, r in pos.iterrows()
                )
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── Optional: full SHAP table ─────────────────────────────────────────
    with st.expander("View full SHAP table", expanded=False):
        display_cols = ["feature", "input_val", "shap_value", "abs_shap", "direction"]
        display_cols = [c for c in display_cols if c in shap_df.columns]
        st.dataframe(
            shap_df[display_cols].head(20)
                .style.format({"shap_value": "{:+.4f}", "abs_shap": "{:.4f}",
                               "input_val": "{:.2f}"})
                .background_gradient(subset=["abs_shap"], cmap="Greens"),
            use_container_width=True,
            hide_index=True,
        )

    # ── Optional: pre-computed global impact table ────────────────────────
    impact_df = load_impact_table()
    if impact_df is not None:
        with st.expander("View global feature impact (from training run)", expanded=False):
            st.markdown(
                "<p style='color:#7a9590;font-size:.8rem'>"
                "These are population-level SHAP values computed over the "
                "full training dataset — not specific to today's inputs.</p>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                impact_df.head(15)
                    .style.format({"mean_abs_shap": "{:.4f}", "delta_shap": "{:+.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

    # ── Metrics row ───────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>INPUT SUMMARY</div>",
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sleep",     f"{user_inputs['sleep_hours']:.1f}h",
              delta="▼ below target" if user_inputs["sleep_hours"] < 7 else "✓ on target",
              delta_color="inverse" if user_inputs["sleep_hours"] < 7 else "normal")
    m2.metric("Steps",     f"{user_inputs['steps']:,}",
              delta="▼ low" if user_inputs["steps"] < 7000 else "✓ good",
              delta_color="inverse" if user_inputs["steps"] < 7000 else "normal")
    m3.metric("Stress",    f"{user_inputs['stress_score']}",
              delta="▲ elevated" if user_inputs["stress_score"] > 60 else "✓ calm",
              delta_color="inverse" if user_inputs["stress_score"] > 60 else "normal")
    m4.metric("HRV",       f"{user_inputs['rmssd']} ms",
              delta="▼ low" if user_inputs["rmssd"] < 30 else "✓ good",
              delta_color="inverse" if user_inputs["rmssd"] < 30 else "normal")
    m5.metric("Risk Score",f"{risk_pct:.0f}%",
              delta=level_txt, delta_color="inverse" if risk_pct > 78 else "normal")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    # Load model artifacts
    model_artifacts = load_model()
    booster, feature_cols, calibrator = model_artifacts

    # Render sidebar inputs
    user_inputs = render_sidebar()
    predict = user_inputs.pop("_predict")

    # Page header
    render_header()

    # Main content
    if not predict:
        render_initial_state(user_inputs)
    else:
        with st.spinner("Computing trigger risk…"):
            render_results(user_inputs, booster, feature_cols, calibrator)


if __name__ == "__main__":
    main()
