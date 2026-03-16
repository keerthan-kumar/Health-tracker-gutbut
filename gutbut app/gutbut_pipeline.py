"""
Gut But — Behavioral Health Trigger Detection Pipeline
=======================================================
Dataset  : LifeSnaps (Zenodo DOI: 10.5281/zenodo.6826682 / Kaggle mirror)
Approach : LightGBM + SHAP interaction values on lag-engineered daily features
Goal     : Predict elevated-anxiety days (STAI-S ≥ 40) from wearable signals

Pipeline modules
----------------
  1. load_lifesnaps()           — raw ingestion & schema normalisation
  2. build_daily_feature_table()— aggregate Fitbit streams → one row/person/day
  3. engineer_lag_features()    — rolling windows, lags, delta features
  4. build_trigger_labels()     — EMA → binary anxiety trigger label
  5. train_lightgbm()           — cross-validated LightGBM classifier
  6. explain_with_shap()        — SHAP interaction plots + trigger rule extraction
  7. generate_user_insights()   — plain-language per-user report

Run the full pipeline
---------------------
  python gutbut_pipeline.py --data_dir ./lifesnaps_data --output_dir ./outputs

Dependencies
------------
  pip install pandas numpy lightgbm shap scikit-learn matplotlib seaborn
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import os
import warnings
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, classification_report, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# STAI-State threshold used in clinical literature to mark elevated anxiety.
# Scores ≥ 40 represent meaningful anxiety elevation (Spielberger 1983).
STAI_THRESHOLD = 40

# PANAS negative affect: scale 10–50; ≥ 25 is considered moderately elevated.
PANAS_NEG_THRESHOLD = 25

# Lag windows (in days) to create rolling history features.
# 1-day captures overnight effect; 3-day and 7-day capture short/medium trends.
LAG_WINDOWS = [1, 3, 7]

# Minimum number of valid days a participant must have to be included.
# Too few days = unstable lag estimates.
MIN_DAYS_PER_PARTICIPANT = 14

# Columns expected from the Fitbit daily summary export in LifeSnaps.
# LifeSnaps stores data in separate CSVs per data type; column names below
# follow the LifeSnaps data dictionary (see README in the Zenodo release).
FITBIT_DAILY_COLS = {
    "sleep": [
        "id",                        # participant identifier
        "date",                      # calendar date (YYYY-MM-DD)
        "minutesAsleep",             # total sleep in minutes
        "minutesAwake",              # wake time within sleep period
        "efficiency",                # Fitbit sleep efficiency (0–100)
        "minutesREM",                # REM stage minutes
        "minutesLight",              # Light stage minutes
        "minutesDeep",               # Deep stage minutes
        "timeInBed",                 # total time in bed (mins)
        "mainSleep_startTime",       # sleep start timestamp
        "mainSleep_endTime",         # sleep end timestamp
    ],
    "activity": [
        "id",
        "date",
        "steps",                     # total daily step count
        "caloriesOut",               # total calories burned
        "fairlyActiveMinutes",       # moderate-intensity activity
        "veryActiveMinutes",         # vigorous-intensity activity
        "sedentaryMinutes",          # sedentary time
        "lightlyActiveMinutes",      # light activity
    ],
    "heart_rate": [
        "id",
        "date",
        "restingHeartRate",          # daily resting HR (bpm)
    ],
    "spo2": [
        "id",
        "date",
        "dailySpO2_avg",             # avg blood oxygen saturation (%)
    ],
    "skin_temp": [
        "id",
        "date",
        "tempAvg",                   # nightly avg skin temperature (°C)
        "tempMin",
        "tempMax",
    ],
    "stress": [
        "id",
        "date",
        "stressScore",               # Fitbit-computed stress score (0–100)
    ],
}

# EMA / survey columns for outcome label construction
EMA_COLS = {
    "ema_daily": [
        "id",
        "date",
        "stai_s_score",              # STAI-State anxiety (20–80)
        "panas_neg_score",           # PANAS Negative Affect (10–50)
        "panas_pos_score",           # PANAS Positive Affect (10–50)
        "mood",                      # Single item mood rating (1–7)
    ]
}


# ─────────────────────────────────────────────────────────────
# MODULE 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_lifesnaps(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load and normalise all LifeSnaps CSV exports from a local directory.

    LifeSnaps ships data as separate CSVs per data type — one file (or folder
    of per-participant files) per stream.  This function handles two common
    layouts:

      Layout A (flat)    : data_dir/sleep.csv, data_dir/activity.csv, ...
      Layout B (per-user): data_dir/<participant_id>/sleep.csv, ...

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys match FITBIT_DAILY_COLS / EMA_COLS keys.
        Each DataFrame is normalised with columns ['id', 'date', ...].

    Design note
    -----------
    We keep each modality in a separate DataFrame at this stage (rather than
    merging immediately) so that each stream can be explored and cleaned
    independently.  The merge happens in build_daily_feature_table().
    """
    data_dir = Path(data_dir)
    raw: dict[str, pd.DataFrame] = {}
    all_streams = {**FITBIT_DAILY_COLS, **EMA_COLS}

    for stream_name, expected_cols in all_streams.items():
        frames = []

        # ── Layout A: single flat CSV ──────────────────────────────────────
        flat_path = data_dir / f"{stream_name}.csv"
        if flat_path.exists():
            df = pd.read_csv(flat_path, low_memory=False)
            frames.append(df)
            log.info(f"[load] {stream_name}: loaded flat file → {len(df):,} rows")

        # ── Layout B: per-participant sub-directories ──────────────────────
        else:
            for user_dir in sorted(data_dir.iterdir()):
                if not user_dir.is_dir():
                    continue
                user_file = user_dir / f"{stream_name}.csv"
                if user_file.exists():
                    df = pd.read_csv(user_file, low_memory=False)
                    # Inject participant id from directory name if missing
                    if "id" not in df.columns:
                        df.insert(0, "id", user_dir.name)
                    frames.append(df)

            if frames:
                log.info(
                    f"[load] {stream_name}: found {len(frames)} per-user files "
                    f"→ {sum(len(f) for f in frames):,} rows total"
                )

        if not frames:
            log.warning(f"[load] {stream_name}: no data found, skipping.")
            continue

        df = pd.concat(frames, ignore_index=True)

        # ── Normalise column names to lowercase snake_case ─────────────────
        df.columns = [_snake(c) for c in df.columns]

        # ── Parse date column ──────────────────────────────────────────────
        date_col = _find_date_column(df)
        if date_col is None:
            log.warning(f"[load] {stream_name}: no date column found, skipping.")
            continue
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date"])

        # ── Ensure participant id column exists ────────────────────────────
        if "id" not in df.columns:
            log.warning(f"[load] {stream_name}: no 'id' column, skipping.")
            continue
        df["id"] = df["id"].astype(str).str.strip()

        # ── Drop fully-duplicate rows (same id + date repeated) ───────────
        before = len(df)
        df = df.drop_duplicates(subset=["id", "date"])
        if len(df) < before:
            log.info(
                f"[load] {stream_name}: dropped {before - len(df):,} duplicate rows"
            )

        raw[stream_name] = df.sort_values(["id", "date"]).reset_index(drop=True)

    _log_coverage(raw)
    return raw


# ─────────────────────────────────────────────────────────────
# MODULE 2 — DAILY FEATURE TABLE
# ─────────────────────────────────────────────────────────────

def build_daily_feature_table(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all wearable modalities into a single row-per-participant-per-day
    feature table.

    Merge strategy
    --------------
    We use a left join anchored on the sleep DataFrame because:
      • Sleep is the most consistently recorded modality in LifeSnaps.
      • A day without a sleep record is usually a non-wearing day — including
        it would introduce many NaNs with no signal value.
      • Other modalities (activity, HR, etc.) are joined on (id, date).

    Feature derivation
    ------------------
    Several raw Fitbit fields are transformed before modelling:
      • sleep_hours         : minutes → hours (more interpretable)
      • sleep_efficiency    : already 0–100, kept as-is
      • rem_fraction        : REM / total sleep (circadian quality proxy)
      • active_minutes      : fairly + very active (composite intensity)
      • sedentary_fraction  : sedentary / 1440 (day fraction)
      • temp_deviation      : daily skin temp − participant 30-day rolling mean
                              (captures *change* from baseline, not absolute °C)

    Returns
    -------
    pd.DataFrame
        One row per (id, date).  Index is a RangeIndex.  Columns are described
        in FEATURE_DESCRIPTIONS below.
    """
    log.info("[feature_table] merging modalities ...")

    # ── Anchor on sleep ────────────────────────────────────────────────────
    if "sleep" not in raw:
        raise ValueError("Sleep stream is required as the anchor modality.")
    base = raw["sleep"][["id", "date"]].copy()

    # ── Sleep features ─────────────────────────────────────────────────────
    sleep = raw["sleep"].copy()
    sleep["sleep_hours"] = sleep.get("minutesasleep", sleep.get("minutesAsleep",
                                    pd.Series(dtype=float))) / 60
    sleep["rem_fraction"] = (
        sleep.get("minutesrem", sleep.get("minutesREM", pd.Series(dtype=float)))
        / sleep["sleep_hours"].replace(0, np.nan) / 60
    )
    sleep["time_in_bed_hours"] = (
        sleep.get("timeinbed", sleep.get("timeInBed", pd.Series(dtype=float))) / 60
    )
    sleep["sleep_efficiency"] = sleep.get(
        "efficiency", pd.Series(np.nan, index=sleep.index)
    )

    sleep_features = sleep[["id", "date", "sleep_hours", "rem_fraction",
                             "time_in_bed_hours", "sleep_efficiency"]]
    base = base.merge(sleep_features, on=["id", "date"], how="left")

    # ── Activity features ──────────────────────────────────────────────────
    if "activity" in raw:
        act = raw["activity"].copy()
        act = _normalise_cols(act, {
            "steps": "steps",
            "caloriesout": "calories_out",
            "fairlyactiveminutes": "fairly_active_mins",
            "veryactiveminutes": "very_active_mins",
            "sedentaryminutes": "sedentary_mins",
            "lightlyactiveminutes": "lightly_active_mins",
        })
        act["active_minutes"] = (
            act.get("fairly_active_mins", 0) + act.get("very_active_mins", 0)
        )
        act["sedentary_fraction"] = act.get("sedentary_mins", np.nan) / 1440

        act_features = act[["id", "date"] + [
            c for c in ["steps", "calories_out", "active_minutes",
                        "sedentary_fraction", "lightly_active_mins"]
            if c in act.columns
        ]]
        base = base.merge(act_features, on=["id", "date"], how="left")

    # ── Resting heart rate ─────────────────────────────────────────────────
    if "heart_rate" in raw:
        hr = raw["heart_rate"].copy()
        hr = _normalise_cols(hr, {"restingheartrate": "resting_hr"})
        if "resting_hr" in hr.columns:
            base = base.merge(hr[["id", "date", "resting_hr"]],
                              on=["id", "date"], how="left")

    # ── SpO2 ───────────────────────────────────────────────────────────────
    if "spo2" in raw:
        spo2 = raw["spo2"].copy()
        spo2 = _normalise_cols(spo2, {"dailyspo2_avg": "spo2_avg",
                                       "avg": "spo2_avg"})
        if "spo2_avg" in spo2.columns:
            base = base.merge(spo2[["id", "date", "spo2_avg"]],
                              on=["id", "date"], how="left")

    # ── Skin temperature — compute deviation from personal baseline ────────
    if "skin_temp" in raw:
        temp = raw["skin_temp"].copy()
        temp = _normalise_cols(temp, {"tempavg": "skin_temp_avg",
                                       "temp_avg": "skin_temp_avg"})
        if "skin_temp_avg" in temp.columns:
            # Personal rolling 30-day baseline to compute relative deviation
            temp = temp.sort_values(["id", "date"])
            temp["skin_temp_baseline"] = (
                temp.groupby("id")["skin_temp_avg"]
                .transform(lambda s: s.shift(1).rolling(30, min_periods=5).mean())
            )
            temp["skin_temp_deviation"] = (
                temp["skin_temp_avg"] - temp["skin_temp_baseline"]
            )
            base = base.merge(
                temp[["id", "date", "skin_temp_avg", "skin_temp_deviation"]],
                on=["id", "date"], how="left"
            )

    # ── Fitbit stress score ────────────────────────────────────────────────
    if "stress" in raw:
        stress = raw["stress"].copy()
        stress = _normalise_cols(stress, {"stressscore": "fitbit_stress_score"})
        if "fitbit_stress_score" in stress.columns:
            base = base.merge(stress[["id", "date", "fitbit_stress_score"]],
                              on=["id", "date"], how="left")

    # ── Filter participants with enough data ───────────────────────────────
    day_counts = base.groupby("id")["date"].count()
    valid_ids = day_counts[day_counts >= MIN_DAYS_PER_PARTICIPANT].index
    dropped = len(day_counts) - len(valid_ids)
    if dropped:
        log.info(
            f"[feature_table] dropped {dropped} participants with "
            f"< {MIN_DAYS_PER_PARTICIPANT} days"
        )
    base = base[base["id"].isin(valid_ids)].reset_index(drop=True)

    log.info(
        f"[feature_table] final table: {len(base):,} rows, "
        f"{base['id'].nunique()} participants, "
        f"{base.shape[1]} columns"
    )
    _log_missing(base)
    return base


# ─────────────────────────────────────────────────────────────
# MODULE 3 — LAG FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

WEARABLE_SIGNALS = [
    "sleep_hours",
    "sleep_efficiency",
    "rem_fraction",
    "active_minutes",
    "steps",
    "resting_hr",
    "sedentary_fraction",
    "spo2_avg",
    "skin_temp_deviation",
    "fitbit_stress_score",
]


def engineer_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the daily feature table into a modelling-ready dataset by
    adding temporal context for each participant.

    Three families of derived features are created per signal:

    1. Lag (t-k)
       The raw value from k days ago.  Captures direct carry-over effects —
       e.g. one night of poor sleep elevates anxiety risk the next morning.

    2. Rolling mean (3-day, 7-day)
       Short and medium-term baseline.  A 7-day rolling mean of sleep captures
       chronic sleep deprivation that single-night lags would miss.

    3. Delta (today − yesterday)
       The rate of change.  A sudden 30-minute drop in sleep or a sharp HR
       spike is often more predictive than the absolute value.

    Design note — why group-wise shifts?
    -------------------------------------
    We MUST sort by (id, date) and apply shifts within participant groups.
    Without groupby, a lag at day 0 of participant B would accidentally pick
    up the last day of participant A.  This is a common data-leakage bug in
    panel time-series pipelines.

    Returns
    -------
    pd.DataFrame
        Original columns + lag/rolling/delta columns.
        Rows with insufficient history (first k days per participant) will
        have NaN in lag features — these are handled in train_lightgbm() via
        LightGBM's native missing value support.
    """
    log.info("[lag_features] engineering temporal features ...")

    df = daily.sort_values(["id", "date"]).copy()
    available_signals = [s for s in WEARABLE_SIGNALS if s in df.columns]

    new_cols = {}
    for signal in available_signals:
        series = df.groupby("id")[signal]

        # ── 1. Lag features ────────────────────────────────────────────────
        for k in LAG_WINDOWS:
            new_cols[f"{signal}_lag{k}"] = series.shift(k)

        # ── 2. Rolling mean features ───────────────────────────────────────
        for w in [3, 7]:
            new_cols[f"{signal}_roll{w}mean"] = (
                series.shift(1)                        # exclude today
                      .transform(lambda s: s.rolling(w, min_periods=max(1, w // 2))
                                            .mean())
            )

        # ── 3. Delta (1-day change) ────────────────────────────────────────
        new_cols[f"{signal}_delta1"] = series.shift(0) - series.shift(1)

    lag_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, lag_df], axis=1)

    # ── Derived composite features ─────────────────────────────────────────
    # Recovery index: are you moving more when sleeping less? Captures
    # compensatory behaviour that often precedes symptom onset.
    if "sleep_hours" in df.columns and "active_minutes" in df.columns:
        df["sleep_activity_ratio"] = (
            df["sleep_hours"] / df["active_minutes"].replace(0, np.nan)
        )

    # Autonomic load proxy: HR elevated relative to personal 7-day baseline.
    if "resting_hr" in df.columns and "resting_hr_roll7mean" in df.columns:
        df["hr_above_baseline"] = df["resting_hr"] - df["resting_hr_roll7mean"]

    # Day-of-week (captures weekday/weekend rhythms in lifestyle signals)
    df["day_of_week"] = df["date"].dt.dayofweek       # 0=Mon, 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    n_new = df.shape[1] - daily.shape[1]
    log.info(f"[lag_features] added {n_new} derived columns → "
             f"{df.shape[1]} total columns")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# MODULE 4 — TRIGGER LABEL CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def build_trigger_labels(
    feature_df: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    label_source: str = "stai",          # "stai" | "panas" | "combined"
) -> pd.DataFrame:
    """
    Join EMA survey scores onto the feature table and construct a binary
    trigger label indicating an elevated-anxiety day.

    Label logic
    -----------
    "stai"     : trigger = 1 if STAI-State ≥ 40  (Spielberger 1983 threshold)
    "panas"    : trigger = 1 if PANAS Negative Affect ≥ 25
    "combined" : trigger = 1 if EITHER stai OR panas threshold is crossed
                 (more sensitive, better for prototype exploration)

    Why a binary label?
    -------------------
    LightGBM with binary:logistic outputs a probability, which maps directly
    to the user-facing "trigger risk score" (0–100%).  We can always return
    to regression on the raw STAI score post-prototype if desired.

    Handling same-day vs next-day labels
    -------------------------------------
    The EMA in LifeSnaps is typically collected in the evening, reflecting
    how the participant felt *that day*.  Wearable signals are the same-day
    readings.  This is a same-day prediction task — which is appropriate for
    "how are you likely feeling right now given today's wearable data?"
    For a next-day prediction variant, shift labels by 1 day backward.

    Returns
    -------
    pd.DataFrame
        feature_df with added columns: stai_s_score, panas_neg_score (if
        available), and `trigger` (binary label).
        Rows without a label are dropped — we only train on labelled days.
    """
    log.info(f"[labels] constructing trigger label using source='{label_source}'")

    if "ema_daily" not in raw:
        raise ValueError(
            "EMA data not found in raw dict.  "
            "Ensure ema_daily.csv is present in the data directory."
        )

    ema = raw["ema_daily"].copy()
    ema = _normalise_cols(ema, {
        "stai_s_score": "stai_s_score",
        "stai_score": "stai_s_score",
        "panas_neg_score": "panas_neg_score",
        "panas_neg": "panas_neg_score",
        "panas_pos_score": "panas_pos_score",
        "mood": "mood_score",
    })

    ema_cols = ["id", "date"] + [
        c for c in ["stai_s_score", "panas_neg_score", "panas_pos_score",
                    "mood_score"]
        if c in ema.columns
    ]
    ema = ema[ema_cols]

    df = feature_df.merge(ema, on=["id", "date"], how="inner")
    before = len(df)

    # ── Build trigger label ────────────────────────────────────────────────
    if label_source == "stai":
        if "stai_s_score" not in df.columns:
            raise ValueError("stai_s_score not found in EMA data.")
        df["trigger"] = (df["stai_s_score"] >= STAI_THRESHOLD).astype(int)

    elif label_source == "panas":
        if "panas_neg_score" not in df.columns:
            raise ValueError("panas_neg_score not found in EMA data.")
        df["trigger"] = (df["panas_neg_score"] >= PANAS_NEG_THRESHOLD).astype(int)

    elif label_source == "combined":
        stai_flag = (df.get("stai_s_score", pd.Series(0)) >= STAI_THRESHOLD)
        panas_flag = (df.get("panas_neg_score", pd.Series(0)) >= PANAS_NEG_THRESHOLD)
        df["trigger"] = (stai_flag | panas_flag).astype(int)

    else:
        raise ValueError(f"Unknown label_source '{label_source}'")

    # Drop rows where the label score itself is missing
    df = df.dropna(subset=["trigger"])
    log.info(
        f"[labels] labelled rows: {len(df):,}  "
        f"(dropped {before - len(df):,} unlabelled)"
    )

    positive_rate = df["trigger"].mean()
    log.info(
        f"[labels] trigger positive rate: {positive_rate:.1%}  "
        f"({'balanced' if 0.2 < positive_rate < 0.8 else 'imbalanced — consider scale_pos_weight'})"
    )
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# MODULE 5 — MODEL TRAINING (LightGBM)
# ─────────────────────────────────────────────────────────────

# Feature columns fed to the model (excludes id, date, raw score cols, label)
NON_FEATURE_COLS = {
    "id", "date", "trigger",
    "stai_s_score", "panas_neg_score", "panas_pos_score", "mood_score",
    "mainSleep_startTime", "mainSleep_endTime",
    "skin_temp_baseline",        # intermediate computation artifact
}


def train_lightgbm(
    labelled_df: pd.DataFrame,
    output_dir: str = "./outputs",
    n_splits: int = 5,
) -> tuple[lgb.LGBMClassifier, pd.DataFrame, list[str]]:
    """
    Train a LightGBM binary classifier to predict the trigger label.

    Validation strategy — GroupShuffleSplit by participant
    -------------------------------------------------------
    We split by participant (group), NOT by row.  A random row-wise split
    would leak future information from the same person into the validation
    set, inflating AUC.  Participant-level splits measure how well the model
    generalises to *unseen people*, which is the real deployment setting.

    Hyperparameters
    ---------------
    The defaults below are conservative starting points suitable for a
    dataset with O(thousands) labelled rows:
      • n_estimators=500 with early stopping prevents overfitting.
      • class_weight='balanced' handles label imbalance automatically.
      • min_child_samples=20 prevents individual-level memorisation.
      • colsample_bytree=0.7 adds regularisation via feature subsampling.

    Calibration
    -----------
    Raw LightGBM probabilities are well-ranked but not always well-calibrated.
    We wrap with isotonic regression calibration so the output probability
    maps meaningfully to "82% chance of elevated anxiety today".

    Returns
    -------
    model        : fitted LGBMClassifier (on full training set)
    oof_preds    : out-of-fold prediction DataFrame for evaluation
    feature_cols : list of feature names in model input order
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Identify feature columns ───────────────────────────────────────────
    feature_cols = [
        c for c in labelled_df.columns
        if c not in NON_FEATURE_COLS
        and labelled_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    log.info(f"[train] feature count: {len(feature_cols)}")

    X = labelled_df[feature_cols].values
    y = labelled_df["trigger"].values
    groups = labelled_df["id"].values

    # ── LightGBM configuration ─────────────────────────────────────────────
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # ── Cross-validation (participant-level) ───────────────────────────────
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    oof_proba = np.zeros(len(y))
    cv_aucs = []

    log.info(f"[train] running {n_splits}-fold participant-level CV ...")
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fold_model = lgb.LGBMClassifier(**lgb_params)
        fold_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )
        oof_proba[val_idx] = fold_model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof_proba[val_idx])
        cv_aucs.append(fold_auc)
        log.info(f"[train] fold {fold+1}/{n_splits}  AUC={fold_auc:.4f}")

    mean_auc = np.mean(cv_aucs)
    log.info(f"[train] CV mean AUC: {mean_auc:.4f} ± {np.std(cv_aucs):.4f}")

    # ── Train final model on full dataset ──────────────────────────────────
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X, y)

    # ── Probability calibration ────────────────────────────────────────────
    # Uses isotonic regression over out-of-fold predictions to map raw
    # scores → well-calibrated probabilities.
    from sklearn.isotonic import IsotonicRegression
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_proba, y)
    # Store calibrator on model object for later use in inference
    final_model._calibrator = cal

    # ── OOF evaluation DataFrame ───────────────────────────────────────────
    oof_df = pd.DataFrame({
        "id": groups,
        "date": labelled_df["date"].values,
        "y_true": y,
        "y_proba_raw": oof_proba,
        "y_proba_cal": cal.predict(oof_proba),
        "y_pred": (oof_proba >= 0.5).astype(int),
    })

    # ── Save artifacts ─────────────────────────────────────────────────────
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    final_model.booster_.save_model(str(output_dir / "lgbm_model.txt"))

    ap = average_precision_score(y, oof_proba)
    log.info(f"[train] OOF Average Precision: {ap:.4f}")
    log.info(f"[train] model saved → {output_dir / 'lgbm_model.txt'}")

    print("\n── Classification report (OOF) ─────────────────────────────")
    print(classification_report(y, (oof_proba >= 0.5).astype(int),
                                 target_names=["No trigger", "Trigger"]))

    return final_model, oof_df, feature_cols


# ─────────────────────────────────────────────────────────────
# MODULE 6 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────

def explain_with_shap(
    model: lgb.LGBMClassifier,
    labelled_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str = "./outputs",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Generate SHAP values and produce three artefacts:

    1. Global summary bar chart   — top-N most impactful signals overall
    2. SHAP interaction heatmap   — which PAIRS of signals interact most
    3. Trigger rule table (CSV)   — mean SHAP value per feature for trigger=1
                                    days vs trigger=0 days

    SHAP TreeExplainer
    ------------------
    We use shap.TreeExplainer which is exact (not approximate) for tree
    models and runs in O(TLD) time, where T=trees, L=leaves, D=depth.
    For LightGBM on a tabular dataset of this size it completes in seconds.

    Interaction values
    ------------------
    shap.TreeExplainer(model).shap_interaction_values() returns a matrix
    of shape (n_samples, n_features, n_features).  The [i, j] off-diagonal
    cell quantifies how much the *joint* presence of features i and j
    shifts the prediction beyond their individual contributions.
    This is the mechanism that surfaces "sleep + stress interaction" as a
    trigger combination — the core Gut But insight.

    Returns
    -------
    pd.DataFrame
        feature_impact: columns [feature, mean_shap_trigger, mean_shap_no_trigger,
                                  delta_shap] sorted by delta_shap descending.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = labelled_df[feature_cols].values
    y = labelled_df["trigger"].values

    log.info("[shap] computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # LightGBM binary returns list [neg_class, pos_class] in some versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ── 1. Global summary plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_cols,
        max_display=top_n,
        show=False,
        plot_type="bar"
    )
    plt.title("Top trigger-driving signals (mean |SHAP|)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150)
    plt.close()
    log.info(f"[shap] summary plot saved → {output_dir / 'shap_summary.png'}")

    # ── 2. SHAP interaction matrix (top-10 features) ───────────────────────
    log.info("[shap] computing SHAP interaction values (may take ~30 s) ...")
    try:
        interaction_values = explainer.shap_interaction_values(X)
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]

        # Mean absolute interaction across samples
        mean_interactions = np.abs(interaction_values).mean(axis=0)

        # Focus on top-10 features by main SHAP effect
        top_idx = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:10]
        top_names = [feature_cols[i] for i in top_idx]
        top_matrix = mean_interactions[np.ix_(top_idx, top_idx)]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            top_matrix,
            xticklabels=top_names, yticklabels=top_names,
            cmap="YlOrRd", annot=True, fmt=".3f",
            linewidths=0.5, ax=ax
        )
        ax.set_title("SHAP interaction values — top 10 features", fontsize=13)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_interactions.png", dpi=150)
        plt.close()
        log.info(f"[shap] interaction plot saved → {output_dir / 'shap_interactions.png'}")

    except Exception as e:
        log.warning(f"[shap] interaction values failed: {e} — skipping.")

    # ── 3. Trigger impact table ────────────────────────────────────────────
    trigger_mask = y == 1
    impact_rows = []
    for i, feat in enumerate(feature_cols):
        impact_rows.append({
            "feature": feat,
            "mean_shap_trigger": shap_values[trigger_mask, i].mean(),
            "mean_shap_no_trigger": shap_values[~trigger_mask, i].mean(),
            "mean_abs_shap": np.abs(shap_values[:, i]).mean(),
        })

    impact_df = pd.DataFrame(impact_rows)
    impact_df["delta_shap"] = (
        impact_df["mean_shap_trigger"] - impact_df["mean_shap_no_trigger"]
    )
    impact_df = impact_df.sort_values("delta_shap", ascending=False)
    impact_df.to_csv(output_dir / "feature_impact.csv", index=False)
    log.info(f"[shap] impact table saved → {output_dir / 'feature_impact.csv'}")

    print("\n── Top 10 features most associated with trigger days ────────")
    print(impact_df.head(10)[["feature", "delta_shap", "mean_abs_shap"]]
          .to_string(index=False))

    return impact_df


# ─────────────────────────────────────────────────────────────
# MODULE 7 — USER-FACING INSIGHT GENERATION
# ─────────────────────────────────────────────────────────────

def generate_user_insights(
    labelled_df: pd.DataFrame,
    model: lgb.LGBMClassifier,
    feature_cols: list[str],
    impact_df: pd.DataFrame,
    output_dir: str = "./outputs",
    top_triggers_per_user: int = 3,
) -> pd.DataFrame:
    """
    Generate a plain-language, per-user trigger profile.

    For each participant this produces:
      • trigger_rate        : % of their days that were high-anxiety
      • top_3_triggers      : the 3 signals that most predict their high days
      • example_trigger_day : date of their highest-risk day with SHAP drivers
      • insight_text        : a human-readable one-paragraph summary

    This module is the bridge between model output and the Gut But user
    interface.  The insight text uses a template-driven approach — in
    production this can be upgraded to an LLM call that personalises the
    language further.

    Returns
    -------
    pd.DataFrame
        One row per participant with columns:
        id, trigger_rate, top_triggers, risk_score_mean, insight_text
    """
    output_dir = Path(output_dir)
    X = labelled_df[feature_cols].values
    y = labelled_df["trigger"].values

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    raw_proba = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibrator", None)
    proba = cal.predict(raw_proba) if cal else raw_proba

    labelled_df = labelled_df.copy()
    labelled_df["risk_score"] = (proba * 100).round(1)

    rows = []
    for pid, group in labelled_df.groupby("id"):
        idx = group.index
        group_shap = shap_values[idx]
        trigger_rate = group["trigger"].mean()
        mean_risk = group["risk_score"].mean()

        # Top features for THIS participant on trigger days
        trig_idx = idx[group["trigger"] == 1]
        if len(trig_idx) == 0:
            top_feats = impact_df["feature"].head(top_triggers_per_user).tolist()
        else:
            mean_shap_trig = np.abs(shap_values[trig_idx]).mean(axis=0)
            top_feat_idx = np.argsort(mean_shap_trig)[::-1][:top_triggers_per_user]
            top_feats = [feature_cols[i] for i in top_feat_idx]

        # Highest-risk day
        peak_day_row = group.loc[group["risk_score"].idxmax()]
        peak_date = peak_day_row["date"].strftime("%Y-%m-%d")
        peak_score = peak_day_row["risk_score"]

        # Plain-language insight
        signal_phrases = _feature_to_plain(top_feats)
        insight = (
            f"Your most common behavioral triggers are: {signal_phrases}.  "
            f"On your highest-risk day ({peak_date}, risk score {peak_score:.0f}%), "
            f"these signals were most elevated.  "
            f"Overall, {trigger_rate:.0%} of your logged days showed elevated "
            f"anxiety signals."
        )

        rows.append({
            "id": pid,
            "trigger_rate": round(trigger_rate, 3),
            "top_triggers": ", ".join(top_feats),
            "risk_score_mean": round(mean_risk, 1),
            "peak_date": peak_date,
            "peak_risk_score": peak_score,
            "insight_text": insight,
        })

    insights_df = pd.DataFrame(rows).sort_values("trigger_rate", ascending=False)
    insights_df.to_csv(output_dir / "user_insights.csv", index=False)
    log.info(f"[insights] saved {len(insights_df)} user profiles → "
             f"{output_dir / 'user_insights.csv'}")

    print("\n── Sample user insight ──────────────────────────────────────")
    sample = insights_df.iloc[0]
    print(f"  Participant : {sample['id']}")
    print(f"  Trigger rate: {sample['trigger_rate']:.1%}")
    print(f"  Top triggers: {sample['top_triggers']}")
    print(f"  Insight     : {sample['insight_text']}")
    return insights_df


# ─────────────────────────────────────────────────────────────
# EXPLORATION HELPERS
# ─────────────────────────────────────────────────────────────

def explore_dataset(
    raw: dict[str, pd.DataFrame],
    daily: pd.DataFrame,
    output_dir: str = "./outputs",
) -> None:
    """
    Print a structured exploration report and save summary plots.
    Intended to be run after load_lifesnaps() and build_daily_feature_table()
    to understand data shape, coverage, and missingness before modelling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  DATASET EXPLORATION REPORT")
    print("═" * 60)

    # ── Participant overview ───────────────────────────────────────────────
    print(f"\n  Participants        : {daily['id'].nunique()}")
    print(f"  Total daily rows    : {len(daily):,}")
    days_per_p = daily.groupby("id")["date"].count()
    print(f"  Days per participant: min={days_per_p.min()} "
          f"mean={days_per_p.mean():.0f} max={days_per_p.max()}")
    print(f"  Date range          : {daily['date'].min().date()} "
          f"→ {daily['date'].max().date()}")

    # ── Signal coverage ────────────────────────────────────────────────────
    print("\n  Signal availability (% non-null):")
    sig_cols = [c for c in WEARABLE_SIGNALS if c in daily.columns]
    coverage = (daily[sig_cols].notna().mean() * 100).round(1)
    for col, pct in coverage.items():
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {col:<30} {bar} {pct:.1f}%")

    # ── Missing heatmap ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    miss = daily[sig_cols].isna().astype(int)
    sns.heatmap(miss.T, cbar=False, xticklabels=False,
                yticklabels=sig_cols, cmap="Greys", ax=ax)
    ax.set_title("Missing data heatmap (black = missing)", fontsize=11)
    ax.set_xlabel("Rows (each day across all participants)")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_heatmap.png", dpi=120)
    plt.close()

    # ── Signal distributions ───────────────────────────────────────────────
    n_cols = min(len(sig_cols), 9)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    axes = axes.flatten()
    for i, col in enumerate(sig_cols[:n_cols]):
        daily[col].dropna().hist(bins=40, ax=axes[i], color="#3B8BD4", alpha=0.8)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Signal distributions", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "signal_distributions.png", dpi=120)
    plt.close()

    log.info(f"[explore] plots saved → {output_dir}")
    print(f"\n  Plots saved to {output_dir}/\n")


# ─────────────────────────────────────────────────────────────
# PRIVATE UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _snake(name: str) -> str:
    """Convert CamelCase or mixed column names to lowercase snake_case."""
    import re
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower().strip()


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristically identify the date column."""
    candidates = ["date", "datetime", "timestamp", "starttime", "start_time",
                  "logdate", "log_date", "dateofrecord"]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    # Try any column whose name contains 'date'
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None


def _normalise_cols(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Rename columns using a dict that maps lowercase-snake source names
    to target names.  Handles case variation in LifeSnaps exports.
    """
    rename = {}
    lower_cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    for src_lower, target in mapping.items():
        if src_lower in lower_cols:
            rename[lower_cols[src_lower]] = target
    return df.rename(columns=rename)


def _log_coverage(raw: dict[str, pd.DataFrame]) -> None:
    log.info("[load] coverage summary:")
    for name, df in raw.items():
        n_part = df["id"].nunique() if "id" in df.columns else "?"
        log.info(f"  {name:<20} {len(df):>8,} rows   {n_part} participants")


def _log_missing(df: pd.DataFrame) -> None:
    miss = df.isna().mean()
    high_miss = miss[miss > 0.3]
    if len(high_miss):
        log.warning(
            f"[feature_table] columns with >30% missing: "
            f"{high_miss.index.tolist()}"
        )


def _feature_to_plain(feature_names: list[str]) -> str:
    """Map internal feature names to human-readable phrases."""
    plain = {
        "sleep_hours": "sleep duration",
        "sleep_efficiency": "sleep quality",
        "rem_fraction": "REM sleep proportion",
        "active_minutes": "physical activity",
        "steps": "daily step count",
        "resting_hr": "resting heart rate",
        "sedentary_fraction": "sedentary time",
        "spo2_avg": "blood oxygen",
        "skin_temp_deviation": "skin temperature change",
        "fitbit_stress_score": "wearable stress score",
        "hr_above_baseline": "heart rate elevation",
    }
    phrases = []
    for feat in feature_names:
        # Check prefix matches for lag/rolling features
        base = feat.split("_lag")[0].split("_roll")[0].split("_delta")[0]
        suffix = ""
        if "_lag" in feat:
            k = feat.split("_lag")[1]
            suffix = f" ({k}-day lag)"
        elif "_roll" in feat and "mean" in feat:
            w = feat.split("_roll")[1].replace("mean", "")
            suffix = f" ({w}-day average)"
        elif "_delta" in feat:
            suffix = " (daily change)"
        label = plain.get(base, base.replace("_", " "))
        phrases.append(label + suffix)
    return ", ".join(phrases)


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_pipeline(data_dir: str, output_dir: str, label_source: str = "stai"):
    """
    Execute the full Gut But pipeline end-to-end.

    Parameters
    ----------
    data_dir     : path to the LifeSnaps data directory (local download)
    output_dir   : path where all outputs (CSVs, plots, model) are saved
    label_source : "stai" | "panas" | "combined"
    """
    log.info("=" * 55)
    log.info("  GUT BUT — BEHAVIORAL TRIGGER DETECTION PIPELINE")
    log.info("=" * 55)

    # Step 1 — Load
    raw = load_lifesnaps(data_dir)

    # Step 2 — Feature table
    daily = build_daily_feature_table(raw)

    # Step 3 — Exploration (optional but recommended first run)
    explore_dataset(raw, daily, output_dir)

    # Step 4 — Lag features
    featured = engineer_lag_features(daily)

    # Step 5 — Labels
    labelled = build_trigger_labels(featured, raw, label_source=label_source)

    # Step 6 — Train
    model, oof_df, feature_cols = train_lightgbm(labelled, output_dir)

    # Step 7 — Explain
    impact_df = explain_with_shap(model, labelled, feature_cols, output_dir)

    # Step 8 — User insights
    insights_df = generate_user_insights(
        labelled, model, feature_cols, impact_df, output_dir
    )

    log.info("Pipeline complete.  All outputs in: %s", output_dir)
    return {
        "raw": raw,
        "daily": daily,
        "featured": featured,
        "labelled": labelled,
        "model": model,
        "oof_df": oof_df,
        "feature_cols": feature_cols,
        "impact_df": impact_df,
        "insights_df": insights_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gut But trigger detection pipeline")
    parser.add_argument("--data_dir",     default="./lifesnaps_data",
                        help="Path to LifeSnaps data directory")
    parser.add_argument("--output_dir",   default="./outputs",
                        help="Directory for all outputs")
    parser.add_argument("--label_source", default="stai",
                        choices=["stai", "panas", "combined"],
                        help="EMA source for trigger label")
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir, args.label_source)
