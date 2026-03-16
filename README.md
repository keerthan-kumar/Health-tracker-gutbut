# Gut But · Behavioral Health Trigger Detection

> *Detect anxiety triggers before they happen — using your wearable data.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is Gut But?

**Gut But** is a behavioral health intelligence prototype that analyzes your daily wearable signals — sleep, activity, heart rate, and stress — to predict the likelihood of an elevated anxiety episode before it happens.

Instead of asking *"how do you feel?"*, it reads the physiological patterns that silently precede bad health days and surfaces them as an explainable risk score.

> **Core question the app answers:**
> *"Based on how I slept, moved, and felt today — how likely am I to experience an anxiety trigger?"*

---

## Live Demo

🔗 **[Open the app →](https://health-tracker-gutbut-b3o2c7rnnfm8v4njokyz4x.streamlit.app/)**

No installation required. Open in any browser on desktop or mobile.

---

## Screenshots

| Risk Dashboard | SHAP Signal Breakdown |
|---|---|
| Enter today's wearable signals | See which signals drove your score |
| Get a 0–100% trigger risk score | Understand the why, not just the what |

---

## How It Works

The system is built in three layers:

```
Wearable Data (LifeSnaps)
        ↓
Feature Engineering (lag features, rolling averages, deltas)
        ↓
LightGBM Classifier → Trigger Risk Score (0–100%)
        ↓
SHAP Explainer → Which signals drove the prediction
        ↓
Plain-language insight + actionable recommendation
```

### 1. Dataset
The model was trained on the **[LifeSnaps dataset](https://zenodo.org/record/6826682)** — a longitudinal wearable dataset from 71 participants over 4 months, containing Fitbit signals and validated anxiety surveys (STAI, PANAS).

### 2. Trigger Label
A day is labelled as a **trigger day** when the participant's STAI State Anxiety score ≥ 40 — the clinically validated threshold for elevated anxiety (Spielberger, 1983).

### 3. Features Used

| Signal | Description |
|---|---|
| `sleep_hours` | Total hours of sleep |
| `sleep_efficiency` | Time asleep ÷ time in bed (%) |
| `sleep_rem_ratio` | Fraction of sleep in REM stage |
| `steps` | Total daily step count |
| `active_minutes` | Moderately + very active minutes |
| `resting_hr` | Resting heart rate (bpm) |
| `rmssd` | Heart rate variability proxy (ms) |
| `stress_score` | Fitbit-computed stress score (0–100) |
| `spo2` | Blood oxygen saturation (%) |
| `sedentary_fraction` | Sedentary minutes ÷ 1440 |

Each signal also has **lag features** (1-day, 3-day, 7-day), **rolling averages**, and **delta (daily change)** features — giving the model temporal context, not just today's snapshot.

### 4. Model
- **Algorithm:** LightGBM (gradient boosted trees)
- **Validation:** 5-fold GroupShuffleSplit by participant (no data leakage)
- **Explainability:** SHAP TreeExplainer — exact interaction values per prediction
- **Calibration:** Isotonic regression maps raw scores to calibrated probabilities

---

## Project Structure

```
gutbut/
│
├── app.py                  # Streamlit web application (UI + inference)
├── gutbut_pipeline.py      # Full ML pipeline (training, features, SHAP)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── model/
│   ├── lgbm_model.txt      # Trained LightGBM booster
│   ├── feature_cols.json   # Ordered feature names (required for inference)
│   └── calibrator.pkl      # Isotonic calibrator 
│
└── outputs/
    ├── feature_impact.csv  # Global SHAP impact table from training
    ├── user_insights.csv   # Per-participant trigger profiles
    └── oof_predictions.csv # Out-of-fold model predictions
```

---

## Running Locally

### Prerequisites
- Python 3.11
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/gutbut.git
cd gutbut

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

> **Note:** The app runs in demo mode if no model files are present in `model/`.
> It uses a small synthetic LightGBM to demonstrate the full UI.

---

## Training the Model From Scratch

### Step 1 — Get the dataset

Download the LifeSnaps dataset from Kaggle:

```python
# In Google Colab
import os, json
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}, f)

!kaggle datasets download -d skywescar/lifesnaps-fitbit-dataset
!unzip -q lifesnaps-fitbit-dataset.zip -d lifesnaps_data
```

### Step 2 — Run the full pipeline

```python
from gutbut_pipeline import run_pipeline

results = run_pipeline(
    data_dir     = "/content/lifesnaps_data",
    output_dir   = "/content/outputs",
    label_source = "stai"    # or "panas" or "combined"
)
```

### Step 3 — Export model artifacts

```python
import json, pickle, os

os.makedirs("model", exist_ok=True)

# Save model
results["model"].booster_.save_model("model/lgbm_model.txt")

# Save feature list
with open("model/feature_cols.json", "w") as f:
    json.dump(results["feature_cols"], f)

# Save calibrator (if present)
cal = getattr(results["model"], "_calibrator", None)
if cal:
    with open("model/calibrator.pkl", "wb") as f:
        pickle.dump(cal, f)
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

---

## Using the App

### Input Signals (Sidebar)
Use the sliders to enter today's wearable readings across three categories:

- **Sleep** — duration, efficiency, REM ratio
- **Activity** — steps, active minutes, sedentary fraction
- **Physiology** — resting HR, HRV (RMSSD), SpO2, stress score

### Output (Main Panel)

| Output | Description |
|---|---|
| **Risk gauge** | 0–100% trigger probability |
| **Risk card** | Color-coded level (low / moderate / high) |
| **Plain insight** | 2–3 sentence explanation of your score |
| **SHAP bar chart** | Which signals raised or lowered your risk |
| **Signal radar** | Your signals vs healthy targets |
| **Top drivers table** | Top 5 features with SHAP values |
| **Metric row** | Quick summary of key input values |

### Try These Scenarios

| Scenario | Sleep | Stress | HR | Steps | Expected |
|---|---|---|---|---|---|
| Healthy day | 8h | 20 | 58 | 10,000 | Low risk (~15%) |
| Stressed day | 5h | 80 | 85 | 3,000 | High risk (~80%) |
| Mixed signals | 6.5h | 55 | 68 | 7,000 | Moderate (~45%) |

---

## Pipeline Architecture

```
load_lifesnaps()
      ↓
build_daily_feature_table()     # merge sleep, activity, HR, SpO2, stress
      ↓
engineer_lag_features()          # lag1, lag3, lag7, roll3, roll7, delta1
      ↓
build_trigger_labels()           # STAI ≥ 40 → trigger = 1
      ↓
train_lightgbm()                 # 5-fold GroupShuffleSplit, early stopping
      ↓
explain_with_shap()              # TreeExplainer, interaction matrix
      ↓
generate_user_insights()         # per-user plain-language profiles
```

Each module is independent and returns a clean DataFrame — you can run any step in isolation.

---

## Key Design Decisions

### Why LightGBM over deep learning?
The LifeSnaps dataset has ~7,000 labelled daily rows across 71 participants. Deep learning (LSTMs, Transformers) requires orders of magnitude more data. LightGBM consistently outperforms neural approaches on tabular data of this size and is orders of magnitude faster to train and explain.

### Why GroupShuffleSplit validation?
Standard k-fold would split individual participants across train and test, allowing the model to memorize personal patterns. GroupShuffleSplit ensures the validation set contains entirely unseen participants — measuring true generalization.

### Why SHAP over feature importance?
Standard feature importance tells you which features matter globally. SHAP tells you *why this specific prediction was made for this specific person today* — which is the clinically meaningful insight.

### Why lag features?
A single day's sleep reading predicts little. Three consecutive nights of poor sleep is a strong predictor. Lag and rolling features give the model temporal memory without requiring a sequence model.

---

## Roadmap

- [ ] Automatic Fitbit API data pull (no manual slider entry)
- [ ] Per-user history tracking with trend visualization
- [ ] Weekly email digest with trigger pattern summary
- [ ] Mobile-optimized layout
- [ ] Multi-label output (anxiety / fatigue / mood separately)
- [ ] HMM-based latent state modeling (phase 2)
- [ ] Personalized model fine-tuning per user

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.32.0 | Web application framework |
| `lightgbm` | 4.1.0 | Gradient boosted tree model |
| `shap` | 0.44.0 | Model explainability |
| `pandas` | 2.1.0 | Data manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `plotly` | 5.18.0 | Interactive charts |
| `matplotlib` | 3.8.0 | Static plots |
| `scikit-learn` | 1.4.0 | Validation, calibration |

---

## Dataset Citation

```
Nasim Sarkaleh, et al. (2022).
LifeSnaps, a 4-month multi-modal dataset capturing unobtrusive snapshots of our lives in the wild.
Zenodo. https://doi.org/10.5281/zenodo.6826682
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Acknowledgements

- **LifeSnaps** dataset — Datalab AUTH research group
- **SHAP** library — Lundberg & Lee (2017)
- **LightGBM** — Microsoft Research
- Inspired by the **Gut But** product concept for behavioral health intelligence

---

*Built as a prototype for behavioral health trigger detection research.*
*Not a medical device. Not a substitute for professional mental health care.*
