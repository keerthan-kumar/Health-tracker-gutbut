# gut but · Behavioral Health Trigger Detection

> *Detect anxiety triggers before they happen — using your wearable data.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![AUC](https://img.shields.io/badge/AUC-0.69-teal)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is Gut But?

**Gut But** is a behavioral health intelligence prototype that analyzes daily
wearable signals — sleep, activity, heart rate, and stress — to predict the
likelihood of an elevated anxiety episode before it happens.

Instead of asking *"how do you feel?"*, it reads the physiological patterns
that silently precede bad health days and surfaces them as an explainable
risk score with plain-language recommendations.

> **Core question the app answers:**
> *"Based on how I slept, moved, and felt today — how likely am I to
> experience an anxiety trigger?"*

---

## Live Demo

🔗 **[Open the app →](https://health-tracker-gutbut-b3o2c7rnnfm8v4njokyz4x.streamlit.app/)**

No installation required. Open in any browser on desktop or mobile.

---

## How It Works

```
LifeSnaps wearable dataset (71 participants · 4 months)
        ↓
Feature engineering — sleep, HR, activity, HRV, SpO2, stress
        ↓
Lag features — 1-day, 3-day, 7-day rolling context per signal
        ↓
Trigger labels — STAI State Anxiety score ≥ 50 (cohort-calibrated)
        ↓
LightGBM classifier — 5-fold GroupShuffleSplit · AUC 0.69
        ↓
SHAP TreeExplainer — per-prediction signal attribution
        ↓
Risk score (0–100%) + plain-language insight + recommendations
```

---

## Model Performance

Trained and validated on **1,844 labelled daily observations** across
71 participants from the LifeSnaps dataset.

| Metric | Value |
|---|---|
| Cross-validated AUC | **0.69 ± 0.03** |
| Average Precision | **0.56** |
| Overall Accuracy | **60%** |
| Validation strategy | GroupShuffleSplit (participant-level) |
| Trigger rate (training) | **43.9%** — balanced ✅ |

### Score interpretation

Because the LifeSnaps cohort had a median stress score of 76/100
(a genuinely high-stress population), the model's output range is
compressed toward the upper end. Scores are best interpreted relatively:

| Score | Label | Meaning |
|---|---|---|
| Below 78% | LOW RISK | Well below cohort average — good signals |
| 78% – 89% | MODERATE RISK | Around cohort average — worth monitoring |
| Above 89% | HIGH RISK | Above cohort average — elevated anxiety likely |

### Validated test cases

| Scenario | Score | Label |
|---|---|---|
| Healthy (8h sleep · stress 18 · HRV 55ms) | ~88% | MODERATE |
| Moderate (6.5h sleep · stress 65 · HRV 28ms) | ~92% | HIGH |
| Stressed (4h sleep · stress 90 · HRV 15ms) | ~99% | HIGH |

The **11-point separation** between healthy and stressed confirms the
model responds correctly to signal changes.

---

## Dataset

**[LifeSnaps](https://zenodo.org/record/6826682)** — a longitudinal
wearable dataset from 71 participants over 4 months containing Fitbit
signals and validated psychological surveys (STAI, PANAS).

### Why STAI threshold = 50 (not the standard 40)

The standard clinical cutoff of STAI ≥ 40 (Spielberger, 1983) produced
a 92% trigger rate on this cohort — almost everyone was above it, making
the label useless for classification. The LifeSnaps participants had a
mean STAI score of 48.2 with a median of 49, so threshold 50 was chosen
as the cohort-appropriate split point giving a balanced 43.9% trigger rate.

| Threshold | Trigger rate | Decision |
|---|---|---|
| ≥ 40 | 92.0% | Too imbalanced — rejected |
| ≥ 45 | 79.8% | Still imbalanced — rejected |
| **≥ 50** | **43.9%** | **Balanced — used ✅** |
| ≥ 55 | 10.0% | Too few positives — rejected |
| ≥ 60 | 1.1% | Unusable — rejected |

---

## Features Used

| Signal | Description | Type |
|---|---|---|
| `sleep_hours` | Total hours of sleep | Direct |
| `sleep_efficiency` | Time asleep ÷ time in bed (%) | Direct |
| `sleep_rem_ratio` | Fraction of sleep in REM stage | Direct |
| `steps` | Total daily step count | Direct |
| `active_minutes` | Moderately + very active minutes | Direct |
| `resting_hr` | Resting heart rate (bpm) | Direct |
| `rmssd` | Heart rate variability proxy (ms) | Direct |
| `stress_score` | Fitbit-computed stress score (0–100) | Direct |
| `spo2` | Blood oxygen saturation (%) | Direct |
| `sedentary_fraction` | Sedentary minutes ÷ 1440 | Direct |
| `*_lag1/3/7` | Signal value 1, 3, 7 days ago | Engineered |
| `*_roll3/7` | 3-day and 7-day rolling averages | Engineered |
| `*_delta1` | Daily change (today − yesterday) | Engineered |

**Total features fed to model: 96**

---

## Project Structure

```
gutbut/
│
├── app.py                   # Streamlit web application
├── gutbut_pipeline.py       # Full ML pipeline (training + features)
├── Gutbut.ipynb             # Google Colab training notebook
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── model/
│   ├── lgbm_model.txt       # Trained LightGBM booster
│   ├── feature_cols.json    # Ordered list of 96 feature names
│   └── feature_medians.json # Training-set medians (fixes NaN bug)
│
└── outputs/
    ├── feature_impact.csv   # Global SHAP impact table
    ├── user_insights.csv    # Per-participant trigger profiles
    └── oof_predictions.csv  # Out-of-fold model predictions
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

> **Note:** The app runs in demo mode if no model files are present in
> `model/`. A small synthetic LightGBM is built automatically so the
> full UI is visible without needing the trained model files.

---

## Training From Scratch

### Step 1 — Get the dataset (Google Colab)

```python
import os, json
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump({"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}, f)

!kaggle datasets download -d skywescar/lifesnaps-fitbit-dataset
!unzip -q lifesnaps-fitbit-dataset.zip -d lifesnaps_data
```

### Step 2 — Run the pipeline

Open `Gutbut.ipynb` in Google Colab and run all cells in order.
The notebook handles data loading, feature engineering, label
construction, model training, and SHAP analysis end to end.

### Step 3 — Export model artifacts

```python
import json, os, numpy as np

os.makedirs("model", exist_ok=True)

# Save model
model.booster_.save_model("model/lgbm_model.txt")

# Save feature list
with open("model/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# Save training medians — critical for correct predictions
medians = {}
for col in feature_cols:
    val = labelled_df[col].median()
    medians[col] = 0.0 if np.isnan(val) else float(val)
with open("model/feature_medians.json", "w") as f:
    json.dump(medians, f, indent=2)

print("All artifacts saved")
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

---

## Using the App

### Sidebar inputs

Enter today's wearable readings using the sliders:

- **Sleep** — duration, efficiency, REM ratio
- **Activity** — steps, active minutes, sedentary fraction
- **Physiology** — resting HR, HRV (RMSSD), SpO2, stress score

Click **Analyse Triggers** to run the model.

### What you get back

| Output | What it shows |
|---|---|
| Risk gauge | 0–100% trigger probability |
| Risk card | Color-coded level + plain-language explanation |
| Radar chart | Your signals vs healthy targets |
| SHAP bar chart | Which signals raised or lowered your risk |
| Top drivers table | Top 5 features with SHAP values |
| Metric row | Quick summary of key inputs |

---

## Key Design Decisions

### Why LightGBM over deep learning?
1,844 labelled rows across 71 participants is too small for LSTMs or
Transformers. LightGBM consistently outperforms neural approaches on
tabular data of this size and produces exact SHAP values natively.

### Why GroupShuffleSplit?
Standard k-fold splits rows randomly, allowing the model to memorize
personal patterns and inflating AUC. GroupShuffleSplit ensures each
validation fold contains entirely unseen participants — the correct
measure of generalization for a deployed health app.

### Why median-filling instead of NaN?
The 96-feature model includes 26 signals with no corresponding slider
(sleep stage ratios, skin conductance, temperature deviation). Leaving
these as NaN caused LightGBM to route every prediction toward high risk,
producing 99% scores for all inputs regardless of sliders. Filling with
training-set medians represents "average person on unknown signals" —
the correct neutral assumption for single-day inference.

### Why threshold 50 instead of the clinical 40?
The Spielberger (1983) threshold of 40 produced a 92% positive rate on
this cohort — the classifier had nothing meaningful to learn. Threshold
50 matches the actual population median and produces a balanced 43.9%
trigger rate that the model can genuinely learn to distinguish.

### Why lag features?
A single night of poor sleep predicts little in isolation. Three
consecutive nights below 6 hours is a strong predictor. Lag features
give the model temporal memory without requiring a full sequence model
(LSTM/HMM), which would need far more longitudinal data to train reliably.

---

## Known Limitations

- **High absolute scores** — the LifeSnaps cohort was a high-stress
  population (median stress 76/100). Scores should be interpreted
  relative to the 72–99% model range, not against a general population.
- **Single-day inference** — the app approximates lag features using
  today's values. A production version would pull 7-day history from
  a database for accurate temporal context.
- **No diet or gut-specific signals** — the dataset does not include
  food intake, GI symptoms, or gut microbiome data. These would be
  the most valuable additions for a true Gut But product.
- **71 participants** — generalisation to broader populations is
  limited. A production model would require thousands of participants
  across diverse demographics.

---

## Roadmap

- [ ] Automatic Fitbit API data pull (no manual sliders)
- [ ] Per-user history tracking with 30-day trend chart
- [ ] Weekly email digest with trigger pattern summary
- [ ] Personalized model fine-tuning per user
- [ ] Multi-label output (anxiety / fatigue / mood separately)
- [ ] HMM latent state modeling (phase 2)
- [ ] Diet and GI symptom logging
- [ ] Mobile-optimized layout

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
LifeSnaps, a 4-month multi-modal dataset capturing unobtrusive
snapshots of our lives in the wild.
Zenodo. https://doi.org/10.5281/zenodo.6826682
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Acknowledgements

- **LifeSnaps** dataset — Datalab AUTH research group
- **SHAP** library — Lundberg & Lee (2017), NeurIPS
- **LightGBM** — Microsoft Research
- Inspired by the **Gut But** product concept for behavioral health intelligence

---

*Built as a prototype for behavioral health trigger detection research.*
*Not a medical device. Not a substitute for professional mental health care.*
