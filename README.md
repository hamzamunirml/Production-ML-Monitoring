# 🏦 Production ML Monitoring — Bank Marketing Prediction System

An end-to-end machine learning system that predicts whether a bank client will subscribe to a **term deposit**, built on the UCI Bank Marketing dataset. The project covers the full lifecycle — EDA, feature engineering, model training and evaluation — and ships the trained model through **three parallel serving layers**: a FastAPI service, a Streamlit app, and a Flask wrapper for lightweight hosting.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/status-active--development-yellow)

---

## 📌 Overview

Banks run outbound telemarketing campaigns to sell term deposits, but calling every client wastes time and annoys most of them. This project trains a classifier on a real campaign dataset (45,211 contacts) to estimate **the probability that a given client will say yes**, so marketing effort can be prioritized toward the clients most likely to convert.

The repository contains:

- A single **Jupyter notebook** that performs the entire analysis — from raw CSV to a saved, deployable model.
- A reusable **`src/` package** mirroring the same pipeline as importable, testable modules (preprocessing, feature engineering, training, evaluation, prediction).
- A **FastAPI** service (`api/`) exposing the model over REST, with request validation, batch scoring, and CSV upload.
- A **Streamlit** app for interactive, form-based predictions.
- A **Flask** wrapper for environments that don't support ASGI servers (e.g. PythonAnywhere).
- A **test suite** (`tests/`) covering config, preprocessing, feature engineering, training, prediction, and a full integration test.

---

## 🎯 Results at a glance

Two models were trained and compared on a stratified 80/20 split:

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC&ndash;AUC |
|---|---|---|---|---|---|
| Decision Tree | 90.0% | 0.61 | 0.40 | 0.48 | 0.868 |
| **Random Forest** ✅ *(deployed)* | 89.8% | 0.75 | 0.19 | 0.30 | **0.918** |

**Random Forest** was selected for deployment on the strength of its ROC&ndash;AUC (0.918 vs 0.868). It's worth being explicit about the trade-off this implies: when the model predicts "Yes" it is correct 75% of the time, but it only catches 19% of the clients who actually subscribe. It's a high-precision, low-recall model — well suited to a "call the safest bets first" strategy, less suited to "don't miss a single lead." If recall matters more for your use case, the Decision Tree or a re-tuned/threshold-adjusted Random Forest is worth revisiting.

**Dataset class balance:** 88.3% "No" vs 11.7% "Yes" (≈7.5 : 1 imbalance) — handled via stratified sampling; the `src/` pipeline additionally supports SMOTE oversampling.

**Top predictive features:** `duration`, `pdays`, `previous`, job type, and contact month.

> ⚠️ **Leakage caveat:** `duration` (the length of the last call) is the single strongest predictor, but it's only known *after* the call has already happened — it can't be used to decide who to call *before* dialing. It's kept here for benchmarking against the original UCI study, but a model meant to drive real outbound targeting should be retrained without it.

---

## 🗂️ Dataset

[UCI Bank Marketing Data Set](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-full.csv`) — 45,211 client records from a Portuguese bank's telemarketing campaigns.

| Column | Description |
|---|---|
| `age` | Client's age |
| `job` | Job category (admin, technician, blue-collar, retired, student, …) |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Has credit in default? |
| `balance` | Average yearly account balance (€) |
| `housing` | Has a housing loan? |
| `loan` | Has a personal loan? |
| `contact` | Contact communication type |
| `day`, `month` | Date of last contact in this campaign |
| `duration` | Last contact duration, in seconds *(see leakage caveat above)* |
| `campaign` | Number of contacts in this campaign |
| `pdays` | Days since last contact from a previous campaign (`-1` = never contacted) |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of the previous campaign |
| `y` | **Target** — did the client subscribe to a term deposit? |

`reference_data.csv` is a 500-row baseline sample shipped alongside the full dataset — intended as the comparison point for a future drift-monitoring routine (see [Roadmap](#-roadmap)).

---

## 🧠 Pipeline

```
Raw CSV  →  Age binning  →  Train/test split (stratified)  →  Preprocessing  →  Model training  →  Evaluation  →  Serialized model
```

**Feature engineering**
- `age_group` is derived from `age`: `Young` (≤30), `Middle-Aged` (31–50), `Senior` (51+).
- IQR-based outlier detection is run on `age`, `balance`, and `duration` for diagnostic purposes.

**Preprocessing**
- Numeric features (`age`, `balance`, `duration`, `campaign`, `pdays`, `previous`, `day`) are standardized with `StandardScaler`.
- Categorical features (`job`, `marital`, `education`, `contact`, `month`, `poutcome`, `age_group`, plus the binary fields) are one-hot encoded.
- The deployed `saved_models/preprocessor.pkl` is a `ColumnTransformer` fitted inside the notebook; the `src/data_preprocessing.py` module implements an equivalent pipeline as a reusable, testable component (see [Notes](#-notes--known-gaps) for a small divergence between the two).

**Models**
- `DecisionTreeClassifier(max_depth=10, min_samples_split=20)`
- `RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, n_jobs=-1)` — the deployed model.
- `src/model_training.py` additionally supports `SMOTE` oversampling for teams that want to trade precision for recall on the minority class.

---

## 📁 Project structure

```
Production-ML-Monitoring/
├── api/                            # FastAPI service
│   ├── main.py                     # App instance, routes, startup hooks
│   ├── models.py                   # Pydantic request/response schemas
│   └── dependencies.py             # Model loading, validation, confidence scoring
├── src/                            # Reusable training pipeline package
│   ├── config.py                   # Central configuration (paths, params, splits)
│   ├── data_preprocessing.py       # DataPreprocessor + BinaryEncoder
│   ├── feature_engineering.py      # Age binning, outlier detection
│   ├── model_training.py           # SMOTE, Decision Tree, Random Forest
│   ├── model_evaluation.py         # Metrics + model comparison
│   ├── prediction.py               # Single-customer inference helper
│   └── utils.py                    # File / JSON / CSV utilities
├── tests/                          # Unit + integration tests
├── saved_models/                   # Trained artifacts (tracked via Git LFS)
│   ├── random_forest_model.pkl
│   └── preprocessor.pkl
├── Production-ML-Monitoring.ipynb  # Full EDA → training → evaluation notebook
├── bank-full.csv                   # UCI Bank Marketing dataset (45,211 rows)
├── reference_data.csv              # 500-row baseline sample for drift comparisons
├── streamlit_app.py                # Interactive prediction UI
├── flask_app.py                    # Lightweight Flask wrapper for static hosting
├── test_api.py                     # Manual smoke-test script for the FastAPI service
├── run_tests.py                    # Unittest discovery/runner
└── requirements.txt
```

---

## ⚙️ Setup

```bash
git clone https://github.com/hamzamunirml/Production-ML-Monitoring.git
cd Production-ML-Monitoring

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> 📌 `requirements.txt` covers the data-science, Streamlit, and Flask stack. The FastAPI service and the SMOTE-based training pipeline pull in a few packages that aren't listed there yet — install them separately for now:
> ```bash
> pip install fastapi uvicorn pydantic python-multipart imbalanced-learn
> ```

---

## ▶️ Usage

**Streamlit app** (interactive form, runs the model directly):
```bash
streamlit run streamlit_app.py
```

**FastAPI service**:
```bash
uvicorn api.main:app --reload
# Interactive docs → http://localhost:8000/docs
```

**Flask wrapper** (e.g. for PythonAnywhere):
```bash
python flask_app.py
```

**Smoke-test the running FastAPI service**:
```bash
python test_api.py
```

**Run the unit/integration test suite**:
```bash
python run_tests.py
# or target one module:
python run_tests.py --test test_feature_engineering
```

---

## 🔌 API reference (FastAPI)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — status, version, whether the model is loaded |
| `POST` | `/predict` | Predict for a single customer |
| `POST` | `/predict/batch` | Predict for a list of customers, with summary stats |
| `POST` | `/predict/csv` | Upload a CSV of customers, get predictions for every row |
| `GET` | `/model/info` | Model type, hyperparameters, top feature importances |
| `GET` | `/docs` / `/redoc` | Auto-generated interactive API documentation |

**Example — single prediction**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "balance": 1000,
    "housing": "yes",
    "loan": "no",
    "duration": 200
  }'
```

```json
{
  "prediction": "Yes",
  "probability_yes": 0.65,
  "probability_no": 0.35,
  "confidence_level": "Medium"
}
```

Only `age`, `job`, `marital`, `education`, `balance`, `housing`, `loan`, and `duration` are required — every other field (`contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `poutcome`, `default`) has a sensible default and is optional.

---

## 🧪 Testing

| File | Covers |
|---|---|
| `tests/test_config.py` | Config attributes and value ranges |
| `tests/test_data_preprocessing.py` | `BinaryEncoder`, `DataPreprocessor` |
| `tests/test_feature_engineering.py` | Age binning, outlier detection |
| `tests/test_model_training.py` | SMOTE balancing, Decision Tree / Random Forest training |
| `tests/test_prediction.py` | `Predictor` inference on single customers and DataFrames |
| `tests/test_integration.py` | Full pipeline, end-to-end, on a synthetic dataset |

---

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn (Decision Tree, Random Forest), imbalanced-learn (SMOTE) |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn, Plotly |
| Serving | FastAPI + Uvicorn, Streamlit, Flask |
| Validation | Pydantic |
| Testing | unittest |
| Persistence | joblib, Git LFS |

---

## 📝 Notes & known gaps

- The artifacts in `saved_models/` were produced directly inside the notebook. The parallel `src/` pipeline (`DataPreprocessor`, `ModelTrainer` with SMOTE, a 70/30 split) is a clean, testable reimplementation but isn't yet what generates the shipped model — the two should be reconciled into one source of truth.
- `src/prediction.py` and `src/data_preprocessing.py` reference a `saved_encoders/` folder that isn't part of this repo; the live services (`api/`, `streamlit_app.py`, `flask_app.py`) correctly load from `saved_models/` instead.
- `reference_data.csv` ships as a baseline sample but isn't wired into an actual drift check yet.

## 🚧 Roadmap

- [ ] Wire `reference_data.csv` into a real drift-detection job (e.g. PSI or KS-test against live traffic)
- [ ] Unify the notebook and `src/` pipelines so one script produces the deployed artifacts
- [ ] Complete `requirements.txt` (FastAPI, Uvicorn, Pydantic, imbalanced-learn)
- [ ] Add CI (GitHub Actions) to run `run_tests.py` on every push
- [ ] Containerize the FastAPI service for deployment parity
- [ ] Re-evaluate the model without `duration` for a leakage-free, pre-call scoring version

---

## 👤 Author

**Hamza Munir** — AI / ML Engineering
KFUEIT, Rahim Yar Khan
[github.com/hamzamunirml](https://github.com/hamzamunirml)

---

## 📄 License

No license file is currently included in this repository. All rights reserved by default until one is added — consider adding an [MIT](https://choosealicense.com/licenses/mit/) license if you'd like others to freely use or build on this work.
