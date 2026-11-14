# CoralGuard: Multimodal Early Warning of Coral Reef Bleaching

> **Predicting coral bleaching risk 2 to 8 weeks ahead per reef site by fusing underwater acoustic soundscapes, satellite SST, and citizen diver reports.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 🌊 Overview

Coral bleaching occurs when corals become thermally stressed and expel their photosynthetic endosymbionts (*Symbiodiniaceae*). While satellite observations of Sea Surface Temperature (SST) and Degree Heating Weeks (DHW) provide broad thermal stress indices, they measure only physical conditions at the sea surface and lack real-time biological ground truth.

**CoralGuard** is a tri-modal early warning system that forecasts bleaching risk **14 to 28 days ahead** per $0.25^\circ$ reef site grid by fusing:
1. **Satellite SST & Thermal Stress**: Daily SST, SST Anomaly (SSTA), Degree Heating Weeks (DHW), and rolling thermal dynamics from NOAA Coral Reef Watch.
2. **Citizen Diver Reports**: Observations from Reef Check, CoralWatch, and Global Coral Bleaching databases, weighted by observer experience and confidence.
3. **Underwater Acoustic Soundscapes**: Reef biophony metrics (snapping shrimp broadband energy at 2–20 kHz, fish chorus pulses at 50–1000 Hz, Acoustic Complexity Index, Acoustic Diversity Index) and learned CNN autoencoder embeddings.

---

## 📁 Repository Structure

```text
├── backend/                 # Production FastAPI backend (PostgreSQL/SQLite, Redis, APScheduler)
│   ├── api/                 # FastAPI REST API routes (/sites, /risk, /sst, /acoustics, /alerts, /reports)
│   ├── ingestion/           # Scheduled NOAA ERDDAP fetch, hydrophone acoustics, citizen validation
│   ├── features/            # Weekly multi-modal feature pipeline
│   ├── models/              # XGBoost production inference engine
│   └── alembic/             # Database schema migrations
├── powerbi/                 # Power BI Enterprise Assets (Star Schema, DAX, Theme, PQ Queries)
│   ├── data/                # Exported clean star-schema CSVs (0 nulls, 100% referential integrity)
│   ├── queries/             # Power Query M transformation scripts (*.pq)
│   ├── measures.dax         # Production DAX measures library
│   ├── coral_theme.json     # Custom Power BI visual theme
│   ├── MODEL.md             # Star schema ERD & relationship matrix
│   └── BUILD_STEPS.md       # 30-minute click-by-click assembly guide
├── data/
│   ├── raw/                 # Downloaded / generated raw datasets & audio clips (.wav)
│   └── processed/           # Standardized Parquet datasets & PyTorch tensors
├── src/
│   ├── ingest/              # Ingestion scripts for SST, Citizen science, and Acoustics
│   ├── features/            # Feature engineering and temporal-spatial alignment
│   ├── models/              # Baselines, Boosted Trees, Neural Net, Tuning, Evaluation
│   ├── synthetic.py         # Physics-informed fallback synthetic data generator
│   ├── utils.py             # Logging, YAML config loader, random seed locking
│   └── run_all.py           # End-to-end master execution pipeline
├── app/
│   ├── streamlit_app.py     # Interactive Streamlit 6-tab dashboard
│   └── api_client.py        # Frontend HTTP client connecting to FastAPI backend
├── tests/                   # Backend unit tests (features, trust scoring, API endpoints)
├── reports/                 # Evaluation figures, ablation tables, JSON metrics
├── config.yaml              # Centralized configuration (paths, dates, hyperparams)
├── .env.example             # Environment configuration template
├── requirements.txt         # Pinned Python package dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/coralguard.git
cd coralguard

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
```

### 2. Run the End-to-End Pipeline
To run data ingestion, feature extraction, model training, Optuna tuning, and evaluation in one command:
```bash
python -m src.run_all
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501` to view:
- **Global Risk Map (Folium)**: Color-coded Watch/Warning/Alert status per reef site.
- **SST & Risk Trajectory (Plotly)**: Dual-axis time series of thermal stress and forecasted bleaching probability.
- **Acoustic Health Index**: Biophony energy ratios, audio player, and interactive WAV uploader with real-time bioacoustic health verdicts.
- **Community Diver Ingestion**: Drag-and-drop CSV upload for local diver observations.
- **Model Explainability**: SHAP beeswarm feature importance attribution.
- **What-If Heatwave Simulator**: Interactive SST & acoustic stress slider with real-time risk recalculation.

### 4. Command-Line Interface (CLI) Queries
For automated cron jobs, terminal queries, or operational site assessments:
```bash
# Query active bleaching warnings/alerts (+14 days ahead)
python -m src.cli alerts --lead 14

# Predict detailed multimodal risk for a specific reef site
python -m src.cli predict --lead 14 --site great_barrier_reef_-24.00_142.00

# Analyze underwater audio soundscape file for bioacoustic health
python -m src.cli audio --file data/raw/audio/clip_00001_healthy.wav

# Export self-contained HTML executive report (with embedded figures)
python -m src.cli export

# Execute automated monitoring scan across all regional reef sites
python -m src.cli monitor --single-shot

# Re-evaluate all models and display False Alarm Rate (FAR) benchmarks
python -m src.cli evaluate
```

### 5. Programmatic REST API Microservice
For webhooks, hydrophone sensor streams, and external GIS platforms:
```bash
python -m src.server --port 8000
```
Standard endpoints provided:
- `GET /health` — Service health check & monitored regional boundaries.
- `GET /sites` — List of all monitored reef sites and GPS coordinates.
- `GET /alerts?lead=14` — Filtered list of active Watch/Warning/Alert sites.
- `GET /predict?site=<site_id>&lead=14` — Site-specific forecasted probabilities and alert tier.
- `GET /metrics` — Latest ablation benchmark table and FAR reduction metrics.
- `POST /predict` — JSON payload inference with custom SST, diver, or acoustic feature inputs.

### 6. Automated Test Suite
To verify feature calculations, absence of temporal leakage, REST endpoints, and model inference:
```bash
python -m unittest discover tests
```

### 7. Docker Deployment
Deploy the full system with Docker:
```bash
# Build and run the dashboard container
docker compose up -d coralguard-dashboard

# Or run the offline training pipeline via Docker
docker compose run --rm coralguard-pipeline
```

---

## 📡 Data Sources & Fallback Disclosure

### Primary Data Sources:
1. **Satellite SST**: NOAA Coral Reef Watch Daily 5km v3.1 via CoastWatch ERDDAP.
2. **Citizen Diver Science**: Global Coral Bleaching Database (van Woesik & Kratochwill, 2022) and Reef Check surveys.
3. **Underwater Acoustics**: Marine soundscape hydrophone recordings (healthy vs degraded biophony) from Zenodo and Dryad repositories.

### ⚠️ Synthetic Fallback Disclosure:
In accordance with project execution guidelines, if external API connections or remote hydrophone endpoints are unavailable (e.g. rate limits, network boundaries, missing credentials), the system automatically falls back to the documented generator in `src/synthetic.py`.
- **SST Simulation**: Implements sinusoidal hemispheric seasonality, El Niño anomaly shocks (2016, 2020, 2024), and 12-week accumulating Degree Heating Weeks ($DHW = \sum \max(0, SST - MMM \ge 1.0) / 7$).
- **Citizen Science Simulation**: Spatially clustered diver surveys with log-normal depth distribution and observer confidence weighting.
- **Acoustic Simulation**: Synthesizes snapping shrimp impulse trains (2–20 kHz) and low-frequency fish chorus harmonics (50–1000 Hz) using Poisson process arrival times.

---

## 🧪 Modeling & Technical Methodology

### 1. Leakage-Free Time-Based Splitting
To prevent future-information leakage and spatial autocorrelation inflation:
- **Training Set (<= 2021)**: 9,855 weekly observations.
- **Validation Set (2022)**: 1,404 weekly observations (used for early stopping & Optuna tuning).
- **Test Set (2023–2024)**: 2,835 weekly observations (strict holdout).

### 2. Masked-Modality Acoustic Alignment
Because hydrophone recordings rarely coincide continuously with satellite passes:
- When a local recording exists within $\pm 21$ days, extracted features are used directly (`acoustic_is_missing = 0`).
- When missing, features are imputed with regional running baselines and flagged with a binary indicator (`acoustic_is_missing = 1`).

### 3. Model Architectures
- **Baselines**: NOAA DHW Threshold Rule ($DHW \ge 8 \rightarrow \text{Severe}, 4 \le DHW < 8 \rightarrow \text{Mild}$), Logistic Regression (SST only), XGBoost (SST only).
- **Ablations**: Evaluated on SST only, SST + Citizen, SST + Acoustic, and full Tri-modal.
- **Optuna Tuning**: 30 trials optimizing Macro F1 with tree depth, subsampling, and learning rate.
- **PyTorch Late-Fusion Neural Network**: Separate encoders for SST (MLP), Citizen (MLP), and Acoustics (CNN/MLP) joined by a late-fusion head with Focal Loss ($\gamma = 2.0$) to counter severe class imbalance.

---

## 📊 Benchmark Results

| Model | Lead Time | Macro F1 | Severe Recall | PR-AUC (Severe) | ROC-AUC (Severe) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| NOAA DHW Baseline | t+14d | 0.6050 | 0.8749 | 0.9685 | 0.9828 | 0.2087 |
| Logistic Regression (SST) | t+14d | 0.8426 | 0.9553 | 0.9942 | 0.9962 | 0.0857 |
| XGBoost (SST only) | t+14d | 0.8577 | 0.9587 | 0.9950 | 0.9965 | 0.0673 |
| XGBoost (SST + Citizen) | t+14d | 0.8598 | 0.9598 | 0.9965 | 0.9978 | 0.0651 |
| XGBoost (SST + Acoustic) | t+14d | **0.8809** | **0.9743** | 0.9962 | 0.9976 | 0.0561 |
| **XGBoost (Multimodal: All 3)** | **t+14d** | 0.8753 | **0.9743** | **0.9972** | **0.9983** | **0.0559** |
| LightGBM (Multimodal) | t+14d | 0.8793 | 0.9732 | 0.9970 | 0.9981 | **0.0543** |
| NOAA DHW Baseline | t+28d | 0.5452 | 0.7559 | 0.8780 | 0.9303 | 0.3581 |
| LightGBM (Multimodal) | t+28d | **0.8543** | **0.9216** | **0.9928** | **0.9964** | **0.0769** |

---

## ⚖️ Ethical & Environmental Considerations

1. **Equitable Action vs Tourism Interests:** Bleaching alert declarations can impact local dive tourism revenue. Predictions must be transparently reported with confidence bounds.
2. **Indigenous & Community Rights:** Automated early warnings should inform non-punitive ecological interventions (e.g., nursery shading, temporary boat speed reductions) and involve traditional owners.
3. **Data Bias:** Volunteer diver surveys cluster around popular tourist destinations; autonomous hydrophone buoys and remote sensing balance spatial sampling equity.

