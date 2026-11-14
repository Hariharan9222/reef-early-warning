# CoralGuard Production Backend

Production-grade, multimodal backend for **Multimodal Early Warning of Coral Reef Bleaching using Underwater Acoustic Soundscapes, Satellite SST, and Citizen Diver Reports**.

---

## 🏗️ Architecture & Technology Stack

| Layer | Component | Description |
|---|---|---|
| **API** | **FastAPI** | High-performance asynchronous REST API with Pydantic v2 schemas and OpenAPI interactive docs (`/docs`). |
| **Database** | **PostgreSQL + PostGIS** / **SQLite** | Spatial database storing reef sites, daily satellite SST observations, hydrophone recordings, and citizen diver reports. Seamless local SQLite fallback (`data/coralguard.db`). |
| **Migrations** | **Alembic** | Automated database schema version control and migrations (`backend/alembic/`). |
| **Caching** | **Redis** / **In-Memory TTL** | Dual-mode caching for frequent geospatial lookups and catalog queries with automatic in-memory TTL fallback. |
| **Scheduler** | **APScheduler** | Automated background scheduler for daily NOAA ERDDAP SST ingestion and feature aggregation. |
| **ML Inference** | **XGBoost & scikit-learn** | Multimodal early warning model predicting bleaching risk at $t+14\text{d}$ and $t+28\text{d}$ horizons. |
| **Bioacoustics** | **librosa & soundfile** | Extracts Acoustic Complexity Index (ACI), Acoustic Diversity Index (ADI), normalized Spectral Entropy ($H$), snapping shrimp band energy (2–20 kHz), and fish chorus energy (50–1000 Hz). |

---

## 📡 Live Data Sources & Production Ingestion

1. **Satellite SST & Thermal Stress**:
   - Primary: **NOAA Coral Reef Watch (CRW) 5km Daily via ERDDAP**.
   - Fields: Sea Surface Temperature (`sst`), SST Anomaly (`ssta`), Degree Heating Weeks (`dhw`), and Bleaching Alert Levels (Watch, Warning, Alert 1, Alert 2).
   - Resiliency: Automated retry with exponential backoff and explicit `status="no_data"` / `status="source_offline"` reporting (no hardcoded or fabricated mock values).

2. **Underwater Hydrophone Acoustics**:
   - Ingestion of raw `.wav` hydrophone audio recordings (SoundTrap, Ocean Sonics, Cetacean Research).
   - Computes normalized spectral entropy $H = -\frac{\sum p_i \log_2 p_i}{\log_2 N}$, ACI, BI, NDSI, and 13 MFCC coefficients per recording.
   - Degraded/bleached soundscapes reflect biophonic quietude; healthy soundscapes exhibit dense snapping shrimp clicks and active fish choruses.

3. **Citizen Diver Science**:
   - Ingestion of field survey records from **Reef Check**, **CoralWatch**, and **Reef Life Survey**.
   - Live submission via `POST /reports` with automated geospatial matching to nearest reef sites (Haversine formula).
   - Automated trust score algorithm evaluating observer credentials (marine biologist, dive master, recreational diver), photo proof, survey program, and photosynthetic depth plausibility ($2\text{m} \le \text{depth} \le 35\text{m}$).

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and adjust database/cache credentials if using PostgreSQL or Redis:
```bash
cp .env.example .env
```
Default configuration runs zero-config SQLite (`data/coralguard.db`) and in-memory TTL caching.

### 3. Run Database Migrations (Alembic)
```bash
# Check migration head
python -m alembic -c backend/alembic.ini heads

# Upgrade database to head
python -m alembic -c backend/alembic.ini upgrade head
```

### 4. Seed Database with Initial Reef Sites & Historical Data
```bash
python -m backend.scripts.seed_sites
```
Populates 15 monitored reef sites across the Great Barrier Reef, Caribbean, and Coral Triangle with 900+ daily SST records, hydrophone recordings, and verified diver surveys.

### 5. Launch FastAPI Backend Server
```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation is accessible at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status, database connectivity, and scheduler state |
| `GET` | `/sites` | Catalog of all 15 monitored reef sites with GPS coordinates & depths |
| `GET` | `/risk/{site_id}` | Real-time multimodal bleaching risk probabilities (`p_none`, `p_mild`, `p_severe`), operational alert tier, and modality breakdown for 14d or 28d lead time |
| `GET` | `/sst/{site_id}` | Trailing daily satellite SST, anomaly, DHW, and alert levels |
| `GET` | `/acoustics/{site_id}` | Hydrophone soundscape recordings, ACI, spectral entropy, and band energy ratios |
| `GET` | `/alerts` | Sites actively triggering Watch, Warning, or Alert thresholds |
| `POST` | `/reports` | Submit field diver survey report with geospatial matching and automated trust score |
| `GET` | `/metrics` | PR-AUC, recall, lead-time accuracy, confusion matrix, and FAR benchmarks |

---

## 🧪 Running Unit Tests

Execute the feature extraction and API route test suites:
```bash
# Test ecoacoustic extraction, trust scoring, and feature aggregation
python -m unittest tests/test_backend_features.py

# Test FastAPI endpoints and route handlers
python -m unittest tests/test_backend_api.py

# Run all tests together
python -m unittest tests/test_backend_features.py tests/test_backend_api.py
```

