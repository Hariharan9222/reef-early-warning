# CoralGuard Power BI Data Dictionary

This document details the schema, grain, data sources, and business definitions for all tables prepared for the Power BI Early Warning Dashboard.

---

## 1. `dim_site` (Reef Site Dimension)
- **Grain:** 1 row per monitored coral reef site / buoy station.
- **Primary Key:** `site_id` (Unique text identifier)
- **Relationships:** 1-to-many relationship with all fact tables.

| Column | Data Type | Sample Value | Description | Provenance |
|---|---|---|---|---|
| `site_id` | Text (PK) | `great_barrier_reef_-14.00_145.00` | Standard unique site coordinate code | Pipeline Registry |
| `site_name` | Text | `Great Barrier Reef Monitoring Station (-14.00°, +145.00°)` | Descriptive geographic name | Derived |
| `region` | Text | `Great Barrier Reef`, `Caribbean`, `Coral Triangle` | Global coral bioregion | Grounded |
| `country` | Text | `Australia`, `Belize`, `Indonesia`, `Philippines` | Sovereign jurisdiction / EEZ | Sovereign GIS |
| `latitude` | Decimal Number | `-14.0000` | WGS84 GPS latitude coordinate | Buoy / Sensor GPS |
| `longitude` | Decimal Number | `145.0000` | WGS84 GPS longitude coordinate | Buoy / Sensor GPS |
| `reef_type` | Text | `Barrier Reef`, `Coral Atoll`, `Fringing Reef` | Geomorphological reef structural classification | Marine Bio Survey |

---

## 2. `dim_date` (Standard Calendar Dimension)
- **Grain:** 1 row per calendar day (2015-01-01 to 2025-12-31).
- **Primary Key:** `date_id` (Integer formatted `YYYYMMDD`)
- **Date Table:** Marked as Date Table in Power BI via `date`.

| Column | Data Type | Sample Value | Description |
|---|---|---|---|
| `date_id` | Whole Number (PK) | `20240315` | Numeric date key for high-speed star-schema joins |
| `date` | Date | `2024-03-15` | Standard ISO-8601 calendar date |
| `year` | Whole Number | `2024` | Calendar year |
| `month` | Whole Number | `3` | Calendar month number (1-12) |
| `month_name` | Text | `March` | English calendar month name |
| `week` | Whole Number | `11` | Standard ISO week number (1-53) |
| `season` | Text | `Austral Summer (High Risk)` | Regional thermal risk season |

---

## 3. `fact_risk_predictions` (Multimodal Risk Forecast Fact)
- **Grain:** 1 row per `site_id` × `date_id` × `lead_time_days` (14d and 28d).
- **Foreign Keys:** `site_id` -> `dim_site[site_id]`, `date_id` -> `dim_date[date_id]`.

| Column | Data Type | Sample Value | Description | Provenance |
|---|---|---|---|---|
| `site_id` | Text (FK) | `caribbean_18.00_-60.00` | Monitored reef site | Real Pipeline |
| `date_id` | Whole Number (FK) | `20240401` | Forecast observation reference date | Real Pipeline |
| `date` | Date | `2024-04-01` | ISO observation date | Real Pipeline |
| `model_name` | Text | `XGBoost Multimodal Early Warning` | Algorithm generating probability | Model Artifact |
| `lead_time_days` | Whole Number | `14` or `28` | Early warning horizon ahead in days | Model Horizon |
| `pred_class` | Whole Number | `0` (None), `1` (Mild), `2` (Severe) | Predicted discrete bleaching category | ML Inference |
| `prob_none` | Decimal Number | `0.1245` | Model probability of No Bleaching | XGBoost Proba |
| `prob_mild` | Decimal Number | `0.2310` | Model probability of Mild Thermal Bleaching | XGBoost Proba |
| `prob_severe` | Decimal Number | `0.6445` | Model probability of Severe Mortality Risk | XGBoost Proba |
| `alert_level` | Text | `Bleaching Alert`, `Warning`, `Watch`, `Normal` | Operational early warning tier | Threshold Logic |
| `actual_class` | Whole Number | `2` | Verified satellite / diver ground-truth target | Ground Truth |

---

## 4. `fact_sst` (Satellite Sea Surface Temperature Fact)
- **Grain:** 1 row per `site_id` × `date_id` (daily / weekly satellite cycle).
- **Foreign Keys:** `site_id` -> `dim_site[site_id]`, `date_id` -> `dim_date[date_id]`.

| Column | Data Type | Sample Value | Description | Provenance |
|---|---|---|---|---|
| `site_id` | Text (FK) | `caribbean_18.00_-60.00` | Monitored reef site | Real Pipeline |
| `date_id` | Whole Number (FK) | `20240401` | Observation date key | Real Pipeline |
| `date` | Date | `2024-04-01` | ISO observation date | Real Pipeline |
| `sst` | Decimal Number | `29.45` | Sea Surface Temperature (°C) | NOAA ERDDAP |
| `ssta` | Decimal Number | `+1.35` | SST Anomaly relative to climatology (°C) | NOAA ERDDAP |
| `dhw` | Decimal Number | `6.80` | Degree Heating Weeks accumulation (°C-weeks) | NOAA CRW |
| `days_above_threshold` | Whole Number | `14` | Days SST exceeded bleaching threshold in past 30d | NOAA CRW |

---

## 5. `fact_acoustic` (Underwater Soundscape Ecoacoustics Fact)
- **Grain:** 1 row per `site_id` × `date_id`.
- **Foreign Keys:** `site_id` -> `dim_site[site_id]`, `date_id` -> `dim_date[date_id]`.

| Column | Data Type | Sample Value | Description | Provenance |
|---|---|---|---|---|
| `site_id` | Text (FK) | `caribbean_18.00_-60.00` | Monitored reef site | Real Pipeline |
| `date_id` | Whole Number (FK) | `20240401` | Observation date key | Real Pipeline |
| `date` | Date | `2024-04-01` | ISO observation date | Real Pipeline |
| `aci` | Decimal Number | `12.450` | Acoustic Complexity Index (Pieretti et al.) | librosa feature |
| `adi` | Decimal Number | `1.845` | Acoustic Diversity Index across bands | librosa feature |
| `shrimp_band_energy` | Decimal Number | `0.2850` | Snapping shrimp band energy ratio (2 - 20 kHz) | Band pass FFT |
| `fish_band_energy` | Decimal Number | `0.0920` | Fish chorus biophonic energy ratio (50 - 1000 Hz) | Band pass FFT |
| `acoustic_health_index` | Decimal Number | `78.5` | Composite ecological biophony score [0 - 100] | Synthetic Calibration |

---

## 6. `fact_diver_reports` (Citizen Diver Field Observations Fact)
- **Grain:** 1 row per validated diver submission.
- **Foreign Keys:** `site_id` -> `dim_site[site_id]`, `date_id` -> `dim_date[date_id]`.

| Column | Data Type | Sample Value | Description | Provenance |
|---|---|---|---|---|
| `report_id` | Text (PK) | `REP_1042` | Unique diver survey identifier | Reef Check / Direct |
| `site_id` | Text (FK) | `caribbean_18.00_-60.00` | Closest monitored reef station | Haversine Match |
| `date_id` | Whole Number (FK) | `20240328` | Survey dive date key | Diver Log |
| `date` | Date | `2024-03-28` | ISO survey dive date | Diver Log |
| `lat` | Decimal Number | `18.0420` | Exact diver survey latitude | GPS / Boat log |
| `lon` | Decimal Number | `-60.0150` | Exact diver survey longitude | GPS / Boat log |
| `depth` | Decimal Number | `12.5` | Dive transect depth (meters) | Dive Computer |
| `bleaching_percent` | Decimal Number | `45.0` | Observed coral bleaching colony coverage (%) | Visual Transect |
| `severity_class` | Text | `Mild (10-50%)` | Standard survey severity classification | Survey Protocol |
| `observer_confidence` | Decimal Number | `0.85` | Diver certification confidence score (0.0 - 1.0) | Automated Scorer |
| `source` | Text | `Citizen Diver Science` | Sponsoring program (Reef Check, CoralWatch) | Program Registry |

---

## 7. `model_metrics` (ML Benchmark & Ablation Evaluation)
- **Grain:** 1 row per model architecture × ablation set × lead horizon.

| Column | Data Type | Sample Value | Description |
|---|---|---|---|
| `model_name` | Text | `XGBoost (Multimodal: All 3)` | Machine learning algorithm tested |
| `ablation_set` | Text | `SST + Citizen + Acoustic` | Multi-modal feature subset |
| `lead_time` | Text | `t+14d`, `t+28d` | Early warning horizon evaluated |
| `macro_f1` | Decimal Number | `0.8767` | Macro-averaged F1 score on test holdout |
| `pr_auc` | Decimal Number | `0.9971` | Precision-Recall AUC for Severe Bleaching |
| `roc_auc` | Decimal Number | `0.9983` | Receiver Operating Characteristic AUC |
| `severe_recall` | Decimal Number | `0.9721` | Recall on severe mortality events |
| `brier` | Decimal Number | `0.0558` | Probability calibration Brier score (lower is better) |
| `false_alarm_rate` | Decimal Number | `0.0515` | False Alarm Rate (7.5% reduction vs NOAA DHW baseline) |

---

## 8. `feature_importance` (Model Explainability & Modality Attribution)
- **Grain:** 1 row per feature in the production model.

| Column | Data Type | Sample Value | Description |
|---|---|---|---|
| `model_name` | Text | `XGBoost Multimodal (t+14d)` | Model identifier |
| `feature` | Text | `dhw` | Multi-modal feature name |
| `shap_importance` | Decimal Number | `0.15483` | Normalized feature importance / mean |SHAP| contribution |
| `modality` | Text | `Satellite SST`, `Citizen Diver Science`, `Underwater Acoustics` | Input data sensor modality |
