# CoralGuard Power BI Report Layout & Visuals Specification

Standard Canvas Dimensions: **1280 × 720 px (16:9)**  
Theme: `powerbi/coral_theme.json`  
Global Interactivity: **Cross-filtering enabled** across all visuals; **Drill-through** enabled from any visual with `dim_site[site_id]` to a dedicated Site Detail view.

---

## Page 1: 🗺️ Executive Overview & Early Warning

### Purpose
Executive mission control displaying real-time bleaching risk across global coral biomes, operational alert tiers, and spatial hotspot distribution.

### Visuals Grid Specification

| Visual # | Visual Type | Title | Data Fields & Measures | Position (X, Y) | Dimensions (W × H) | Formatting Notes |
|---|---|---|---|---|---|---|
| **V1.1** | Slicer (Dropdown) | Select Region | `dim_site[region]` | `(20, 20)` | `200 × 50` | Search enabled, single or multi-select |
| **V1.2** | Slicer (Pill/Tiles) | Forecast Horizon | `fact_risk_predictions[lead_time_days]` | `(230, 20)` | `220 × 50` | Horizontal tiles: "14 Days", "28 Days" |
| **V1.3** | Slicer (Between) | Date Window | `dim_date[date]` | `(460, 20)` | `280 × 50` | Default to trailing 12 months |
| **V1.4** | Card (KPI) | 🚨 Sites at Alert | Measure: `[Sites at Alert]` | `(20, 80)` | `230 × 90` | Callout font: 26pt bold, Red border `#EF4444` |
| **V1.5** | Card (KPI) | ⚠️ Sites at Warning | Measure: `[Sites at Warning]` | `(260, 80)` | `230 × 90` | Callout font: 26pt bold, Orange `#F97316` |
| **V1.6** | Card (KPI) | 🔥 Max DHW | Measure: `[Max DHW]` | `(500, 80)` | `230 × 90` | Format: `0.0 °C-wks`, Amber `#D97706` |
| **V1.7** | Card (KPI) | 🌊 Mean Bleaching Risk | Measure: `[Avg Bleaching Risk]` | `(740, 80)` | `230 × 90` | Format: `0.0%`, dynamic background |
| **V1.8** | Card (KPI) | 🌿 Unstressed Sites | Measure: `[Unstressed Sites]` | `(980, 80)` | `280 × 90` | Callout font: 26pt bold, Emerald `#10B981` |
| **V1.9** | Map (Bubble Map) | Global Reef Bleaching Early Warning Map | **Location:** `dim_site[latitude]`, `dim_site[longitude]`<br>**Size:** `[Avg Bleaching Risk]`<br>**Color / Legend:** `fact_risk_predictions[alert_level]`<br>**Tooltips:** `dim_site[site_name]`, `dim_site[country]`, `[Max DHW]`, `[Acoustic Health Index (avg)]` | `(20, 180)` | `740 × 520` | Map style: Light / Grayscale. Legend colors: Alert (Red), Warning (Orange), Watch (Yellow), Normal (Green) |
| **V1.10** | Line Chart | Multimodal Bleaching Risk Trajectory Over Time | **X-axis:** `dim_date[date]` (Continuous)<br>**Y-axis:** `[Avg Bleaching Risk]`, `[Avg Total Bleaching Probability]`<br>**Constant Line:** `Y = 0.35` (Severe Alert Level) | `(770, 180)` | `490 × 290` | Stroke width: 2.5px. Red alert reference line dotted. |
| **V1.11** | Matrix Table | Monitored Reef Sites Risk Summary | **Rows:** `dim_site[site_name]`, `dim_site[region]`<br>**Values:** `[Avg Bleaching Risk]`, `[Max DHW]`, `[Acoustic Health Index (avg)]`, `[Diver Reports Count]` | `(770, 480)` | `490 × 220` | Conditional formatting data bars on `[Avg Bleaching Risk]` |

---

## Page 2: 🛰️ Satellite SST & Thermal Stress Dynamics

### Purpose
In-depth diagnostic of NOAA Coral Reef Watch Degree Heating Weeks (DHW), SST anomalies, and thermal threshold exceedance.

### Visuals Grid Specification

| Visual # | Visual Type | Title | Data Fields & Measures | Position (X, Y) | Dimensions (W × H) | Formatting Notes |
|---|---|---|---|---|---|---|
| **V2.1** | Slicer | Reef Site Selection | `dim_site[site_name]` | `(20, 20)` | `350 × 50` | Search enabled dropdown |
| **V2.2** | Card (KPI) | Peak DHW Accumulation | Measure: `[Max DHW]` | `(390, 20)` | `270 × 70` | Text: Amber `#D97706` |
| **V2.3** | Card (KPI) | Mean SST Anomaly (SSTA) | Measure: `[Avg SSTA]` | `(680, 20)` | `270 × 70` | Format: `+0.00 °C` |
| **V2.4** | Card (KPI) | Days Above Threshold | Measure: `[Days Above Threshold]` | `(970, 20)` | `290 × 70` | Format: `0 days in 30d` |
| **V2.5** | Line & Clustered Column Chart | Thermal Stress Evolution: SST vs DHW | **X-axis:** `dim_date[date]`<br>**Column Y-axis:** `fact_sst[dhw]` (Amber)<br>**Line Y-axis:** `fact_sst[sst]` (Blue), `fact_sst[ssta]` (Cyan)<br>**Threshold Line:** `Y = 4.0` (DHW Bleaching Threshold) | `(20, 100)` | `850 × 330` | Dual Y-axis: Left SST (°C), Right DHW (°C-weeks) |
| **V2.6** | Scatter Chart | Thermal Dose vs Forecasted Bleaching Risk | **X-axis:** `fact_sst[dhw]`<br>**Y-axis:** `fact_risk_predictions[prob_severe]`<br>**Details:** `dim_site[site_name]`<br>**Color:** `dim_site[region]` | `(880, 100)` | `380 × 330` | Bubble transparency 30%. Displays non-linear logistic tipping point. |
| **V2.7** | Matrix (Heatmap) | Monthly DHW Heatmap by Site | **Rows:** `dim_site[site_name]`<br>**Columns:** `dim_date[month_name]` (sorted by month)<br>**Values:** `[Max DHW]` | `(20, 440)` | `1240 × 260` | Background color scale: White -> Yellow -> Red (`#FFFFFF` to `#EF4444`) |

---

## Page 3: 🎧 Acoustic Soundscape & Biophony Health

### Purpose
Underwater ecoacoustic telemetry analysis showing snapping shrimp attenuation (2–20 kHz), fish chorus dynamics (50–1000 Hz), and acoustic quieting.

### Visuals Grid Specification

| Visual # | Visual Type | Title | Data Fields & Measures | Position (X, Y) | Dimensions (W × H) | Formatting Notes |
|---|---|---|---|---|---|---|
| **V3.1** | Card (KPI) | Mean Acoustic Health Index | Measure: `[Acoustic Health Index (avg)]` | `(20, 20)` | `380 × 70` | Gauge scale: 0 to 100. Emerald `#10B981` |
| **V3.2** | Card (KPI) | Snapping Shrimp Energy Ratio | Measure: `[Snapping Shrimp Band Energy %]` | `(420, 20)` | `420 × 70` | Healthy benchmark > 20% |
| **V3.3** | Card (KPI) | Fish Chorus Energy Ratio | Measure: `[Fish Chorus Band Energy %]` | `(860, 20)` | `400 × 70` | Biophony benchmark > 8% |
| **V3.4** | Line Chart | Bioacoustic Soundscape Health Index Trend | **X-axis:** `dim_date[date]`<br>**Y-axis:** `[Acoustic Health Index (avg)]`<br>**Legend:** `dim_site[region]` | `(20, 100)` | `780 × 320` | Smooth spline line curves. Shaded area below 40 (Degraded status). |
| **V3.5** | Scatter Chart | Biophonic Partitioning: Fish Chorus vs Snapping Shrimp | **X-axis:** `fact_acoustic[fish_band_energy]`<br>**Y-axis:** `fact_acoustic[shrimp_band_energy]`<br>**Color Legend:** `fact_risk_predictions[alert_level]` | `(810, 100)` | `450 × 320` | Quadrant visualization: Top-Right = Healthy Reef; Bottom-Left = Silenced Degraded Reef |
| **V3.6** | Clustered Column Chart | Acoustic Complexity (ACI) vs Acoustic Diversity (ADI) | **X-axis:** `dim_site[region]`<br>**Values:** `[Acoustic Complexity Index (avg)]`, `AVERAGE(fact_acoustic[adi])` | `(20, 430)` | `600 × 270` | Direct comparison across global reef provinces |
| **V3.7** | Donut Chart | Acoustic Reef Condition Distribution | **Category:** Condition ("Healthy Biophony", "Stressed", "Acoustically Silenced")<br>**Values:** Count of records | `(630, 430)` | `630 × 270` | Colors: Emerald `#10B981`, Amber `#EAB308`, Red `#EF4444` |

---

## Page 4: 🤿 Citizen Diver Science Ground Truth

### Purpose
Community diver reports integration (Reef Check, CoralWatch) providing verified ground observations and observer confidence scoring.

### Visuals Grid Specification

| Visual # | Visual Type | Title | Data Fields & Measures | Position (X, Y) | Dimensions (W × H) | Formatting Notes |
|---|---|---|---|---|---|---|
| **V4.1** | Slicer | Observer Severity Filter | `fact_diver_reports[severity_class]` | `(20, 20)` | `280 × 50` | Multi-select checkboxes |
| **V4.2** | Card (KPI) | Total Submitted Diver Surveys | Measure: `[Diver Reports Count]` | `(320, 20)` | `300 × 70` | Format: `#,##0` |
| **V4.3** | Card (KPI) | Mean Reported Bleaching | Measure: `[Avg Reported Bleaching %]` | `(640, 20)` | `300 × 70` | Format: `0.0%` |
| **V4.4** | Card (KPI) | Mean Observer Credential Score | Measure: `[Avg Observer Confidence]` | `(960, 20)` | `300 × 70` | Benchmark > 0.80 |
| **V4.5** | Map | Citizen Diver Survey Observations | **Location:** `fact_diver_reports[lat]`, `fact_diver_reports[lon]`<br>**Size:** `fact_diver_reports[bleaching_percent]`<br>**Color:** `fact_diver_reports[severity_class]`<br>**Tooltips:** `fact_diver_reports[report_id]`, `fact_diver_reports[depth]`, `fact_diver_reports[source]` | `(20, 100)` | `740 × 350` | Satellite or Light basemap |
| **V4.6** | Stacked Column Chart | Monthly Diver Survey Frequency by Severity | **X-axis:** `dim_date[month_name]`<br>**Y-axis:** `COUNT(fact_diver_reports[report_id])`<br>**Legend:** `fact_diver_reports[severity_class]` | `(780, 100)` | `480 × 350` | Stacks: None (Green), Mild (Yellow), Severe (Red) |
| **V4.7** | Table | Validated Field Diver Reports Log | **Columns:** `fact_diver_reports[report_id]`, `fact_diver_reports[date]`, `dim_site[site_name]`, `fact_diver_reports[depth]`, `fact_diver_reports[bleaching_percent]`, `fact_diver_reports[severity_class]`, `fact_diver_reports[observer_confidence]` | `(20, 460)` | `1240 × 240` | Alternate row shading, sortable by date |

---

## Page 5: 🧠 ML Model Performance & Modality Ablation

### Purpose
Machine learning explainability, modality ablation comparison, and SHAP feature attribution benchmarking the multimodal early warning system.

### Visuals Grid Specification

| Visual # | Visual Type | Title | Data Fields & Measures | Position (X, Y) | Dimensions (W × H) | Formatting Notes |
|---|---|---|---|---|---|---|
| **V5.1** | Card (KPI) | Severe Bleaching Recall (t+14d) | Measure: `[Model Severe Bleaching Recall]` | `(20, 20)` | `300 × 70` | Value: `97.2%`, Subtitle: "NOAA Baseline: 87.5%" |
| **V5.2** | Card (KPI) | Precision-Recall AUC (PR-AUC) | Measure: `[Model PR-AUC (Severe)]` | `(340, 20)` | `300 × 70` | Value: `0.9971`, Near-perfect discrimination |
| **V5.3** | Card (KPI) | Multimodal Macro F1 | Measure: `[Model Macro F1 Score]` | `(660, 20)` | `300 × 70` | Value: `0.8767`, +44.9% vs DHW rule |
| **V5.4** | Card (KPI) | False Alarm Rate (FAR) Reduction | Static Callout: `-7.54%` | Static Text: `-7.5% False Alarms` | `(980, 20)` | `280 × 70` | Emerald `#10B981` |
| **V5.5** | Clustered Bar Chart | Multimodal Ablation Study: Severe Recall by Modality | **Y-axis:** `model_metrics[ablation_set]`<br>**X-axis:** `model_metrics[severe_recall]`<br>**Legend:** `model_metrics[lead_time]` | `(20, 100)` | `620 × 330` | Highlights progressive gain from SST -> SST+Citizen -> SST+Acoustic -> Multimodal All 3 |
| **V5.6** | Clustered Bar Chart | Modality Ablation: Macro F1 Performance | **Y-axis:** `model_metrics[ablation_set]`<br>**X-axis:** `model_metrics[macro_f1]`<br>**Legend:** `model_metrics[lead_time]` | `(650, 100)` | `610 × 330` | Direct comparison against baseline rules |
| **V5.7** | Horizontal Bar Chart | SHAP Feature Importance by Sensor Modality | **Y-axis:** `feature_importance[feature]` (Top 15)<br>**X-axis:** `feature_importance[shap_importance]`<br>**Color Legend:** `feature_importance[modality]` | `(20, 440)` | `740 × 260` | Modality colors: Satellite SST (Cyan), Diver Science (Orange), Acoustics (Green) |
| **V5.8** | Matrix Table | Model Evaluation Benchmark Matrix | **Rows:** `model_metrics[model_name]`, `model_metrics[lead_time]`<br>**Values:** `macro_f1`, `pr_auc`, `roc_auc`, `severe_recall`, `brier`, `false_alarm_rate` | `(780, 440)` | `480 × 260` | Number formatting: `0.0000` |

