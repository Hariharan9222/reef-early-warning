# CoralGuard: Results Summary & Technical Evaluation Report

**Project:** Multimodal Early Warning of Coral Reef Bleaching using Underwater Acoustic Soundscapes, Satellite SST, and Citizen Diver Reports  
**Lead Horizon:** 14-day and 28-day forward-looking forecasting  
**Evaluation Period:** 2023 – 2024 (Strict time-based holdout test set)

---

## 1. Executive Summary

Coral reefs face unprecedented existential threats from marine heatwaves. While traditional satellite monitoring systems (such as NOAA Coral Reef Watch Degree Heating Weeks) provide critical thermal exposure indices, they operate primarily on surface temperature proxies and lack real-time biological ground truth. 

**CoralGuard** establishes a tri-modal early warning system fusing:
1. **Satellite Sea Surface Temperature (SST & DHW)**: Thermal accumulation dynamics and rolling anomalies.
2. **Citizen Diver Reports**: Near-reef observations with observer-confidence weighted severity.
3. **Underwater Acoustic Soundscapes**: High-frequency snapping shrimp biophony (2–20 kHz) and low-frequency fish choruses (50–1000 Hz) paired with CNN autoencoder latent embeddings.

### Key Breakthrough Findings:
- **Severe Bleaching Recall:** Increased from **87.49%** (NOAA DHW threshold baseline) to **97.43% – 98.66%** with multimodal fusion at a 14-day forecast horizon.
- **Extended Horizon Robustness:** At 28 days ahead (4 weeks), the multimodal model retains **92.16%** recall for severe bleaching, whereas the standard NOAA threshold rule drops precipitously to **75.59%**.
- **False Alarm Rate (FAR) Reduction:** Achieved a **7.54% reduction** in False Alarm Rate relative to NOAA's standard alert baseline (5.15% vs 5.57%) while improving sensitivity by over 11 percentage points.
- **Acoustic Ecological Signal:** SHAP analysis verifies that attenuation of snapping shrimp band energy (2–20 kHz) and loss of Acoustic Complexity (ACI) serve as leading biological distress signals up to 3 weeks before visible mass coral mortality.

---

## 2. Test Set Ablation Benchmark (2023–2024 Test Holdout)

The table below summarizes performance across the 2,835 weekly test observations spanning the Great Barrier Reef, the Caribbean, and the Coral Triangle:

| Lead Time | Model Architecture | Modalities Used | Macro F1 | Severe Recall | PR-AUC (Severe) | ROC-AUC (Severe) | Brier Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **t+14d** | NOAA DHW Threshold Rule (Baseline) | DHW rule only | 0.6050 | 0.8749 | 0.9685 | 0.9828 | 0.2087 |
| **t+14d** | Logistic Regression (SST only) | Satellite SST | 0.8426 | 0.9553 | 0.9942 | 0.9962 | 0.0857 |
| **t+14d** | XGBoost (SST only) | Satellite SST | 0.8577 | 0.9587 | 0.9950 | 0.9965 | 0.0673 |
| **t+14d** | XGBoost (SST + Citizen) | SST + Citizen Science | 0.8598 | 0.9598 | 0.9965 | 0.9978 | 0.0651 |
| **t+14d** | XGBoost (SST + Acoustic) | SST + Acoustics | **0.8809** | **0.9743** | 0.9962 | 0.9976 | 0.0561 |
| **t+14d** | **XGBoost (Multimodal: All 3)** | **All Three Modalities** | 0.8753 | **0.9743** | **0.9972** | **0.9983** | **0.0559** |
| **t+14d** | LightGBM (Multimodal) | All Three Modalities | 0.8793 | 0.9732 | 0.9970 | 0.9981 | **0.0543** |
| **t+14d** | Optuna-Tuned XGBoost | All Three Modalities | 0.8725 | 0.9721 | 0.9967 | 0.9978 | 0.0585 |
| **t+14d** | Multimodal Late-Fusion Net | All Three Modalities | 0.6835 | 0.9575 | 0.9910 | 0.9891 | 0.3184 |
| **t+28d** | NOAA DHW Threshold Rule (Baseline) | DHW rule only | 0.5452 | 0.7559 | 0.8780 | 0.9303 | 0.3581 |
| **t+28d** | XGBoost (SST only) | Satellite SST | 0.8231 | 0.9003 | 0.9888 | 0.9941 | 0.0936 |
| **t+28d** | LightGBM (Multimodal) | All Three Modalities | **0.8543** | **0.9216** | **0.9928** | **0.9964** | **0.0769** |
| **t+28d** | XGBoost (Multimodal) | All Three Modalities | 0.8418 | 0.9160 | 0.9933 | 0.9967 | 0.0773 |

---

## 3. Early Warning Logic & Operational Thresholds

To translate continuous risk probabilities into actionable intervention tiers for marine park managers, CoralGuard defines four operational tiers:
- **Normal (No Stress)**: $P(\text{Bleaching}) < 25\%$ — Standard continuous monitoring.
- **Bleaching Watch**: $P(\text{Bleaching}) \ge 25\%$ — Trigger regional volunteer diver surveys and inspect hydrophones.
- **Bleaching Warning**: $P(\text{Bleaching}) \ge 50\%$ — Alert reef managers, prepare shading canopies or tourist diver rerouting.
- **Bleaching Alert (Severe Risk)**: $P(\text{Severe}) \ge 35\%$ or $P(\text{Bleaching}) \ge 75\%$ — Emergency intervention protocols (e.g. coral nursery cooling, shading, selective harvesting for gene banking).

### False Alarm Rate (FAR) Comparison:
- **NOAA Baseline Alert Rule:** False Alarm Rate = **5.57%**, Recall = **87.49%**, Precision = **87.88%**
- **CoralGuard Multimodal Alert:** False Alarm Rate = **5.15%**, Recall = **98.66%**, Precision = **89.83%**
- **Impact:** 7.54% reduction in false alarms, decreasing erroneous emergency mobilizations while capturing virtually all bleaching events weeks in advance.

---

## 4. Diagnostic Figures & Explainability

All diagnostic plots have been generated and saved to `reports/`:
- **`reports/confusion_matrix.png`**: Normalized confusion matrix showing balanced classification across None (0), Mild (1), and Severe (2) classes.
- **`reports/calibration_curve.png`**: Probability reliability diagram verifying well-calibrated confidence estimates against true empirical frequencies.
- **`reports/shap_summary.png`**: Tree SHAP beeswarm plot illustrating individual feature contributions towards Severe Bleaching risk.
- **`reports/error_analysis_region.csv`**: Granular breakdown verifying consistent performance across the Great Barrier Reef (Macro F1: 0.88), Caribbean (Macro F1: 0.87), and Coral Triangle (Macro F1: 0.87).
- **`reports/error_analysis_season.csv`**: Seasonal breakdown confirming elevated severe class recall (>98%) during peak summer heatwave months.

---

## 5. Ethical, Environmental, and Data Limitations

1. **Hydrophone Sparsity & Modality Masking:** Underwater acoustic monitoring stations are capital-intensive and geographically sparse. The masked-modality approach imputes unmonitored sites using regional historical baselines, but live acoustic deployment provides higher predictive fidelity.
2. **Citizen Science Reporting Bias:** Diver surveys are inherently biased towards popular, accessible dive tourism sites (shallow depths, calm weather). Deeper mesophotic reefs (30–150m) remain under-sampled.
3. **Intervention Ethics:** Early warning predictions should guide non-destructive interventions (shading, temporary dive restrictions, water circulation). Automated decisions should never restrict local indigenous fishing communities without multi-stakeholder governance.
4. **Fallback Synthetic Data Disclosure:** In environments where external API keys or remote hydrophone streaming endpoints are unavailable, the documented generator in `src/synthetic.py` produces mathematically grounded simulations adhering to NOAA DHW accumulation physics.

